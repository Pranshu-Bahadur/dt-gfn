# src/trainer.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Deque
from collections import deque
import copy
import math
import random

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from src.tokenizer import Tokenizer, Vocab
from src.env import TabularEnv
from src.policy import PolicyPaperMLP, PolicyTransformer
from src.utils import (
    ReplayBuffer,
    tb_loss,
    fl_loss,
    _safe_sample,
    get_tree_predictor,
    deltaE_split_gain_regression,
    deltaE_split_gain_classification,
    calculate_bayesian_reward,
    create_gain_bias,
    calculate_bayesian_reward_regression
)

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    feature_cols: List[str]
    target_col: str = "target"
    task: str = "classification"                   # "classification" | "regression"
    reward_function: str = "bayesian"              # "bayesian" | "gini" | "variance"
    n_classes: Optional[int] = None
    n_bins: int = 255
    binning_strategy: str = "global_uniform"
    device: str = "cuda"

    # training mode
    random_forest: bool = True

    # GFN Training / Boosting
    updates: int = 50
    rollouts: int = 60
    batch_size: int = 8192
    max_depth: int = 7
    top_k_trees: int = 10
    boosting_lr: float = 0.1
    redundancy_aware: bool = False

    # Policy network
    lstm_hidden: int = 256
    mlp_layers: int = 3
    mlp_width: int = 256
    lr: float = 1e-4

    # Priors & annealing
    beta: Optional[float] = None
    prior_scale: float = 0.5

    # Parallelism
    num_parallel: int = 10

    # Inference
    policy_inference_trees: int = 500

    # ---- NEW: memory/throughput knobs ----
    amp: bool = True                               # mixed precision for policy
    eval_on_cpu: bool = True                       # run metric preds on CPU tensors
    metric_sample_size: int = 20000                # subsample rows for per-update metrics
    eval_batch_size: int = 16384                   # minibatch size for prediction passes


# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer: Optional[Tokenizer] = None
        self.ensemble: list[list[int]] = []
        self.boosting_ensemble: list = []
        self.y_mean: float = 0.0

        self.pf = None
        self.pb = None
        self.log_z: Optional[torch.Tensor] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.le: Optional[LabelEncoder] = None
        self.scaler = GradScaler(enabled=cfg.amp)

    # -------------------------
    # Fit
    # -------------------------
    def fit(self, df_train: pd.DataFrame) -> "Trainer":
        c = self.cfg

        v = Vocab(len(c.feature_cols), c.n_bins, 1)
        self.tokenizer = Tokenizer(v)

        env_template = TabularEnv(
            df_train,
            feature_cols=c.feature_cols,
            target_col=c.target_col,
            n_bins=c.n_bins,
            task=c.task,
            binning_strategy=c.binning_strategy,
            device=c.device,            # the env stores tensors on device, but we’ll move copies to CPU when needed
        )
        if c.task == "classification":
            self.le, c.n_classes = env_template.le, env_template.n_classes

        y_true = env_template.y_full.clone()
        X_binned = env_template.X_full.clone()

        # policy nets (keep on device)
        self.pf = torch.jit.script(PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device))
        self.pb = torch.jit.script(PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device))
        self.log_z = torch.nn.Parameter(torch.tensor(1.0, device=c.device))

        optim_pfs = torch.optim.AdamW(self.pf.parameters(), lr=c.lr)
        optim_pbs = torch.optim.AdamW(self.pb.parameters(), lr=c.lr)
        optim_z   = torch.optim.Adam([self.log_z], lr=c.lr / 10)

        warmup, tmax = 10, max(1, c.updates - 10)
        schedulers = [
            SequentialLR(optim_pfs, [LambdaLR(optim_pfs, lambda u: min(1.0, u / warmup)), CosineAnnealingLR(optim_pfs, T_max=tmax)], milestones=[warmup]),
            SequentialLR(optim_pbs, [LambdaLR(optim_pbs, lambda u: min(1.0, u / warmup)), CosineAnnealingLR(optim_pbs, T_max=tmax)], milestones=[warmup]),
            SequentialLR(optim_z,   [LambdaLR(optim_z,   lambda u: min(1.0, u / warmup)), CosineAnnealingLR(optim_z,   T_max=tmax)], milestones=[warmup]),
        ]
        optimizers = [optim_pfs, optim_pbs, optim_z]

        self.replay_buffer = ReplayBuffer(capacity=100)

        if c.beta is None:
            c.beta = math.log(4) + math.log(len(c.feature_cols))
            tqdm.write(f"[trainer] β (structure prior) = {c.beta:.4f}")

        if c.random_forest:
            self._fit_dt_gfn_random_forest(env_template, y_true, X_binned, optimizers, schedulers)
        else:
            self._fit_boost_gfn(env_template, y_true, X_binned, optimizers, schedulers)

        return self

    # ----------------------------
    # Reward wrapper (per sequence)
    # ----------------------------
    def _calculate_reward(self, tok: torch.Tensor, reward_env: TabularEnv) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        c = self.cfg
        if c.reward_function == 'bayesian':
            fn = calculate_bayesian_reward if c.task == 'classification' else calculate_bayesian_reward_regression
            R_t = fn(tok, self.tokenizer, reward_env, c.beta)
            return R_t, None
        else:
            fn = deltaE_split_gain_classification if (c.task == 'classification' and c.reward_function == 'gini') else deltaE_split_gain_regression
            dR = fn(tok, self.tokenizer, reward_env)                # [1, T-1]
            R_t = torch.clamp(dR.sum(), min=1e-9)                   # scalar
            return R_t, dR

    # --------------------------------
    # Batched policy update (one step)
    # --------------------------------
    def _update_policy(self, all_tuples_with_targets: List, env_template: TabularEnv, optimizers: List) -> Tuple[float, float]:
        if not all_tuples_with_targets: return 0.0, 0.0
        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device

        seqs, priors, targets = zip(*all_tuples_with_targets)
        toks = [torch.tensor(s, device=device, dtype=torch.long) for s in seqs]
        padded = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=v.PAD)
        flipped = torch.nn.utils.rnn.pad_sequence([t.flip(0) for t in toks], batch_first=True, padding_value=v.PAD)
        priors_tensor = torch.as_tensor(priors, device=device, dtype=torch.float32)

        # forward
        for opt in optimizers: opt.zero_grad(set_to_none=True)

        with autocast(enabled=c.amp):
            log_pf = self.pf.log_prob(padded)
            log_pb = self.pb.log_prob(flipped)
            logF   = self.pf.log_F(padded)

            # rewards
            reward_env = copy.copy(env_template)
            R_list, dR_list = [], []
            for i, t in enumerate(toks):
                tok_i = padded[i:i+1, :t.numel()]
                target = targets[i]
                reward_env.y = target
                reward_env.reset(len(target))
                R_t, dR = self._calculate_reward(tok_i, reward_env)
                R_list.append(R_t.squeeze())
                if dR is not None: dR_list.append(dR.squeeze(0))

            R = torch.stack(R_list, dim=0)

            if self.cfg.reward_function == 'bayesian':
                l_tb = tb_loss(log_pf, log_pb, self.log_z, R, priors_tensor)
                loss = l_tb
                tb_val, fl_val = l_tb, None
            else:
                # shape Δreward so that sum == log R
                logR = torch.log(R + 1e-9)
                gains = torch.nn.utils.rnn.pad_sequence(dR_list, batch_first=True, padding_value=0.0)
                gains = torch.relu(gains)
                gsum = gains.sum(1, keepdim=True).clamp_min(1e-9)
                dR_shaped = gains * (logR.unsqueeze(1) / gsum)

                Tm1 = log_pf.size(1)
                if dR_shaped.size(1) < Tm1:
                    pad = torch.zeros((dR_shaped.size(0), Tm1 - dR_shaped.size(1)), device=device)
                    dR_shaped = torch.cat([dR_shaped, pad], dim=1)
                elif dR_shaped.size(1) > Tm1:
                    dR_shaped = dR_shaped[:, :Tm1]

                l_tb = tb_loss(log_pf, log_pb, self.log_z, R, priors_tensor)
                l_fl = fl_loss(logF, log_pf, log_pb, dR_shaped)
                loss = l_tb + l_fl
                tb_val, fl_val = l_tb, l_fl

        # backward
        self.scaler.scale(loss).backward()
        for opt in optimizers:
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
            self.scaler.step(opt)
        self.scaler.update()

        self.replay_buffer.mark_policy_update()

        tb_loss_acc = float(tb_val.item())
        fl_loss_acc = float(fl_val.item()) if fl_val is not None else 0.0
        return tb_loss_acc, fl_loss_acc

    # --------------------------
    # RF training (policy + gen)
    # --------------------------
    def _fit_dt_gfn_random_forest(self, env_template, y_true, X_binned, optimizers, schedulers):
        c = self.cfg
        tqdm.write("--- Starting DT-GFN (Random Forest) Training ---")

        y_target_for_reward = (
            torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float)
            if c.task == "classification" else y_true.clone()
        )
        env_template.y = y_true.clone()

        # CPU mirror for metric evaluation to save VRAM
        X_for_metric = X_binned.cpu() if c.eval_on_cpu else X_binned
        y_for_metric = y_true.cpu() if c.eval_on_cpu else y_true

        N = X_for_metric.size(0)
        metric_idx = torch.arange(N) if c.metric_sample_size <= 0 else torch.randperm(N)[:min(c.metric_sample_size, N)]
        Xm = X_for_metric[metric_idx]
        ym = y_for_metric[metric_idx]
        Ytarget_metric = (
            torch.nn.functional.one_hot(ym, num_classes=c.n_classes).to(torch.float)
            if c.task == "classification" else ym
        )

        for upd in tqdm(range(1, c.updates + 1), desc="Policy Training & Tree Generation"):
            forward_tuples = self._collect_rollouts(env_template, temp=0.0, residuals=y_true, beta=c.beta)
            replay_tuples  = self.sample_replay(c.top_k_trees)
            all_tuples     = forward_tuples + replay_tuples
            if not all_tuples:
                for sch in schedulers: sch.step()
                continue

            all_tuples_with_targets = [(seq, prior, y_target_for_reward) for seq, prior in all_tuples]
            avg_tb_loss, avg_fl_loss = self._update_policy(all_tuples_with_targets, env_template, optimizers)
            for sch in schedulers: sch.step()

            # ---- memory-safe metrics on subsample (CPU, running mean) ----
            trees = [seq for seq, _ in all_tuples if seq]
            avg_pred = None
            with torch.no_grad():
                for seq in trees:
                    pred_fn = get_tree_predictor(seq, X_binned.cpu() if c.eval_on_cpu else X_binned, Ytarget_metric, self.tokenizer)
                    # mini-batch over Xm
                    running = self._predict_in_batches(pred_fn, Xm, c.eval_batch_size, device="cpu" if c.eval_on_cpu else c.device)
                    avg_pred = running if avg_pred is None else (avg_pred + running)
                if avg_pred is not None:
                    avg_pred = avg_pred / max(1, len(trees))

            log_str = f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f} | Trees: {len(trees)}"
            if avg_pred is not None:
                if c.task == "classification":
                    acc = (avg_pred.argmax(1).cpu() == ym.cpu()).float().mean().item()
                    log_str += f" | Train Acc@sub: {acc:.4f}"
                else:
                    if avg_pred.std() > 0 and ym.std() > 0:
                        corr = torch.corrcoef(torch.stack([avg_pred.squeeze().cpu(), ym.squeeze().cpu()]))[0, 1].item()
                        log_str += f" | Train Corr@sub: {corr:.4f}"

            tqdm.write(log_str)

        self.ensemble = [seq for seq, _ in all_tuples if seq] if 'all_tuples' in locals() else []
        tqdm.write(f"--- RF finished. Final forest size: {len(self.ensemble)} ---")

    # --------------------------
    # Boosting training (unchanged algorithm; memory-safe eval)
    # --------------------------
    def _fit_boost_gfn(self, env_template, y_true, X_binned, optimizers, schedulers):
        c = self.cfg
        tqdm.write("--- Starting Boost-GFN Training ---")

        if c.task == "classification":
            class_counts = torch.bincount(y_true, minlength=c.n_classes).float()
            class_probs  = class_counts / class_counts.sum()
            base_pred = torch.log(class_probs + 1e-9).unsqueeze(0).repeat(len(y_true), 1)
        else:
            self.y_mean = y_true.mean().item()
            base_pred = torch.full_like(y_true, self.y_mean, dtype=torch.float32)

        for upd in tqdm(range(1, c.updates + 1), desc="Boost Updates"):
            # reset baseline each round (as requested)
            if c.task == "classification":
                class_counts = torch.bincount(y_true, minlength=c.n_classes).float()
                class_probs  = class_counts / class_counts.sum()
                base_pred = torch.log(class_probs + 1e-9).unsqueeze(0).repeat(len(y_true), 1)
                residuals = torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float) - torch.softmax(base_pred, dim=1)
            else:
                base_pred = torch.full_like(y_true, self.y_mean, dtype=torch.float32)
                residuals = y_true - base_pred

            replay = self.sample_replay(c.top_k_trees)
            env_template.y = residuals.clone()
            fresh = self._collect_rollouts(env_template, 0.0, residuals, c.beta)
            candidates = replay + fresh
            if not candidates:
                for sch in schedulers: sch.step()
                continue

            tuples = []
            current_res = residuals.clone()
            for seq, prior in candidates:
                tuples.append((seq, prior, current_res.clone()))
                pred = get_tree_predictor(seq, X_binned, current_res, self.tokenizer)
                add_train = pred(X_binned)
                base_pred += c.boosting_lr * add_train
                if c.task == "classification":
                    probs = torch.softmax(base_pred, dim=1)
                    current_res = torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float) - probs
                else:
                    current_res = y_true - base_pred

            avg_tb_loss, avg_fl_loss = self._update_policy(tuples, env_template, optimizers)
            for sch in schedulers: sch.step()

            # light metric (no stacking)
            if c.task == "classification":
                acc = (base_pred.argmax(1) == y_true).float().mean().item()
                tqdm.write(f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f} | Acc: {acc:.4f}")
            else:
                if base_pred.std() > 0 and y_true.std() > 0:
                    corr = torch.corrcoef(torch.stack([base_pred.squeeze(), y_true.squeeze()]))[0, 1].item()
                    tqdm.write(f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f} | Corr: {corr:+.4f}")
                else:
                    tqdm.write(f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f} | Corr: nan")

    # --------------------------
    # Rollouts (policy)
    # --------------------------
    def _collect_rollouts(self, env_template, temp, residuals, beta):
        forward_tuples, done = [], 0
        ras_counts = {} if self.cfg.redundancy_aware else None

        with tqdm(total=self.cfg.rollouts, desc="Rollouts", leave=False) as pbar:
            while done < self.cfg.rollouts:
                if ras_counts is not None: ras_counts.clear()
                batch = min(self.cfg.num_parallel, self.cfg.rollouts - done)
                envs = [copy.copy(env_template) for _ in range(batch)]
                # keep idxs on CPU in buffer to save VRAM
                results = self.batched_rollout(envs, temp, residuals, beta, ras_counts)
                for res in results:
                    if res:
                        seq, prior, idxs = res
                        self.replay_buffer.add(0.0, seq, prior, idxs.cpu())
                        forward_tuples.append((seq, prior))
                done += batch
                pbar.update(batch)
        return forward_tuples

    # --------------------------
    # Replay sampling
    # --------------------------
    def sample_replay(self, k: int, REFRESH_INTERVAL: int = 5) -> list[tuple[list[int], float]]:
        buf = self.replay_buffer
        if not buf or not buf.data: return []

        stale = [i for i, e in enumerate(buf.data) if e[4] is None or buf.step - e[5] >= REFRESH_INTERVAL]
        if stale:
            stale_seqs = [buf.data[i][1] for i in stale]
            with torch.no_grad():
                flipped = [torch.tensor(s, device=self.cfg.device).flip(0) for s in stale_seqs]
                padded = torch.nn.utils.rnn.pad_sequence(flipped, batch_first=True, padding_value=self.tokenizer.v.PAD)
                logp = self.pb.log_prob(padded)
                mask = (padded != self.tokenizer.v.PAD).float()
                T = min(mask.size(1), logp.size(1))
                w = (logp[:, :T] * mask[:, :T]).sum(1).exp()
                for i, wi in zip(stale, w):
                    r, t, p, idxs, _, _ = buf.data[i]
                    buf.data[i] = (r, t, p, idxs, max(r, 0) * float(wi.item()), buf.step)

        entries = list(buf.data)
        valid = [i for i, e in enumerate(entries) if e[4] is not None]
        if not valid: return []
        weights = np.array([entries[i][4] for i in valid], dtype=np.float32)

        k = min(k, len(valid))
        s = weights.sum()
        if s > 1e-9:
            prob = weights / s
            chosen = np.random.choice(len(valid), size=k, p=prob, replace=False)
            idxs = [valid[i] for i in chosen]
        else:
            idxs = np.random.choice(valid, size=k, replace=False)
        return [(entries[i][1], entries[i][2]) for i in idxs]

    # --------------------------
    # Batched rollout (policy)
    # --------------------------
    def batched_rollout(self, envs, temp, residuals, beta, ras_counts: Optional[dict] = None):
        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        num = len(envs)
        END_TOKEN = v.EOS

        for env in envs:
            env.y = residuals
            env.reset(c.batch_size)

        seqs = [[v.BOS] for _ in range(num)]
        depths: List[Deque[int]] = [deque([0]) for _ in range(num)]
        active, out = list(range(num)), [None] * num

        with torch.no_grad():
            while active:
                pad = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(seqs[i], device=device) for i in active],
                    batch_first=True, padding_value=v.PAD
                )
                logits, _ = self.pf(pad)
                last = logits[:, -1, :]

                if ras_counts is not None:
                    for i, oidx in enumerate(active):
                        path = tuple(seqs[oidx])
                        if path in ras_counts:
                            last[i, :] -= ras_counts[path] * 1e9

                masks = torch.zeros((len(active), v.size()), dtype=torch.bool, device=device)
                for i, oidx in enumerate(active):
                    d = depths[oidx][-1] if depths[oidx] else c.max_depth
                    if envs[oidx].open_leaves > 0 and d < c.max_depth:
                        masks[i, v.split_start : v.split_start + v.num_feat] = True
                    masks[i, v.EOS] = (len(seqs[oidx]) > 3)

                toks1 = _safe_sample(last, masks, temp)

                needs_th, still = {}, []
                for i, oidx in enumerate(active):
                    token = toks1[i].item()
                    seqs[oidx].append(token)

                    if ras_counts is not None:
                        path = tuple(seqs[oidx])
                        ras_counts[path] = ras_counts.get(path, 0) + 1

                    if token == END_TOKEN:
                        envs[oidx].done = True
                        continue

                    kind, idx = self.tokenizer.decode_one(token)
                    envs[oidx].step((kind, idx))
                    if kind == 'feat':
                        d0 = depths[oidx].pop()
                        depths[oidx].extend([d0 + 1, d0 + 1])
                        needs_th[len(needs_th)] = i
                    elif kind == 'th':
                        pass
                    else:
                        if depths[oidx]:
                            depths[oidx].pop()

                    if depths[oidx]:
                        still.append(oidx)
                    else:
                        envs[oidx].done = True

                if needs_th:
                    sub_idx = [active[i] for i in needs_th.values()]
                    sub_pad = torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(seqs[i], device=device) for i in sub_idx],
                        batch_first=True, padding_value=v.PAD
                    )
                    sub_logits, _ = self.pf(sub_pad)
                    th_mask = torch.zeros((sub_logits.size(0), v.size()), dtype=torch.bool, device=device)
                    th_mask[:, v.split_start + v.num_feat : v.split_start + v.num_feat + v.num_th] = True
                    toks2 = _safe_sample(sub_logits[:, -1, :], th_mask, temp)
                    for i, oidx in enumerate(sub_idx):
                        token = toks2[i].item()
                        seqs[oidx].append(token)
                        if token == END_TOKEN:
                            envs[oidx].done = True
                            continue
                        envs[oidx].step(self.tokenizer.decode_one(token))

                active = still

        for i in range(num):
            if envs[i].done:
                if seqs[i][-1] != v.EOS: seqs[i].append(v.EOS)
                out[i] = (seqs[i], envs[i].get_prior(beta).item(), envs[i].idxs.clone())

        return out

    # --------------------------
    # Predict
    # --------------------------
    def predict(self, df_test, df_train, use_policy=False, policy_inference_trees=None):
        ensemble_exists = self.ensemble or self.boosting_ensemble
        if not self.tokenizer or (not ensemble_exists and not use_policy):
            raise RuntimeError("Fit the model before predicting.")
        c = self.cfg

        env_template = TabularEnv(df_train, c.feature_cols, c.target_col, c.n_bins, c.task,
                                  binning_strategy=c.binning_strategy, device=c.device)
        X_te = env_template._featurise(df_test, df_train, c.feature_cols, c.n_bins)
        X_tr, y_tr = env_template.X_full.clone(), env_template.y_full.clone()

        if c.random_forest:
            preds = self._predict_random_forest(X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees)
        else:
            preds = self._predict_boosting(X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees)
        return preds.cpu().numpy()

    # memory-safe batching helper
    def _predict_in_batches(self, pred_fn, X, bs: int, device: str):
        out = []
        N = X.size(0)
        for s in range(0, N, bs):
            xb = X[s:s+bs].to(device, non_blocking=True)
            out.append(pred_fn(xb).to("cpu"))
        return torch.cat(out, dim=0)

    def _predict_random_forest(self, X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees):
        c = self.cfg
        device = X_tr.device  # keep everything on the same device as training bins

        # Targets for tree predictors
        y_train_target = (
            torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float)
            if c.task == "classification" else y_tr
        ).to(device)

        # Build the set of trees to use
        trees_to_use: List[List[int]] = []
        if use_policy:
            total_trees = policy_inference_trees if policy_inference_trees is not None else c.policy_inference_trees
            num_batches = math.ceil(total_trees / c.num_parallel)
            env_template.y = y_tr.clone().to(device)
            for _ in tqdm(range(num_batches), desc="Policy-based Tree Generation", leave=False):
                trees_in_batch = min(c.num_parallel, total_trees - len(trees_to_use))
                if trees_in_batch <= 0:
                    break
                batch_results = self.batched_rollout(
                    [copy.copy(env_template) for _ in range(trees_in_batch)],
                    temp=1.0, residuals=y_tr.to(device), beta=c.beta, ras_counts=({} if c.redundancy_aware else None)
                )
                trees_to_use.extend([res[0] for res in batch_results if res])
        else:
            trees_to_use = self.ensemble

        if not trees_to_use:
            raise RuntimeError("The forest is empty.")

        # Reward env lives on the right device
        reward_env = copy.copy(env_template)
        reward_env.y = y_tr.clone().to(device)
        reward_env.y_full = y_tr.clone().to(device)
        reward_env.reset(len(y_tr))

        # Accumulators
        if c.task == "classification":
            sum_preds = torch.zeros((X_te.shape[0], c.n_classes), device=device)
        else:
            sum_preds = torch.zeros(X_te.shape[0], device=device, dtype=torch.float32)
        total_weight = 0.0

        # Simple chunking to avoid VRAM spikes
        eval_bs = getattr(c, "eval_batch_size", 65536)

        for seq in tqdm(trees_to_use, desc="RF Prediction", leave=False):
            predictor = get_tree_predictor(seq, X_tr, y_train_target, self.tokenizer)
            # weight by Bayesian reward
            tok = torch.tensor([seq], device=device, dtype=torch.long)
            if c.task == 'classification':
                R = calculate_bayesian_reward(tok, self.tokenizer, reward_env, c.beta)
            else:
                R = calculate_bayesian_reward_regression(tok, self.tokenizer, reward_env, c.beta)
            w = float(R.item())
            sum_preds += w * predictor(X_te)
            total_weight += w

        if total_weight <= 0:
            raise RuntimeError("Total weight is zero.")

        preds = sum_preds / total_weight
        return torch.softmax(preds, dim=1) if c.task == "classification" else preds


    def _predict_boosting(self, X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees):
        c = self.cfg
        device = X_tr.device
        eval_bs = getattr(c, "eval_batch_size", 65536)

        # Running logits/values on the SAME device as training bins
        if c.task == "classification":
            test_preds  = torch.zeros((len(X_te), c.n_classes), device=device)
            train_preds = torch.zeros((len(y_tr), c.n_classes), device=device)
        else:
            test_preds  = torch.full((len(X_te),), self.y_mean, device=device, dtype=torch.float32)
            train_preds = torch.full_like(y_tr, self.y_mean, dtype=torch.float32, device=device)

        if use_policy:
            tqdm.write("--- Generating Boosting Ensemble with Policy (Sequential Inference) ---")
            total_trees = policy_inference_trees if policy_inference_trees is not None else c.updates
            num_batches = math.ceil(total_trees / c.num_parallel)

            # initial residuals
            if c.task == "classification":
                residuals = torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float) - torch.softmax(train_preds, dim=1)
            else:
                residuals = y_tr - train_preds

            env_template.y = residuals.clone().to(device)

            # generate candidates
            candidate_trees: List[List[int]] = []
            for _ in tqdm(range(num_batches), desc="Policy-based Tree Generation", leave=False):
                res = self.batched_rollout(
                    [copy.copy(env_template) for _ in range(c.num_parallel)],
                    temp=0.0, residuals=residuals, beta=c.beta, ras_counts=({} if c.redundancy_aware else None)
                )
                candidate_trees.extend([r[0] for r in res if r])

            # sequentially apply
            for seq in tqdm(candidate_trees, desc="Sequential Boosting Prediction", leave=False):
                if c.task == "classification":
                    residuals = torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float) - torch.softmax(train_preds, dim=1)
                else:
                    residuals = y_tr - train_preds

                predictor = get_tree_predictor(seq, X_tr, residuals, self.tokenizer)                
                train_preds += c.boosting_lr * predictor(X_tr)
                test_preds  += c.boosting_lr * predictor(X_te)

        else:
            # already-built ensemble: apply to test only
            for fn in tqdm(self.boosting_ensemble, desc="Ensemble Prediction", leave=False):
                test_preds  += c.boosting_lr * predictor(X_te)

        return torch.softmax(test_preds, dim=1) if c.task == "classification" else test_preds


    # sklearn compat
    def get_params(self, deep=True): return asdict(self.cfg)
    def set_params(self, **params):
        for k, v in params.items(): setattr(self.cfg, k, v)
        return self
