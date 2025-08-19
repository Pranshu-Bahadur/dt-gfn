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
from src.policy import PolicyPaperMLP
from src.utils import (
    ReplayBuffer, tb_loss, fl_loss, _safe_sample, get_tree_predictor,
    deltaE_split_gain_regression, deltaE_split_gain_classification, deltaE_split_gain_sse,
    calculate_bayesian_reward, calculate_bayesian_reward_regression,
    uniform_backward_log_prob,     # optional
    _build_tree_by_data,           # << add this
)

from src.utils import decode_tree_from_seq  # NEW


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    feature_cols: List[str]
    target_col: str = "target"
    task: str = "classification"                   # "classification" | "regression"
    reward_function: str = "bayesian"              # "bayesian" | "gini" | "variance" | "sse"
    n_classes: Optional[int] = None
    n_bins: int = 255
    binning_strategy: str = "global_uniform"
    device: str = "cuda"

    # training mode
    random_forest: bool = True                     # RF if True, Boosting if False

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

    # Backward policy choice
    backward_policy: str = "uniform"               # "uniform" | "network"

    # Priors & annealing
    beta: Optional[float] = None
    prior_scale: float = 0.5

    # Parallelism
    num_parallel: int = 10

    # Inference
    policy_inference_trees: int = 500

    # Memory/throughput
    amp: bool = True
    eval_on_cpu: bool = False
    metric_sample_size: int = 20000                # (unused; we eval on full data)
    eval_batch_size: int = 16384

    # Sampling / predictor guards
    rollout_temperature: float = 0.0               # sampling temp for rollouts
    min_child_size: int = 20                       # predictor split guard
    min_gain: float = 0.0                          # min impurity reduction

    # training reward scope
    training_reward_scope: str = "per_tree"        # "per_tree" | "ensemble"
    ensemble_reward_metric: str = "mse"            # (regression-only for now)

    # inference-time weighting reward (for RF & Boost)
    # None -> use training reward_function, "none" -> equal weights
    infer_reward_function: Optional[str] = None

    # policy-based predictor mode hint
    policy_predictor_mode: str = "dirichlet_sample"       # "dirichlet" | "mean"

    # Track best single-tree train accuracy while training (classification)
    show_best_tree_acc: bool = False

    # -------- Local expansion around the current best tree (optional) --------
    use_local_expansion: bool = False       # turn on/off shallow-leaf expansion
    local_expand_every: int = 5            # do expansion every N updates
    local_expand_k_leaves: int = 4         # how many shallow leaves to seed from
    local_expand_max_depth: int = 3        # only leaves with depth <= this
    local_expand_per_leaf: int = 3         # guided samples per chosen leaf

        # --- Visualization ---
    viz_every: int = 25                  # 0 disables
    viz_dir: str = "runs_v14/trees"
    viz_format: str = "png"
    show_best_tree_acc: bool = True      # helps decide which seq to render

    leaf_cooldown_steps: int = 1          # forbid immediate LEAF on fresh child if it can split
    leaf_bias: float = 0.0                # e.g., -0.2 to mildly discourage LEAF near root
    threshold_balance_gamma: float = 0.0  # e.g., 0.25 to penalize imbalanced thresholds



# ============================================================
# Trainer
# ============================================================
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer: Optional[Tokenizer] = None
        self.ensemble: List[List[int]] = []
        self.boosting_ensemble: List = []
        self.y_mean: float = 0.0

        self.pf = None
        self.pb = None
        self.log_z: Optional[torch.Tensor] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.le: Optional[LabelEncoder] = None
        self.classes_: Optional[np.ndarray] = None
        self.scaler = GradScaler(enabled=cfg.amp)

        # best-single-tree tracker
        self._best_tree_seq: Optional[List[int]] = None
        self._best_tree_acc: float = 0.0

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
            device=c.device,
        )
        if c.task == "classification":
            self.le, c.n_classes = env_template.le, env_template.n_classes
            self.classes_ = np.asarray(self.le.classes_)
        else:
            self.classes_ = None

        y_true = env_template.y_full.clone()
        X_binned = env_template.X_full.clone()

        # policy nets
        self.pf = torch.jit.script(
            PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device)
        )
        if c.backward_policy == "network":
            self.pb = torch.jit.script(
                PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device)
            )
        else:
            self.pb = None

        self.log_z = torch.nn.Parameter(torch.tensor(1.0, device=c.device))

        # optimizers / schedulers
        optim_pfs = torch.optim.AdamW(self.pf.parameters(), lr=c.lr)
        sched_pfs = SequentialLR(
            optim_pfs,
            [LambdaLR(optim_pfs, lambda u: min(1.0, u / max(1, 10))),
             CosineAnnealingLR(optim_pfs, T_max=max(1, c.updates - 10))],
            milestones=[10],
        )

        opt_list = [optim_pfs]
        sch_list = [sched_pfs]

        if self.pb is not None:
            optim_pbs = torch.optim.AdamW(self.pb.parameters(), lr=c.lr)
            sched_pbs = SequentialLR(
                optim_pbs,
                [LambdaLR(optim_pbs, lambda u: min(1.0, u / max(1, 10))),
                 CosineAnnealingLR(optim_pbs, T_max=max(1, c.updates - 10))],
                milestones=[10],
            )
            opt_list.append(optim_pbs)
            sch_list.append(sched_pbs)

        optim_z = torch.optim.Adam([self.log_z], lr=c.lr / 10)
        sched_z = SequentialLR(
            optim_z,
            [LambdaLR(optim_z, lambda u: min(1.0, u / max(1, 10))),
             CosineAnnealingLR(optim_z, T_max=max(1, c.updates - 10))],
            milestones=[10],
        )
        opt_list.append(optim_z)
        sch_list.append(sched_z)

        optimizers = opt_list
        schedulers = sch_list

        self.replay_buffer = ReplayBuffer(capacity=200)

        if c.beta is None:
            c.beta = math.log(4) + math.log(len(c.feature_cols))
            tqdm.write(f"[trainer] β (structure prior) = {c.beta:.4f}")

        if c.random_forest:
            self._fit_dt_gfn_random_forest(env_template, y_true, X_binned, optimizers, schedulers)
        else:
            self._fit_boost_gfn(env_template, y_true, X_binned, optimizers, schedulers)

        return self

    # ========================================================
    # Reward helpers
    # ========================================================
    def _per_tree_reward(self, tok: torch.Tensor, reward_env: TabularEnv) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        c = self.cfg

        # Detect residual-matrix training (multi-class boosting)
        is_residual_matrix = (
            hasattr(reward_env, "y") and isinstance(reward_env.y, torch.Tensor) and reward_env.y.dim() == 2
            and not torch.allclose(reward_env.y.sum(1), torch.ones_like(reward_env.y.sum(1)), atol=1e-3, rtol=0.0)
        )

        if c.reward_function == 'bayesian' and not is_residual_matrix:
            fn = calculate_bayesian_reward if c.task == 'classification' else calculate_bayesian_reward_regression
            R_t = fn(tok, self.tokenizer, reward_env, c.beta)
            return R_t, None

        # Otherwise shape by SSE gains (regression OR residual-matrix classification)
        if is_residual_matrix or c.reward_function in ("variance", "sse"):
            dR = deltaE_split_gain_sse(tok, self.tokenizer, reward_env)
            R_t = torch.clamp(dR.sum(), min=1e-9)
            return R_t, dR

        # Fallback to Gini or variance by task
        if c.task == 'classification':
            dR = deltaE_split_gain_classification(tok, self.tokenizer, reward_env)
        else:
            dR = deltaE_split_gain_regression(tok, self.tokenizer, reward_env)
        R_t = torch.clamp(dR.sum(), min=1e-9)
        return R_t, dR

    @torch.no_grad()
    def _weight_for_tree(self, seq: List[int], reward_env: TabularEnv, mode: str) -> float:
        device = reward_env.device
        tok = torch.tensor([seq], device=device, dtype=torch.long)
        task = self.cfg.task
        if mode == "none":
            return 1.0
        if mode == "bayesian":
            R = calculate_bayesian_reward(tok, self.tokenizer, reward_env, self.cfg.beta) if task == "classification" \
                else calculate_bayesian_reward_regression(tok, self.tokenizer, reward_env, self.cfg.beta)
            return float(R.item())
        if mode in ("variance", "sse"):
            dR = deltaE_split_gain_sse(tok, self.tokenizer, reward_env)
            return float(torch.clamp(dR.sum(), min=1e-9).item())
        if mode == "gini":
            dR = deltaE_split_gain_classification(tok, self.tokenizer, reward_env)
            return float(torch.clamp(dR.sum(), min=1e-9).item())
        # default to training reward function
        return self._weight_for_tree(seq, reward_env, self.cfg.reward_function)

    @torch.no_grad()
    def _ensemble_reward_mse(
        self,
        seqs: List[List[int]],
        X_binned: torch.Tensor,
        y_true: torch.Tensor,
        base_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.cfg
        device = X_binned.device
        if c.task != "regression":
            return torch.tensor(1.0, device=device)

        if base_pred is None:
            if not seqs:
                return torch.tensor(1.0, device=device)
            preds_list = []
            for seq in seqs:
                pred_fn = get_tree_predictor(
                    seq, X_binned, y_true, self.tokenizer,
                    min_child_size=c.min_child_size, min_gain=c.min_gain,
                    predictor_mode=c.policy_predictor_mode
                )
                preds_list.append(pred_fn(X_binned))
            ens_pred = torch.stack(preds_list, dim=0).mean(dim=0)
        else:
            ens_pred = base_pred.to(device)

        ys = y_true.to(device).float()
        mse_ens = torch.mean((ens_pred - ys) ** 2)
        baseline = ys.mean()
        mse_base = torch.mean((baseline - ys) ** 2)

        clip = (mse_base + 1e-6).detach()
        logR = c.beta * ((mse_base - mse_ens) / clip)
        logR = torch.clamp(logR, min=-50.0, max=50.0)
        R = torch.exp(logR).clamp_min(1e-9)
        return R  # scalar

    # ========================================================
    # Batched policy update (one step) — v13 logic, but tb_loss(log_r)
    # ========================================================
    def _update_policy(
        self,
        all_tuples_with_targets: List,
        env_template: TabularEnv,
        optimizers: List,
        ensemble_reward_override: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        if not all_tuples_with_targets:
            return 0.0, 0.0
        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device

        seqs, priors, targets = zip(*all_tuples_with_targets)
        toks = [torch.tensor(s, device=device, dtype=torch.long) for s in seqs]
        padded = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=v.PAD)
        priors_tensor = torch.as_tensor(priors, device=device, dtype=torch.float32)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        with autocast(enabled=c.amp):
            log_pf = self.pf.log_prob(padded)

            # backward policy: network on reversed tokens OR uniform surrogate
            if self.pb is not None and c.backward_policy == "network":
                flipped = torch.nn.utils.rnn.pad_sequence(
                    [t.flip(0) for t in toks], batch_first=True, padding_value=v.PAD
                )
                log_pb = self.pb.log_prob(flipped)
            elif c.backward_policy == "uniform":
                # simple surrogate: cancels in TB up to a constant → use zeros
                # (you can swap to uniform_backward_log_prob(padded, self.tokenizer, c.max_depth) if desired)
                log_pb = torch.zeros_like(log_pf)
            else:
                log_pb = torch.zeros_like(log_pf)

            logF = self.pf.log_F(padded)

            if ensemble_reward_override is not None:
                # tb_loss now expects log_r
                R = ensemble_reward_override.expand(len(seqs)).to(device)
                log_r = torch.log(R.clamp_min(1e-9))
                l_tb = tb_loss(log_pf, log_pb, self.log_z, log_r, priors_tensor)
                loss = l_tb
                tb_val, fl_val = l_tb, None
            else:
                reward_env = copy.copy(env_template)
                R_list, dR_list = [], []
                for i, t in enumerate(toks):
                    tok_i = padded[i:i+1, :t.numel()]
                    target = targets[i]
                    reward_env.y = target
                    reward_env.reset(len(target))  # uses shared sampler internally
                    R_t, dR = self._per_tree_reward(tok_i, reward_env)
                    R_list.append(R_t.squeeze())
                    if dR is not None:
                        dR_list.append(dR.squeeze(0))

                R = torch.stack(R_list, dim=0)

                if self.cfg.reward_function == 'bayesian':
                    log_r = torch.log(R.clamp_min(1e-9))  # <— minimal change for new tb_loss
                    l_tb = tb_loss(log_pf, log_pb, self.log_z, log_r, priors_tensor)
                    loss = l_tb
                    tb_val, fl_val = l_tb, None
                else:
                    logR = torch.log(R + 1e-9)
                    if dR_list:
                        gains = torch.nn.utils.rnn.pad_sequence(dR_list, batch_first=True, padding_value=0.0)
                        gains = torch.relu(gains)
                        gsum = gains.sum(1, keepdim=True).clamp_min(1e-9)
                        dR_shaped = gains * (logR.unsqueeze(1) / gsum)
                    else:
                        dR_shaped = torch.zeros_like(log_pf)

                    Tm1 = log_pf.size(1)
                    if dR_shaped.size(1) < Tm1:
                        pad = torch.zeros((dR_shaped.size(0), Tm1 - dR_shaped.size(1)), device=device)
                        dR_shaped = torch.cat([dR_shaped, pad], dim=1)
                    elif dR_shaped.size(1) > Tm1:
                        dR_shaped = dR_shaped[:, :Tm1]

                    log_r = torch.log(R.clamp_min(1e-9))
                    l_tb = tb_loss(log_pf, log_pb, self.log_z, log_r, priors_tensor)
                    l_fl = fl_loss(logF, log_pf, log_pb, dR_shaped)
                    loss = l_tb + l_fl
                    tb_val, fl_val = l_tb, l_fl

        self.scaler.scale(loss).backward()
        for opt in optimizers:
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
            self.scaler.step(opt)
        self.scaler.update()

        self.replay_buffer.mark_policy_update()

        tb_loss_acc = float(tb_val.item())
        fl_loss_acc = float(fl_val.item()) if fl_val is not None else 0.0
        return tb_loss_acc, fl_loss_acc

    # ========================================================
    # RF training (policy + gen)
    # ========================================================
    def _visualize_best_tree_live(
        self,
        env_template,                     # TabularEnv already in training
        *,
        save_dir: str = "runs/trees",
        step: int | None = None,
        format: str = "png",
    ):
        """
        Render the best-known tree using a structure rebuilt *by data*:
          • Replays tokens on TRAIN rows, accepts a split only if both children are non-empty.
          • Uses LIFO expansion (push Right then Left so Left expands next), matching rollout.
          • Labels leaves with train counts and class probs (or mean±sd for regression).

        Returns a graphviz.Digraph (and writes <save_dir>/best_tree_step_<step>.<format>).
        """
        import os
        try:
            import graphviz
        except Exception:
            from tqdm import tqdm
            tqdm.write("[viz] graphviz not available; skipping.")
            return None

        # pick a sequence to render
        seq = self._best_tree_seq or (self.ensemble[0] if self.ensemble else None)
        if not seq:
            return None

        device = env_template.device
        tok = torch.tensor([seq], device=device, dtype=torch.long)

        # Build a data-consistent tree on ALL train rows
        N = env_template.X_full.size(0)
        all_rows = torch.arange(N, device=device)
        root, _ = _build_tree_by_data(tok, self.tokenizer, env_template.X_full, all_rows)

        Xb = env_template.X_full
        y  = env_template.y_full
        is_cls = (self.cfg.task == "classification")
        if is_cls:
            n_classes = int(self.cfg.n_classes) if self.cfg.n_classes is not None else int(y.max().item()) + 1
            if self.classes_ is not None:
                class_names = [str(c) for c in self.classes_.tolist()]
            else:
                class_names = [f"C{i}" for i in range(n_classes)]

        def leaf_label(indices: torch.Tensor) -> str:
            n = int(indices.numel())
            if is_cls:
                if n == 0:
                    return "Leaf • n=0"
                counts = torch.bincount(y.index_select(0, indices), minlength=n_classes)
                total = int(counts.sum().item())
                maj = int(torch.argmax(counts).item()) if total > 0 else 0
                probs = (counts.float() / max(1, total)).cpu().numpy()
                prob_str = ", ".join([f"{class_names[i]}:{probs[i]:.2f}" for i in range(n_classes)])
                return f"Leaf • n={n}\nmajority={class_names[maj]}\n{prob_str}"
            else:
                if n == 0:
                    return "Leaf • n=0\nmean=0.0"
                vals = y.index_select(0, indices).float()
                mu = float(vals.mean().item())
                sd = float(vals.std(unbiased=False).item())
                return f"Leaf • n={n}\nmean={mu:.4f} ± {sd:.4f}"

        dot = graphviz.Digraph(comment="Best Decision Tree")
        dot.attr("node", shape="box", style="rounded")
        nid = 0

        # Traverse the rebuilt tree, routing TRAIN rows to compute labels
        def walk(node, idxs: torch.Tensor):
            nonlocal nid
            if node.get("type") != "split":
                lid = str(nid); nid += 1
                dot.node(lid, leaf_label(idxs), style="rounded,filled", fillcolor="lightblue")
                return lid

            f = int(node["f"]); t = int(node["t"])
            fname = self.cfg.feature_cols[f] if 0 <= f < len(self.cfg.feature_cols) else f"X[{f}]"
            myid = str(nid); nid += 1
            dot.node(myid, f"{fname} ≤ bin {t}")

            fv = Xb.index_select(0, idxs)[:, f]
            m = fv <= t
            lid = walk(node["L"], idxs[m])
            rid = walk(node["R"], idxs[~m])
            dot.edge(myid, lid, label="True")
            dot.edge(myid, rid, label="False")
            return myid

        walk(root, all_rows)

        os.makedirs(save_dir, exist_ok=True)
        fname = f"best_tree_step_{step}" if step is not None else "best_tree"
        dot.render(os.path.join(save_dir, fname), format=format, cleanup=True)
        return dot


    def _fit_dt_gfn_random_forest(self, env_template, y_true, X_binned, optimizers, schedulers):
        c = self.cfg
        tqdm.write("--- Starting DT-GFN (Random Forest) Training ---")

        y_target_for_reward = (
            torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float)
            if c.task == "classification" else y_true.clone()
        )
        env_template.y = y_true.clone()

        X_metric = X_binned.cpu() if c.eval_on_cpu else X_binned
        y_metric = y_true.cpu() if c.eval_on_cpu else y_true
        Ytarget_full = (
            torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float)
            if c.task == "classification" else y_true
        )
        Ytarget_full = Ytarget_full.cpu() if c.eval_on_cpu else Ytarget_full
        X_build_for_pred = X_binned.cpu() if c.eval_on_cpu else X_binned

        all_tuples_last = []

        for upd in tqdm(range(1, c.updates + 1), desc="Policy Training & Tree Generation"):
            forward_tuples = self._collect_rollouts(env_template, temp=c.rollout_temperature, residuals=y_true, beta=c.beta)

            # NEW: shallow-leaf expansion around current best tree
            if c.use_local_expansion and (upd % max(1, c.local_expand_every) == 0):
                expanded = self._local_expand_best(env_template, y_true)
                if expanded:
                    forward_tuples.extend(expanded)

            replay_tuples  = self.sample_replay(c.top_k_trees)
            all_tuples     = forward_tuples + replay_tuples
            if not all_tuples:
                for sch in schedulers: sch.step()
                continue

            # optional ensemble reward (regression-only)
            ensemble_R_override = None
            if c.training_reward_scope == "ensemble" and c.task == "regression" and len(all_tuples) > 0:
                seqs = [seq for seq, _ in all_tuples]
                R_scalar = self._ensemble_reward_mse(seqs, X_binned, y_true)
                ensemble_R_override = R_scalar

            all_tuples_with_targets = [(seq, prior, y_target_for_reward) for seq, prior in all_tuples]
            avg_tb_loss, avg_fl_loss = self._update_policy(
                all_tuples_with_targets, env_template, optimizers, ensemble_reward_override=ensemble_R_override
            )
            for sch in schedulers:
                sch.step()

            # ---- FULL DATA metrics (reward-weighted mean for speed) ----
            trees = [seq for seq, _ in all_tuples if seq]

            sum_pred, total_w = None, 0.0
            with torch.no_grad():
                for seq in trees:
                    pred_fn = get_tree_predictor(
                        seq, X_build_for_pred, Ytarget_full, self.tokenizer,
                        min_child_size=c.min_child_size, min_gain=c.min_gain,
                        predictor_mode=c.policy_predictor_mode
                    )
                    w = self._weight_for_tree(seq, env_template, mode=self.cfg.reward_function)
                    running = self._predict_in_batches(
                        pred_fn, X_metric, c.eval_batch_size,
                        device=("cpu" if c.eval_on_cpu else c.device),
                    )
                    if sum_pred is None:
                        sum_pred = w * running
                    else:
                        sum_pred += w * running
                    total_w += w

            log_str = f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f} | Trees: {len(trees)}"
            if sum_pred is not None and total_w > 0:
                avg_pred = sum_pred / total_w
                if c.task == "classification":
                    acc = (avg_pred.argmax(1).cpu() == y_metric.cpu()).float().mean().item()
                    log_str += f" | Train Acc (w): {acc:.4f}"
                else:
                    if avg_pred.std() > 0 and y_metric.std() > 0:
                        corr = torch.corrcoef(torch.stack([avg_pred.squeeze().cpu(), y_metric.squeeze().cpu()]))[0, 1].item()
                        log_str += f" | Train Corr (w): {corr:.4f}"

            # ---- Optional: best single-tree (unweighted) train accuracy tracker ----
            if c.show_best_tree_acc and c.task == "classification" and trees:
                with torch.no_grad():
                    for seq in trees:
                        pred_fn = get_tree_predictor(
                            seq, X_build_for_pred, Ytarget_full, self.tokenizer,
                            min_child_size=c.min_child_size, min_gain=c.min_gain,
                            predictor_mode=c.policy_predictor_mode
                        )
                        preds = self._predict_in_batches(
                            pred_fn, X_metric, c.eval_batch_size,
                            device=("cpu" if c.eval_on_cpu else c.device),
                        )
                        acc_i = (preds.argmax(1).cpu() == y_metric.cpu()).float().mean().item()
                        if acc_i > self._best_tree_acc:
                            self._best_tree_acc = acc_i
                            self._best_tree_seq = seq
                    log_str += f" | BestTreeAcc: {self._best_tree_acc:.4f}"

            tqdm.write(log_str)
            # Auto-visualize every N updates
            viz_every = int(getattr(self.cfg, "viz_every", 0) or 0)
            if viz_every and (upd % viz_every == 0):
                out_dir = getattr(self.cfg, "viz_dir", "runs/trees")
                fmt = getattr(self.cfg, "viz_format", "png")
                self._visualize_best_tree_live(env_template, save_dir=out_dir, step=upd, format=fmt)

            all_tuples_last = all_tuples

        self.ensemble = [seq for seq, _ in all_tuples_last if seq] if all_tuples_last else []
        tqdm.write(f"--- RF finished. Final forest size: {len(self.ensemble)} ---")

    # ========================================================
    # Boosting training (unchanged core logic)
    # ========================================================
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
            fresh = self._collect_rollouts(env_template, c.rollout_temperature, residuals, c.beta)
            candidates = replay + fresh
            if not candidates:
                for sch in schedulers: sch.step()
                continue

            tuples = []
            current_res = residuals.clone()
            for seq, prior in candidates:
                tuples.append((seq, prior, current_res.clone()))
                pred = get_tree_predictor(
                    seq, X_binned, current_res, self.tokenizer,
                    min_child_size=c.min_child_size, min_gain=c.min_gain,
                    predictor_mode=c.policy_predictor_mode
                )
                add_train = pred(X_binned)
                base_pred += c.boosting_lr * add_train
                if c.task == "classification":
                    probs = torch.softmax(base_pred, dim=1)
                    current_res = torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float) - probs
                else:
                    current_res = y_true - base_pred

            ensemble_R_override = None
            if c.training_reward_scope == "ensemble" and c.task == "regression":
                R_scalar = self._ensemble_reward_mse([], X_binned, y_true, base_pred=base_pred)
                ensemble_R_override = R_scalar

            avg_tb_loss, avg_fl_loss = self._update_policy(
                tuples, env_template, optimizers, ensemble_reward_override=ensemble_R_override
            )
            for sch in schedulers:
                sch.step()

            if c.task == "classification":
                acc = (base_pred.argmax(1) == y_true).float().mean().item()
                tqdm.write(f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f} | Acc: {acc:.4f}")
            else:
                if base_pred.std() > 0 and y_true.std() > 0:
                    corr = torch.corrcoef(torch.stack([base_pred.squeeze(), y_true.squeeze()]))[0, 1].item()
                    tqdm.write(f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f} | Corr: {corr:+.4f}")
                else:
                    tqdm.write(f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f} | Corr: nan")

    # ========================================================
    # Rollouts (policy can *close* a branch early via LEAF)
    # ========================================================
    def _collect_rollouts(self, env_template, temp, residuals, beta):
        forward_tuples, done = [], 0
        ras_counts = {} if self.cfg.redundancy_aware else None

        with tqdm(total=self.cfg.rollouts, desc="Rollouts", leave=False) as pbar:
            while done < self.cfg.rollouts:
                if ras_counts is not None:
                    ras_counts.clear()
                batch = min(self.cfg.num_parallel, self.cfg.rollouts - done)

                envs = [copy.copy(env_template) for _ in range(batch)]
                idx_batches = [env_template.draw_indices(self.cfg.batch_size) for _ in range(batch)]
                for env, idxs in zip(envs, idx_batches):
                    env.y = residuals
                    env.paths = []
                    env.open_leaves = 1
                    env.done = False
                    env.idxs = idxs

                results = self.batched_rollout(envs, temp, residuals, beta, ras_counts)

                for res in results:
                    if not res:
                        continue
                    seq, prior, idxs = res
                    reward_env = copy.copy(env_template)
                    reward_env.idxs = idxs.to(self.cfg.device)
                    reward_env.y_full = env_template.y_full
                    reward_env.X_full = env_template.X_full
                    reward_env.y = residuals
                    r = self._weight_for_tree(seq, reward_env, mode=self.cfg.reward_function)
                    self.replay_buffer.add(r, seq, prior, idxs.cpu())
                    forward_tuples.append((seq, prior))

                done += batch
                pbar.update(batch)
        return forward_tuples

    def sample_replay(self, k: int, REFRESH_INTERVAL: int = 5) -> List[Tuple[List[int], float]]:
        buf = self.replay_buffer
        if not buf or not buf.data:
            return []

        # If using uniform backward, just return top-k by reward (no pb weighting refresh).
        if self.cfg.backward_policy != "network":
            entries = list(buf.data)
            k = min(k, len(entries))
            return [(entries[i][1], entries[i][2]) for i in range(k)]

        # Otherwise: refresh backward weights via learned pb on reversed sequences.
        stale = [i for i, e in enumerate(buf.data) if e[4] is None or buf.step - e[5] >= REFRESH_INTERVAL]
        if stale:
            stale_seqs = [buf.data[i][1] for i in stale]
            with torch.no_grad():
                flipped = [torch.tensor(s, device=self.cfg.device).flip(0) for s in stale_seqs]
                padded = torch.nn.utils.rnn.pad_sequence(flipped, batch_first=True, padding_value=self.tokenizer.v.PAD)
                logp = self.pb.log_prob(padded)  # type: ignore[union-attr]
                mask = (padded != self.tokenizer.v.PAD).float()
                T = min(mask.size(1), logp.size(1))
                w = (logp[:, :T] * mask[:, :T]).sum(1).exp()
                for i, wi in zip(stale, w):
                    r, t, p, idxs, _, _ = buf.data[i]
                    buf.data[i] = (r, t, p, idxs, float(wi.item()), buf.step)

        entries = list(buf.data)
        valid = [i for i, e in enumerate(entries) if e[4] is not None]
        if not valid:
            return []

        weights = np.array(
            [max(entries[i][0], 1e-9) * float(entries[i][4]) for i in valid],
            dtype=np.float32
        )

        k = min(k, len(valid))
        pos = np.flatnonzero(weights > 0)

        if pos.size == 0:
            chosen_valid_idx = np.random.choice(len(valid), size=k, replace=False)
            idxs = [valid[j] for j in chosen_valid_idx]
        elif pos.size < k:
            prob_pos = weights[pos] / weights[pos].sum()
            first = np.random.choice(pos, size=pos.size, replace=False, p=prob_pos)
            remaining_pool = np.setdiff1d(np.arange(len(valid)), first, assume_unique=False)
            fill = np.random.choice(remaining_pool, size=k - pos.size, replace=False)
            chosen_local = np.concatenate([first, fill])
            idxs = [valid[j] for j in chosen_local]
        else:
            prob = weights[pos] / weights[pos].sum()
            chosen_local = np.random.choice(pos, size=k, replace=False, p=prob)
            idxs = [valid[j] for j in chosen_local]

        return [(entries[i][1], entries[i][2]) for i in idxs]


    def batched_rollout(self, envs, temp, residuals, beta, ras_counts: Optional[dict] = None):
        """
        Roll out trees with *feasible* actions only (two-stage FEAT -> THRESHOLD).
        Fixes:
          • Aligns masks when filtering thresholds (no IndexError).
          • Respects per-leaf feasible bin windows (lo/hi).
          • Optional threshold imbalance penalty (cfg.threshold_balance_gamma).
          • Optional LEAF cooldown to avoid premature closures (cfg.leaf_cooldown_steps).

        Returns: list of (seq, prior, idxs) or None per env.
        """
        import torch
        from collections import deque

        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        num = len(envs)
        END_TOKEN  = v.EOS
        LEAF_TOKEN = self.tokenizer._leaf(0)

        # optional knobs
        leaf_cd_steps = int(getattr(c, "leaf_cooldown_steps", 0) or 0)
        th_imbal_gamma = float(getattr(c, "threshold_balance_gamma", 0.0) or 0.0)

        # Prepare envs
        for env in envs:
            env.y = residuals
            env.reset(c.batch_size)

        # Per-trajectory state
        seqs = [[v.BOS] for _ in range(num)]
        depths: List[Deque[int]] = [deque([0]) for _ in range(num)]
        lo_stacks: List[Deque[torch.Tensor]] = [deque([torch.zeros(v.num_feat, dtype=torch.long, device=device)]) for _ in range(num)]
        hi_stacks: List[Deque[torch.Tensor]] = [deque([torch.full((v.num_feat,), v.num_th - 1, dtype=torch.long, device=device)]) for _ in range(num)]
        row_stacks: List[Deque[torch.Tensor]] = [deque([torch.arange(envs[i].idxs.numel(), device=device)]) for i in range(num)]
        # cooldown stacks (one counter per open leaf)
        cd_stacks: List[Deque[int]] = [deque([0]) for _ in range(num)]

        def _mark_done_if_finished(ti: int):
            if not depths[ti]:
                envs[ti].done = True

        active = [i for i in range(num) if not envs[i].done]
        out = [None] * num

        with torch.no_grad():
            while active:
                pad = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(seqs[i], device=device) for i in active],
                    batch_first=True, padding_value=v.PAD
                )
                logits, _ = self.pf(pad)
                last = logits[:, -1, :]

                # Strong "don't repeat this exact prefix" guard if ras_counts provided
                if ras_counts is not None:
                    for bi, oidx in enumerate(active):
                        pref = tuple(seqs[oidx])
                        if pref in ras_counts:
                            last[bi, :] -= 1e9  # effectively forbids repeating this exact action at this prefix

                # ---------------- first decision: choose FEAT or LEAF ----------------
                mask1 = torch.zeros((len(active), v.size()), dtype=torch.bool, device=device)

                for bi, oidx in enumerate(active):
                    if not depths[oidx]:
                        continue

                    d = depths[oidx][-1]
                    can_split = (d < c.max_depth)

                    rows_rel = row_stacks[oidx][-1]
                    Xb = envs[oidx].X_full[envs[oidx].idxs]   # [B,F]
                    Xleaf = Xb.index_select(0, rows_rel)       # [n_leaf,F]
                    n_leaf = int(Xleaf.size(0))

                    # Valid features: need ≥2 distinct bins; if min_child_size>0, at least one viable threshold
                    valid_feats = []
                    if can_split and n_leaf > 1:
                        mcs = int(c.min_child_size or 0)
                        for f in range(Xleaf.size(1)):
                            bf = Xleaf[:, f]
                            uniq, counts = torch.unique(bf, return_counts=True)
                            if uniq.numel() < 2:
                                continue
                            if mcs > 1:
                                csum = counts.cumsum(0)[:-1]   # positions align with uniq[:-1]
                                if not bool(((csum >= mcs) & ((n_leaf - csum) >= mcs)).any()):
                                    continue
                            valid_feats.append(f)

                    # LEAF allowed unless cooldown blocks and a split is feasible
                    allow_leaf = True
                    if leaf_cd_steps > 0 and can_split:
                        if cd_stacks[oidx][-1] > 0:
                            allow_leaf = False

                    if allow_leaf:
                        mask1[bi, LEAF_TOKEN] = True

                    if valid_feats:
                        feat_ids = v.split_start + torch.as_tensor(valid_feats, device=device, dtype=torch.long)
                        mask1[bi, feat_ids] = True

                    # never allow EOS here; branches close via LEAF
                    mask1[bi, v.EOS] = False

                    # cooldown ticks down one step while this leaf remains open
                    if cd_stacks[oidx]:
                        cd_stacks[oidx][-1] = max(0, cd_stacks[oidx][-1] - 1)

                toks1 = _safe_sample(last, mask1, temp)

                # Collect leaves that chose a FEAT (need threshold next)
                need_threshold: List[Tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
                still_for_round: List[int] = []

                for bi, oidx in enumerate(active):
                    tok = int(toks1[bi].item())
                    if not mask1[bi].any():
                        envs[oidx].done = True
                        continue

                    seqs[oidx].append(tok)
                    if ras_counts is not None:
                        pref = tuple(seqs[oidx])
                        ras_counts[pref] = ras_counts.get(pref, 0) + 1

                    if tok == LEAF_TOKEN:
                        # close this leaf
                        envs[oidx].step(("leaf", 0))
                        depths[oidx].pop(); lo_stacks[oidx].pop(); hi_stacks[oidx].pop(); row_stacks[oidx].pop(); cd_stacks[oidx].pop()
                        _mark_done_if_finished(oidx)
                        if not envs[oidx].done:
                            still_for_round.append(oidx)
                        continue

                    # FEAT chosen -> request threshold
                    kind, f_idx = self.tokenizer.decode_one(tok)  # 'feat'
                    envs[oidx].step((kind, f_idx))

                    d0 = depths[oidx].pop()
                    lo_top = lo_stacks[oidx].pop()
                    hi_top = hi_stacks[oidx].pop()
                    rows_rel = row_stacks[oidx].pop()
                    cd_top = cd_stacks[oidx].pop()

                    need_threshold.append((oidx, int(f_idx), d0, lo_top.clone(), hi_top.clone(), rows_rel.clone(), cd_top))

                # ---------------- second decision: choose THRESHOLD ----------------
                if need_threshold:
                    sub_idx = [oidx for (oidx, *_) in need_threshold]
                    sub_pad = torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(seqs[i], device=device) for i in sub_idx],
                        batch_first=True, padding_value=v.PAD
                    )
                    sub_logits, _ = self.pf(sub_pad)
                    last_th = sub_logits[:, -1, :]

                    mask2 = torch.zeros((len(sub_idx), v.size()), dtype=torch.bool, device=device)
                    th_base = v.split_start + v.num_feat

                    for si, (oidx, f_idx, d0, lo_top, hi_top, rows_rel, cd_top) in enumerate(need_threshold):
                        Xb = envs[oidx].X_full[envs[oidx].idxs]
                        bf = Xb.index_select(0, rows_rel)[:, f_idx]

                        if bf.numel() == 0:
                            continue

                        uniq, counts = torch.unique(bf, sorted=True, return_counts=True)
                        if uniq.numel() < 2:
                            continue

                        # Candidate thresholds are the *bin values* uniq[:-1]; positions 0..len-1
                        th_all_bins = uniq[:-1]
                        th_pos_all  = torch.arange(th_all_bins.numel(), device=device)

                        # min_child_size mask
                        mask_mc = torch.ones_like(th_pos_all, dtype=torch.bool)
                        mcs = int(c.min_child_size or 0)
                        if mcs > 1:
                            csum_all = counts.cumsum(0)[:-1]
                            n_leaf = int(bf.numel())
                            left_ok  = csum_all >= mcs
                            right_ok = (n_leaf - csum_all) >= mcs
                            mask_mc = left_ok & right_ok

                        # feasible bin window mask
                        lo_f = int(lo_top[f_idx].item())
                        hi_f = int(hi_top[f_idx].item())
                        mask_win = (th_all_bins >= lo_f) & (th_all_bins <= hi_f)

                        # final mask for these thresholds
                        mask_final = mask_mc & mask_win
                        if mask_final.any():
                            cand_bins = th_all_bins[mask_final]                 # bin values
                            cand_pos  = th_pos_all[mask_final]                  # positions into csum_all

                            th_ids = th_base + cand_bins.to(device=device, dtype=torch.long)
                            mask2[si, th_ids] = True

                            # Optional imbalance penalty: push logits down for skewed splits
                            if th_imbal_gamma > 0.0 and mcs > 0:
                                csum_all = counts.cumsum(0)[:-1].float()
                                n_leaf = float(bf.numel())
                                left_counts = csum_all.index_select(0, cand_pos)
                                imbal = (2.0 * (left_counts / max(1.0, n_leaf)) - 1.0).abs()  # ∈ [0,1]
                                for jj, bval in enumerate(cand_bins.tolist()):
                                    last_th[si, th_base + int(bval)] -= th_imbal_gamma * float(imbal[jj].item())

                        # never allow EOS/LEAF in threshold step
                        mask2[si, v.EOS] = False
                        mask2[si, LEAF_TOKEN] = False

                    toks2 = _safe_sample(last_th, mask2, temp)

                    for si, (oidx, f_idx, d0, lo_top, hi_top, rows_rel, cd_top) in enumerate(need_threshold):
                        t_tok = int(toks2[si].item())
                        seqs[oidx].append(t_tok)
                        if ras_counts is not None:
                            pref = tuple(seqs[oidx])
                            ras_counts[pref] = ras_counts.get(pref, 0) + 1

                        _, t_idx = self.tokenizer.decode_one(t_tok)

                        # Split rows
                        Xb = envs[oidx].X_full[envs[oidx].idxs]
                        fv = Xb.index_select(0, rows_rel)[:, f_idx]
                        m = fv <= t_idx
                        rows_L = rows_rel[m]
                        rows_R = rows_rel[~m]

                        # Update feasible bin windows for children
                        lo_L, hi_L = lo_top.clone(), hi_top.clone()
                        hi_L[f_idx] = torch.minimum(hi_L[f_idx], torch.as_tensor(t_idx, device=device))
                        lo_R, hi_R = lo_top.clone(), hi_top.clone()
                        lo_R[f_idx] = torch.maximum(lo_R[f_idx], torch.as_tensor(t_idx + 1, device=device))

                        # Push children (R then L) for LIFO expansion order
                        depths[oidx].append(d0 + 1); lo_stacks[oidx].append(lo_R); hi_stacks[oidx].append(hi_R); row_stacks[oidx].append(rows_R)
                        depths[oidx].append(d0 + 1); lo_stacks[oidx].append(lo_L); hi_stacks[oidx].append(hi_L); row_stacks[oidx].append(rows_L)

                        # Set cooldown for children (if enabled)
                        if leaf_cd_steps > 0:
                            cd_stacks[oidx].append(leaf_cd_steps)
                            cd_stacks[oidx].append(leaf_cd_steps)
                        else:
                            cd_stacks[oidx].append(0)
                            cd_stacks[oidx].append(0)

                        envs[oidx].step(("th", int(t_idx)))
                        if depths[oidx]:
                            still_for_round.append(oidx)
                        else:
                            envs[oidx].done = True

                active = still_for_round

        # Finalize outputs
        for i in range(num):
            if envs[i].done:
                if seqs[i][-1] != END_TOKEN:
                    seqs[i].append(END_TOKEN)
                out[i] = (seqs[i], envs[i].get_prior(beta).item(), envs[i].idxs.clone())
            else:
                out[i] = None

        return out





    # ========================================================
    # Predict
    # ========================================================
    def predict(
        self,
        df_test: pd.DataFrame,
        df_train: pd.DataFrame,
        use_policy: bool = False,
        policy_inference_trees: Optional[int] = None,
        *,
        policy_predictor_mode: Optional[str] = None,
        infer_reward: Optional[str] = None,
        algorithm: Optional[str] = None,
    ):
        c = self.cfg
        algo_rf = c.random_forest if algorithm is None else (algorithm == "rf")

        env_template = TabularEnv(
            df_train, c.feature_cols, c.target_col, c.n_bins, c.task,
            binning_strategy=c.binning_strategy, device=c.device
        )
        X_te = env_template._featurise(df_test, df_train, c.feature_cols, c.n_bins)
        X_tr, y_tr = env_template.X_full.clone(), env_template.y_full.clone()

        if algo_rf:
            preds = self._predict_random_forest(
                X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees,
                policy_predictor_mode=policy_predictor_mode, infer_reward=infer_reward
            )
        else:
            preds = self._predict_boosting(
                X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees,
                policy_predictor_mode=policy_predictor_mode, infer_reward=infer_reward
            )
        return preds.cpu().numpy()

    def _predict_in_batches(self, pred_fn, X, bs: int, device: str):
        out = []
        N = X.size(0)
        for s in range(0, N, bs):
            xb = X[s:s+bs].to(device, non_blocking=True)
            out.append(pred_fn(xb).to("cpu"))
        return torch.cat(out, dim=0)

    # ---------------- RF inference ----------------
    def _predict_random_forest(
        self,
        X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees,
        *, policy_predictor_mode: Optional[str], infer_reward: Optional[str],
    ):
        c = self.cfg
        device = X_tr.device

        y_train_target = (
            torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float)
            if c.task == "classification" else y_tr
        ).to(device)

        trees_to_use: List[List[int]] = []
        if use_policy:
            total_trees = policy_inference_trees if policy_inference_trees is not None else c.policy_inference_trees
            num_batches = math.ceil(total_trees / c.num_parallel)
            env_template.y = y_tr.clone().to(device)
            for _ in tqdm(range(num_batches), desc="Policy-based Tree Generation", leave=False):
                trees_in_batch = min(c.num_parallel, total_trees - len(trees_to_use))
                if trees_in_batch <= 0:
                    break
                # draw indices via shared sampler
                envs = [copy.copy(env_template) for _ in range(trees_in_batch)]
                idx_batches = [env_template.draw_indices(c.batch_size) for _ in range(trees_in_batch)]
                for env, idxs in zip(envs, idx_batches):
                    env.paths = []; env.open_leaves = 1; env.done = False; env.idxs = idxs
                batch_results = self.batched_rollout(
                    envs, temp=c.rollout_temperature, residuals=y_tr.to(device), beta=c.beta,
                    ras_counts=({} if c.redundancy_aware else None)
                )
                trees_to_use.extend([res[0] for res in batch_results if res])
        else:
            trees_to_use = self.ensemble

        if not trees_to_use:
            raise RuntimeError("The forest is empty.")

        infer_reward = (c.infer_reward_function if infer_reward is None else infer_reward) or c.reward_function

        reward_env = copy.copy(env_template)
        reward_env.y = y_tr.clone().to(device)
        reward_env.y_full = y_tr.clone().to(device)
        reward_env.reset(len(y_tr))

        if c.task == "classification":
            sum_preds = torch.zeros((X_te.shape[0], c.n_classes), device=device)
        else:
            sum_preds = torch.zeros(X_te.shape[0], device=device, dtype=torch.float32)
        total_weight = 0.0

        ensemble_scale = 1.0
        if infer_reward == "mse_ensemble" and c.task == "regression":
            R_scalar = self._ensemble_reward_mse(trees_to_use, X_tr, y_tr)
            ensemble_scale = float(R_scalar.item())

        for seq in tqdm(trees_to_use, desc="RF Prediction", leave=False):
            pred_fn = get_tree_predictor(
                seq, X_tr, y_train_target, self.tokenizer,
                min_child_size=c.min_child_size, min_gain=c.min_gain,
                predictor_mode=(policy_predictor_mode or c.policy_predictor_mode)
            )

            if infer_reward in ("none", "mse_ensemble"):
                w = 1.0 * ensemble_scale
            else:
                w = self._weight_for_tree(seq, reward_env, mode=infer_reward)

            sum_preds += w * pred_fn(X_te)
            total_weight += w

        if total_weight <= 0:
            raise RuntimeError("Total weight is zero.")
        preds = sum_preds / total_weight
        return torch.softmax(preds, dim=1) if c.task == "classification" else preds

    # ---------------- Boost inference (sequential) ----------------
    def _predict_boosting(
        self,
        X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees,
        *, policy_predictor_mode: Optional[str], infer_reward: Optional[str],
    ):
        c = self.cfg
        device = X_tr.device

        if c.task == "classification":
            test_preds  = torch.zeros((len(X_te), c.n_classes), device=device)
            train_preds = torch.zeros((len(y_tr), c.n_classes), device=device)
        else:
            test_preds  = torch.full((len(X_te),), self.y_mean, device=device, dtype=torch.float32)
            train_preds = torch.full_like(y_tr, self.y_mean, dtype=torch.float32, device=device)

        if not use_policy:
            # fallback: use stored ensemble as-is
            for fn in tqdm(self.boosting_ensemble, desc="Ensemble Prediction", leave=False):
                test_preds += c.boosting_lr * fn(X_te)
            return torch.softmax(test_preds, dim=1) if c.task == "classification" else test_preds

        tqdm.write("--- Generating Boosting Ensemble with Policy (Sequential Inference / v13 style) ---")
        total_trees = policy_inference_trees if policy_inference_trees is not None else c.updates
        num_batches = math.ceil(total_trees / c.num_parallel)

        # 1) residuals for candidate generation (computed once)
        if c.task == "classification":
            residuals = (
                torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float)
                - torch.softmax(train_preds, dim=1)
            )
        else:
            residuals = y_tr - train_preds

        env_template.y = residuals.clone().to(device)

        # 2) generate all candidates from current residuals
        candidate_trees: List[List[int]] = []
        for _ in tqdm(range(num_batches), desc="Policy-based Tree Generation", leave=False):
            envs = [copy.copy(env_template) for _ in range(c.num_parallel)]
            res = self.batched_rollout(
                envs,
                temp=c.rollout_temperature,
                residuals=residuals,
                beta=c.beta,
                ras_counts=({} if c.redundancy_aware else None),
            )
            candidate_trees.extend([r[0] for r in res if r])

        # 3) walk candidates sequentially, recomputing residuals before each add
        infer_reward = (c.infer_reward_function if infer_reward is None else infer_reward)
        for seq in tqdm(candidate_trees, desc="Sequential Boosting Prediction", leave=False):
            # refresh residuals wrt *current* train_preds
            if c.task == "classification":
                residuals = (
                    torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float)
                    - torch.softmax(train_preds, dim=1)
                )
            else:
                residuals = y_tr - train_preds

            # build tree predictor on the CURRENT residuals
            pred = get_tree_predictor(
                seq, X_tr, residuals, self.tokenizer,
                min_child_size=c.min_child_size, min_gain=c.min_gain,
                predictor_mode=(policy_predictor_mode or c.policy_predictor_mode)
            )
            contrib_tr = pred(X_tr)
            contrib_te = pred(X_te)

            # optional inference-time weighting (keep to v13: only variance for regression)
            if infer_reward in (None, "none"):
                w = 1.0
            elif infer_reward == "variance" and c.task == "regression":
                tok = torch.tensor([seq], device=device, dtype=torch.long)
                reward_env = copy.copy(env_template)
                reward_env.y = residuals.clone().to(device)
                w = float(torch.clamp(deltaE_split_gain_regression(tok, self.tokenizer, reward_env).sum(), min=1e-9).item())
            else:
                w = 1.0

            train_preds += c.boosting_lr * (w * contrib_tr)
            test_preds  += c.boosting_lr * (w * contrib_te)

        return torch.softmax(test_preds, dim=1) if c.task == "classification" else test_preds


    # -------- Local expansion utilities (new) --------
    def _leaf_prefixes_for_expansion(self, seq: List[int], *, max_depth: int, k: int) -> List[List[int]]:
        """Return up to k token prefixes that end right before a shallow LEAF."""
        if not seq or self.tokenizer is None:
            return []
        v = self.tokenizer.v
        s = seq[1:] if seq and seq[0] == v.BOS else seq[:]   # strip BOS
        if s and s[-1] == v.EOS:
            s = s[:-1]

        depth_stack = [0]
        pending = None
        out: List[Tuple[int, List[int]]] = []

        for pos, tid in enumerate(s):
            kind, _ = self.tokenizer.decode_one(tid)
            if pending is None:
                if kind == "feat":
                    if not depth_stack:
                        continue
                    pending = depth_stack.pop()
                elif kind == "leaf":
                    if depth_stack and depth_stack[-1] <= max_depth:
                        out.append((depth_stack[-1], [v.BOS] + s[:pos]))
                    if depth_stack:
                        depth_stack.pop()
                else:
                    pass
            else:
                if kind != "th":
                    pending = None
                    continue
                d = pending
                pending = None
                depth_stack.append(d + 1)  # right then left for LIFO
                depth_stack.append(d + 1)

        out.sort(key=lambda x: (x[0], len(x[1])))
        return [p for _, p in out[:k]]

    def _guided_rollout_one(self, env_template: TabularEnv, prefix: List[int],
                            residuals: torch.Tensor, beta: float,
                            *, force_first_split: bool = True):
        """Continue rollout from prefix, forcing the first decision to split."""
        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        END_TOKEN = v.EOS
        LEAF_TOKEN = v.split_start - 1

        env = copy.copy(env_template)
        env.y = residuals
        idxs = env.draw_indices(c.batch_size)
        env.reset(c.batch_size)
        env.idxs = idxs

        depth = deque([0])
        lo_stack = deque([torch.zeros(v.num_feat, dtype=torch.long, device=device)])
        hi_stack = deque([torch.full((v.num_feat,), v.num_th - 1, dtype=torch.long, device=device)])
        row_stack = deque([torch.arange(idxs.numel(), device=device)])
        seq = prefix[:]

        # replay prefix
        pending = None
        for t in seq[1:]:
            if t == END_TOKEN:
                env.done = True
                break
            kind, val = self.tokenizer.decode_one(t)
            if pending is None:
                if kind == "leaf":
                    if depth: depth.pop(); lo_stack.pop(); hi_stack.pop(); row_stack.pop()
                elif kind == "feat" and depth:
                    d0 = depth.pop()
                    lo_t, hi_t = lo_stack.pop(), hi_stack.pop()
                    rows = row_stack.pop()
                    pending = (d0, lo_t, hi_t, rows, val)
            else:
                if kind != "th":
                    pending = None
                    continue
                d0, lo_t, hi_t, rows, f_idx = pending
                pending = None
                Xb = env.X_full[env.idxs]
                fv = Xb.index_select(0, rows)[:, f_idx]
                m = fv <= val
                rows_L = rows[m]; rows_R = rows[~m]
                lo_L, hi_L = lo_t.clone(), hi_t.clone(); hi_L[f_idx] = torch.minimum(hi_L[f_idx], torch.as_tensor(val, device=device))
                lo_R, hi_R = lo_t.clone(), hi_t.clone(); lo_R[f_idx] = torch.maximum(lo_R[f_idx], torch.as_tensor(val + 1, device=device))
                depth.append(d0 + 1); lo_stack.append(lo_R); hi_stack.append(hi_R); row_stack.append(rows_R)
                depth.append(d0 + 1); lo_stack.append(lo_L); hi_stack.append(hi_L); row_stack.append(rows_L)
                env.step(("th", int(val)))

        if env.done:
            if seq[-1] != END_TOKEN:
                seq.append(END_TOKEN)
            return (seq, env.get_prior(beta).item(), idxs)

        first = True
        while depth:
            pad = torch.tensor([seq], device=device)
            logits, _ = self.pf(pad)
            last = logits[0, -1, :]
            mask = torch.zeros(v.size(), dtype=torch.bool, device=device)

            # allow LEAF / FEAT set
            mask[LEAF_TOKEN] = True
            can_split = (depth[-1] < c.max_depth)
            rows_rel = row_stack[-1]
            Xb = env.X_full[env.idxs]
            Xleaf = Xb.index_select(0, rows_rel)
            if can_split and Xleaf.size(0) > 1:
                n_leaf = Xleaf.size(0)
                valid_feats = []
                for f in range(Xleaf.size(1)):
                    bf = Xleaf[:, f]
                    uniq, counts = torch.unique(bf, return_counts=True)
                    if uniq.numel() < 2:
                        continue
                    if c.min_child_size and c.min_child_size > 1:
                        csum = counts.cumsum(0)[:-1]
                        if not bool(((csum >= c.min_child_size) & ((n_leaf - csum) >= c.min_child_size)).any()):
                            continue
                    valid_feats.append(f)
                if valid_feats:
                    feat_ids = v.split_start + torch.as_tensor(valid_feats, device=device)
                    mask[feat_ids] = True

            if c.rollout_temperature is not None:
                pass
            if first and force_first_split:
                mask[LEAF_TOKEN] = False
                mask[v.EOS] = False

            if seq[-1] < mask.size(0):
                mask[seq[-1]] = False

            tok = _safe_sample(last.unsqueeze(0), mask.unsqueeze(0), c.rollout_temperature)[0].item()
            seq.append(tok)
            first = False
            if tok == v.EOS:
                break
            if tok == LEAF_TOKEN:
                depth.pop(); lo_stack.pop(); hi_stack.pop(); row_stack.pop()
                continue

            kind, f_idx = self.tokenizer.decode_one(tok)
            if kind != "feat" or not depth:
                continue

            pad = torch.tensor([seq], device=device)
            logits, _ = self.pf(pad)
            last_th = logits[0, -1, :]

            mask2 = torch.zeros(v.size(), dtype=torch.bool, device=device)
            th_base = v.split_start + v.num_feat
            lo_top = lo_stack.pop(); hi_top = hi_stack.pop(); rows_rel = row_stack.pop()
            bf = Xleaf[:, f_idx]
            uniq, counts = torch.unique(bf, sorted=True, return_counts=True)
            cand_t = uniq[:-1]
            if c.min_child_size and c.min_child_size > 1:
                csum = counts.cumsum(0)[:-1]
                keep = (csum >= c.min_child_size) & ((bf.numel() - csum) >= c.min_child_size)
                cand_t = cand_t[keep]
            lo_f = int(lo_top[f_idx].item()); hi_f = int(hi_top[f_idx].item())
            if cand_t.numel() > 0:
                cand_t = cand_t[(cand_t >= lo_f) & (cand_t <= hi_f)]
            if cand_t.numel() > 0:
                th_ids = th_base + cand_t.to(device=device, dtype=torch.long)
                mask2[th_ids] = True
            mask2[v.EOS] = False; mask2[LEAF_TOKEN] = False
            if seq[-1] < mask2.size(0):
                mask2[seq[-1]] = False

            t_tok = _safe_sample(last_th.unsqueeze(0), mask2.unsqueeze(0), c.rollout_temperature)[0].item()
            seq.append(t_tok)

            _, t_idx = self.tokenizer.decode_one(t_tok)
            fv = Xb.index_select(0, rows_rel)[:, f_idx]
            m = fv <= t_idx
            rows_L = rows_rel[m]; rows_R = rows_rel[~m]
            lo_L, hi_L = lo_top.clone(), hi_top.clone(); hi_L[f_idx] = torch.minimum(hi_L[f_idx], torch.as_tensor(t_idx, device=device))
            lo_R, hi_R = lo_top.clone(), hi_top.clone(); lo_R[f_idx] = torch.maximum(lo_R[f_idx], torch.as_tensor(t_idx + 1, device=device))
            depth.append(depth.pop() + 1); lo_stack.append(lo_R); hi_stack.append(hi_R); row_stack.append(rows_R)
            depth.append(depth[-1]);       lo_stack.append(lo_L); hi_stack.append(hi_L); row_stack.append(rows_L)
            env.step(("th", int(t_idx)))

        if seq[-1] != END_TOKEN:
            seq.append(END_TOKEN)
        return (seq, env.get_prior(beta).item(), idxs)

    def _local_expand_best(self, env_template: TabularEnv, residuals: torch.Tensor) -> List[Tuple[List[int], float]]:
        """Generate candidates by expanding shallow leaves of the current best tree."""
        if not self._best_tree_seq or not self.cfg.use_local_expansion:
            return []
        c = self.cfg
        prefixes = self._leaf_prefixes_for_expansion(
            self._best_tree_seq,
            max_depth=c.local_expand_max_depth,
            k=c.local_expand_k_leaves,
        )
        if not prefixes:
            return []

        tuples: List[Tuple[List[int], float]] = []
        for pref in prefixes:
            for _ in range(max(1, c.local_expand_per_leaf)):
                res = self._guided_rollout_one(env_template, pref, residuals, c.beta, force_first_split=True)
                if not res:
                    continue
                seq, prior, idxs = res
                reward_env = copy.copy(env_template)
                reward_env.idxs = idxs.to(c.device)
                reward_env.y_full = env_template.y_full
                reward_env.X_full = env_template.X_full
                reward_env.y = residuals
                r = self._weight_for_tree(seq, reward_env, mode=c.reward_function)
                self.replay_buffer.add(r, seq, prior, idxs.cpu())
                tuples.append((seq, prior))
        return tuples

    # sklearn compat
    def get_params(self, deep=True): return asdict(self.cfg)
    def set_params(self, **params):
        for k, v in params.items(): setattr(self.cfg, k, v)
        return self
