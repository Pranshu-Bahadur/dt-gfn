# src/trainer.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Deque
from collections import deque
import copy
import math
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from src.tokenizer import Tokenizer, Vocab
from src.env import TabularEnv
from src.policy import PolicyPaperMLP  # includes prefix-decay readout variant
from src.utils import (
    ReplayBuffer,
    tb_loss,
    fl_loss,
    _safe_sample,
    get_tree_predictor,
    deltaE_split_gain_regression,
    deltaE_split_gain_classification,
    deltaE_split_gain_sse,
    calculate_bayesian_reward,
    calculate_bayesian_reward_regression,
    uniform_backward_log_prob,
)

# -----------------------------
# Helper: TB on log-domain reward
# -----------------------------
@torch.jit.script
def tb_loss_logR(
    log_pf: torch.Tensor, log_pb: torch.Tensor, log_z: torch.Tensor,
    logR: torch.Tensor, prior: torch.Tensor
) -> torch.Tensor:
    if log_pf.dim() == 1: log_pf = log_pf.unsqueeze(0)
    if log_pb.dim() == 1: log_pb = log_pb.unsqueeze(0)
    lp = log_pf.sum(-1)
    lb = log_pb.sum(-1)
    diff = (log_z.squeeze() + lp - lb - logR.squeeze())
    return (diff * diff).mean()


def _lin_anneal(start: float, end: float, step: int, total: int, begin_frac: float, end_frac: float) -> float:
    if total <= 0:
        return end
    b = int(total * begin_frac)
    e = int(total * end_frac)
    if step <= b:
        return start
    if step >= e:
        return end
    t = (step - b) / max(1, e - b)
    return float(start + t * (end - start))


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
    redundancy_aware: bool = True

    # Policy network
    lstm_hidden: int = 256
    mlp_layers: int = 3
    mlp_width: int = 256
    lr: float = 1e-4
    policy_type: str = "mlp"                      # "mlp" | "transformer"

    # Backward policy choice
    backward_policy: str = "uniform"              # "uniform" | "network" | "zero"

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

    # NEW knobs
    rollout_temperature: float = 0.0               # base sampling temp for rollouts
    min_child_size: int = 20                       # predictor split guard
    min_gain: float = 0.0                          # min impurity reduction

    # Stabilizers (EMA used only in non-bayesian path)
    reward_baseline_momentum: float = 0.95         # EMA of logR for centering (legacy path)
    grad_clip: float = 1.0                         # global grad-norm clip
    replay_tau: float = 1.0                        # replay sampling temperature (1=softmax, 0=top-k)

    # training reward scope
    training_reward_scope: str = "per_tree"        # "per_tree" | "ensemble"
    ensemble_reward_metric: str = "mse"            # (regression-only for now)

    # inference-time weighting reward (for RF & Boost)
    infer_reward_function: Optional[str] = None    # None -> use training reward; "none" -> equal weights

    # policy-based predictor mode hint
    policy_predictor_mode: str = "dirichlet"       # "dirichlet" | "mean"

    # Track best single-tree train accuracy while training (classification)
    show_best_tree_acc: bool = False

    # === New schedule knobs for anneals ===
    subtb_enabled: bool = True

    # temperature anneal
    temp_start: float = 1.0
    temp_end: float = 0.1
    temp_begin_frac: float = 0.5
    temp_end_frac: float = 1.0

    # tau anneal (for logR scaling)
    tau_train_start: float = 3.0
    tau_train_end: float = 1.0
    tau_begin_frac: float = 0.0
    tau_end_frac: float = 0.8

    # logR stabilizers
    logr_clamp: float = 100.0                      # clamp on centered logR (None to disable)
    logr_target_std: float = 50.0                  # shrink only when batch std > target


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

        # EMA baseline for legacy TB reward centering (non-bayesian path)
        self._logR_ema = torch.tensor(0.0, device=cfg.device)

        # best-single-tree tracker
        self._best_tree_seq: Optional[List[int]] = None
        self._best_tree_acc: float = 0.0

        # counters for anneals
        self._total_updates: int = 0
        self._cur_update: int = 0

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
        if c.policy_type == "transformer":
            from src.policy import PolicyTransformer
            self.pf = PolicyTransformer(
                v.size(), d_model=c.lstm_hidden, n_layers=c.mlp_layers, n_heads=2,
                d_ff=c.mlp_width * 4, pad_id=v.PAD
            ).to(c.device)
        else:
            self.pf = PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device)

        try:
            if hasattr(torch, "compile"):
                self.pf = torch.compile(self.pf)  # type: ignore
        except Exception:
            pass
        self.pf = torch.jit.script(self.pf)

        if c.backward_policy == "network":
            if c.policy_type == "transformer":
                from src.policy import PolicyTransformer
                self.pb = PolicyTransformer(
                    v.size(), d_model=c.lstm_hidden, n_layers=c.mlp_layers, n_heads=2,
                    d_ff=c.mlp_width * 4, pad_id=v.PAD
                ).to(c.device)
            else:
                self.pb = PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device)
            try:
                if hasattr(torch, "compile"):
                    self.pb = torch.compile(self.pb)  # type: ignore
            except Exception:
                pass
            self.pb = torch.jit.script(self.pb)
        else:
            self.pb = None  # "uniform" or "zero" handled downstream

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

        self.replay_buffer = ReplayBuffer(capacity=200)

        if c.beta is None:
            c.beta = math.log(4) + math.log(len(c.feature_cols))
            tqdm.write(f"[trainer] β (structure prior) = {c.beta:.4f}")

        if c.random_forest:
            self._fit_dt_gfn_random_forest(env_template, y_true, X_binned, opt_list, sch_list)
        else:
            self._fit_boost_gfn(env_template, y_true, X_binned, opt_list, sch_list)

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
    # Batched policy update (one step)
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

        # ====== New path: log-domain TB for Bayesian reward ======
        if c.reward_function == "bayesian" and ensemble_reward_override is None:
            with autocast(enabled=c.amp):
                # forward policy logs
                log_pf = self.pf.log_prob(padded)

                # backward policy logs
                if self.pb is not None and c.backward_policy == "network":
                    flipped = torch.nn.utils.rnn.pad_sequence(
                        [t.flip(0) for t in toks], batch_first=True, padding_value=v.PAD
                    )
                    log_pb = self.pb.log_prob(flipped)
                elif c.backward_policy == "uniform":
                    log_pb = uniform_backward_log_prob(padded, self.tokenizer, c.max_depth)
                else:  # "zero" or anything else ⇒ zeros
                    log_pb = torch.zeros_like(log_pf)

                # per-trajectory log R (computed on the SAME indices each seq was built on)
                logR_list = []
                for i, t in enumerate(toks):
                    tok_i = padded[i:i+1, :t.numel()]
                    # use calculate_bayesian_reward and take log (can be overridden to logR upstream)
                    R_i = calculate_bayesian_reward(tok_i, self.tokenizer, env_template, c.beta)
                    logR_i = torch.log(R_i.clamp_min(1e-9))
                    logR_list.append(logR_i.squeeze())
                logR_batch = torch.stack(logR_list, dim=0)  # [B]

                # per-batch centering (in log-space)
                logR_centered = logR_batch - logR_batch.mean().detach()

                # optional std-shrink (only if variance is large)
                if c.logr_target_std is not None and c.logr_target_std > 0:
                    std = logR_batch.std().detach()
                    scale = torch.clamp(std / float(c.logr_target_std), min=1.0)
                    logR_centered = logR_centered / scale

                # optional clamp
                if c.logr_clamp is not None and c.logr_clamp > 0:
                    logR_centered = logR_centered.clamp(-float(c.logr_clamp), float(c.logr_clamp))

                # τ anneal
                tau_now = _lin_anneal(c.tau_train_start, c.tau_train_end,
                                      getattr(self, "_cur_update", 1),
                                      max(1, getattr(self, "_total_updates", 1)),
                                      c.tau_begin_frac, c.tau_end_frac)
                logR_scaled = logR_centered / max(1e-6, float(tau_now))

                # TB objective (SubTB or regular)
                if c.subtb_enabled:
                    B, Tm1 = log_pf.size()
                    ar = torch.arange(Tm1, device=device).unsqueeze(0)        # (1, T-1)
                    u  = torch.randint(0, Tm1 + 1, (B,), device=device)       # cut per sample
                    mask_pf = ar < u.unsqueeze(1)                              # prefix for pf
                    mask_pb = ~mask_pf                                         # suffix for pb
                    lp = (log_pf * mask_pf).sum(-1)
                    lb = (log_pb * mask_pb).sum(-1)
                    diff = (self.log_z.squeeze() + lp - lb - logR_scaled.squeeze())
                    loss = (diff * diff).mean()
                    tb_val, fl_val = loss, None
                else:
                    loss = tb_loss_logR(log_pf, log_pb, self.log_z, logR_scaled, priors_tensor)
                    tb_val, fl_val = loss, None

        # ====== Legacy / non-bayesian path (unchanged core logic) ======
        else:
            with autocast(enabled=c.amp):
                log_pf = self.pf.log_prob(padded)

                if self.pb is not None and c.backward_policy == "network":
                    flipped = torch.nn.utils.rnn.pad_sequence(
                        [t.flip(0) for t in toks], batch_first=True, padding_value=v.PAD
                    )
                    log_pb = self.pb.log_prob(flipped)
                elif c.backward_policy == "uniform":
                    log_pb = uniform_backward_log_prob(padded, self.tokenizer, c.max_depth)
                else:
                    log_pb = torch.zeros_like(log_pf)

                logF = self.pf.log_F(padded)

                if ensemble_reward_override is not None:
                    R = ensemble_reward_override.expand(len(seqs)).to(device)
                    logR_batch = torch.log(R + 1e-9)
                    self._logR_ema = (
                        self.cfg.reward_baseline_momentum * self._logR_ema
                        + (1 - self.cfg.reward_baseline_momentum) * logR_batch.mean().detach()
                    )
                    R_centered = torch.exp(logR_batch - self._logR_ema)
                    l_tb = tb_loss(log_pf, log_pb, self.log_z, R_centered, priors_tensor)
                    loss = l_tb
                    tb_val, fl_val = l_tb, None
                else:
                    reward_env = copy.copy(env_template)
                    R_list, dR_list = [], []
                    for i, t in enumerate(toks):
                        tok_i = padded[i:i+1, :t.numel()]
                        target = targets[i]
                        reward_env.y = target
                        reward_env.reset(len(target))
                        R_t, dR = self._per_tree_reward(tok_i, reward_env)
                        R_list.append(R_t.squeeze())
                        if dR is not None:
                            dR_list.append(dR.squeeze(0))

                    R = torch.stack(R_list, dim=0)

                    if self.cfg.reward_function == 'bayesian':
                        logR_batch = torch.log(R + 1e-9)
                        self._logR_ema = (
                            self.cfg.reward_baseline_momentum * self._logR_ema
                            + (1 - self.cfg.reward_baseline_momentum) * logR_batch.mean().detach()
                        )
                        R_centered = torch.exp(logR_batch - self._logR_ema)
                        l_tb = tb_loss(log_pf, log_pb, self.log_z, R_centered, priors_tensor)
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

                        logR_batch = torch.log(R + 1e-9)
                        self._logR_ema = (
                            self.cfg.reward_baseline_momentum * self._logR_ema
                            + (1 - self.cfg.reward_baseline_momentum) * logR_batch.mean().detach()
                        )
                        R_centered = torch.exp(logR_batch - self._logR_ema)
                        l_tb = tb_loss(log_pf, log_pb, self.log_z, R_centered, priors_tensor)
                        l_fl = fl_loss(logF, log_pf, log_pb, dR_shaped)
                        loss = l_tb + l_fl
                        tb_val, fl_val = l_tb, l_fl

        # ----- Backprop + clip -----
        self.scaler.scale(loss).backward()
        for opt in optimizers:
            try:
                self.scaler.unscale_(opt)
            except Exception:
                pass
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            try:
                torch.nn.utils.clip_grad_norm_(self.pf.parameters(), self.cfg.grad_clip)
            except Exception:
                pass
            if self.pb is not None:
                try:
                    torch.nn.utils.clip_grad_norm_(self.pb.parameters(), self.cfg.grad_clip)
                except Exception:
                    pass
            try:
                torch.nn.utils.clip_grad_norm_([self.log_z], self.cfg.grad_clip)
            except Exception:
                pass
        for opt in optimizers:
            self.scaler.step(opt)
        self.scaler.update()

        self.replay_buffer.mark_policy_update()

        tb_loss_acc = float(tb_val.item())
        fl_loss_acc = float(fl_val.item()) if fl_val is not None else 0.0
        return tb_loss_acc, fl_loss_acc

    # ========================================================
    # RF training (policy + gen) with anneals
    # ========================================================
    def _fit_dt_gfn_random_forest(self, env_template, y_true, X_binned, optimizers, schedulers):
        c = self.cfg
        tqdm.write("--- Starting DT-GFN (Random Forest) Training ---")

        # set total updates for anneals
        self._total_updates = int(c.updates)

        y_target_for_reward = (
            torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float)
            if c.task == "classification" else y_true.clone()
        )
        env_template.y = y_true.clone()

        all_tuples_last = []

        PRINT_EVERY = 10
        for upd in range(1, c.updates + 1):
            self._cur_update = upd

            # annealed rollout temperature
            temp_now = _lin_anneal(
                c.temp_start, c.temp_end, upd, c.updates, c.temp_begin_frac, c.temp_end_frac
            )

            forward_tuples = self._collect_rollouts(env_template, temp=temp_now, residuals=y_true, beta=c.beta)
            replay_tuples  = self.sample_replay(c.top_k_trees)
            all_tuples     = forward_tuples + replay_tuples
            if not all_tuples:
                for sch in schedulers: sch.step()
                continue

            all_tuples_with_targets = [(seq, prior, y_target_for_reward) for seq, prior in all_tuples]
            tb_val, _ = self._update_policy(all_tuples_with_targets, env_template, optimizers, ensemble_reward_override=None)
            for sch in schedulers:
                sch.step()

            all_tuples_last = all_tuples
            if upd % PRINT_EVERY == 0 or upd == 1 or upd == c.updates:
                tau_now = _lin_anneal(
                    c.tau_train_start, c.tau_train_end, upd, c.updates, c.tau_begin_frac, c.tau_end_frac
                )
                tqdm.write(
                    f"Update {upd}/{c.updates} | "
                    f"{'subTB' if c.subtb_enabled else 'TB'}: {tb_val:.4f} | "
                    f"τ={tau_now:.2f} | temp={temp_now:.2f} | Trees(batch): {len(all_tuples)}"
                )

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
    # Rollouts (silent, annealed temp)
    # ========================================================
    def _collect_rollouts(self, env_template, temp, residuals, beta, ras_counts=None):
        forward_tuples, done = [], 0
        # compute annealed temperature from counters (ignore incoming temp if you want)
        if hasattr(self, "_cur_update") and hasattr(self, "_total_updates"):
            temp = _lin_anneal(
                self.cfg.temp_start, self.cfg.temp_end,
                getattr(self, "_cur_update", 1),
                max(1, getattr(self, "_total_updates", 1)),
                self.cfg.temp_begin_frac, self.cfg.temp_end_frac
            )

        while done < self.cfg.rollouts:
            batch = min(self.cfg.num_parallel, self.cfg.rollouts - done)
            envs = [copy.copy(env_template) for _ in range(batch)]
            idx_batches = [env_template.draw_indices(self.cfg.batch_size) for _ in range(batch)]
            for env, idxs in zip(envs, idx_batches):
                env.y = residuals
                env.paths = []; env.open_leaves = 1; env.done = False; env.idxs = idxs
            results = self.batched_rollout(envs, temp, residuals, beta, ras_counts)
            for res in results:
                if not res: continue
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
        return forward_tuples

    def sample_replay(self, k: int, REFRESH_INTERVAL: int = 5) -> List[Tuple[List[int], float]]:
        buf = self.replay_buffer
        if not buf or not buf.data:
            return []

        if self.cfg.backward_policy != "network":
            entries = buf.sample_tempered(k, tau=self.cfg.replay_tau)
            return [(e[1], e[2]) for e in entries]

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

    # ========================================================
    # Canonical batched rollout (EOS-safe, per-feature bin limits)
    # ========================================================
    def batched_rollout(self, envs, temp, residuals, beta, ras_counts: Optional[dict] = None):
        """
        Parallel rollouts with ONLY contradiction guards + per-feature bin limits.

        • hi-stacks initialized from each env's per-feature effective bins:
            hi0[j] = per_feat_bins[j] - 1   (legal thresholds t ∈ [0, bins_j-2])
          We store hi=B-1 so the window is [lo, hi), and choose t in [lo, hi-1].
        • FEAT feasible iff (hi - lo) >= 1 and depth < max_depth.
        • THRESH candidates are built from window t ∈ [lo_f, hi_f).
        • EOS safety: THRESH mask is guaranteed non-empty per row; _safe_sample can’t inject EOS there.
        """
        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        num = len(envs)
        END_TOKEN  = v.EOS
        LEAF_TOKEN = self.tokenizer._leaf(0)

        # --- init envs ---
        for env in envs:
            env.y = residuals
            env.reset(c.batch_size)

        # sequences start with BOS
        seqs: List[List[int]] = [[v.BOS] for _ in range(num)]

        # Cache per-env binned features for routing rows; idxs are fixed per rollout
        Xb_cache = [envs[i].X_full[envs[i].idxs] for i in range(num)]

        # Stacks per env (DFS over implicit tree)
        depths: List[Deque[int]] = [deque([0]) for _ in range(num)]
        lo_stacks: List[Deque[torch.Tensor]] = [
            deque([torch.zeros(v.num_feat, dtype=torch.long, device=device)]) for _ in range(num)
        ]
        hi_stacks: List[Deque[torch.Tensor]] = []
        for i in range(num):
            hi0 = (envs[i].per_feat_bins.to(device=device) - 1).clamp_min(0)  # shape [F]
            hi_stacks.append(deque([hi0]))
        row_stacks: List[Deque[torch.Tensor]] = [
            deque([torch.arange(envs[i].idxs.numel(), device=device)]) for i in range(num)
        ]

        # Precompute token id blocks
        feat_token_ids = (v.split_start + torch.arange(v.num_feat, device=device, dtype=torch.long))  # [F]
        th_base = v.split_start + v.num_feat
        th_ids_block = th_base + torch.arange(v.num_th, device=device, dtype=torch.long)              # [#th_bins]

        def _mark_done_if_finished(ti: int):
            if not depths[ti]:
                envs[ti].done = True

        active = [i for i in range(num) if not envs[i].done]
        out: List[Optional[Tuple[List[int], float, torch.Tensor]]] = [None] * num

        with torch.no_grad():
            while active:
                # -------- decision 1: FEAT or LEAF --------
                pad = torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(seqs[i], device=device, dtype=torch.long) for i in active],
                    batch_first=True, padding_value=v.PAD
                )
                logits, _ = self.pf(pad)                      # (B, T, V)
                last = logits[:, -1, :]                       # (B, V)

                # redundancy-aware suppression
                if ras_counts:
                    for bi, oidx in enumerate(active):
                        cnt = ras_counts.get(tuple(seqs[oidx]))
                        if cnt:
                            last[bi].sub_(cnt * 1e9)

                B = len(active)
                V = v.size()
                mask1 = torch.zeros((B, V), dtype=torch.bool, device=device)
                mask1[:, LEAF_TOKEN] = True  # always allow closing a branch

                # feasibility: (hi - lo) >= 1 AND depth < max_depth
                lo_batch = torch.stack([lo_stacks[i][-1] for i in active], dim=0)  # (B, F)
                hi_batch = torch.stack([hi_stacks[i][-1] for i in active], dim=0)  # (B, F)
                feas = (hi_batch - lo_batch) >= 1                                  # (B, F)
                depth_vec = torch.tensor([depths[i][-1] if depths[i] else 0 for i in active],
                                         device=device, dtype=torch.long)          # (B,)
                feas &= (depth_vec < c.max_depth).view(B, 1)                        # (B, F)
                if feas.any():
                    mask1[:, feat_token_ids] = feas

                toks1 = _safe_sample(last, mask1, temp, eos_id=END_TOKEN)          # (B,)

                # Collect thresholds to resolve this round
                need_threshold_idx: List[int] = []
                need_threshold_meta: List[Tuple[int,int,int]] = []  # (env_idx, f_idx, depth_before)
                still_for_round: List[int] = []

                for bi, oidx in enumerate(active):
                    if not mask1[bi].any():
                        envs[oidx].done = True
                        continue

                    tok = int(toks1[bi].item())
                    seqs[oidx].append(tok)
                    if ras_counts is not None:
                        key = tuple(seqs[oidx])
                        ras_counts[key] = ras_counts.get(key, 0) + 1

                    if tok == LEAF_TOKEN:
                        envs[oidx].step(('leaf', 0))
                        depths[oidx].pop(); lo_stacks[oidx].pop(); hi_stacks[oidx].pop(); row_stacks[oidx].pop()
                        _mark_done_if_finished(oidx)
                        if not envs[oidx].done:
                            still_for_round.append(oidx)
                    else:
                        kind, f_idx = self.tokenizer.decode_one(tok)  # 'feat', f
                        envs[oidx].step((kind, f_idx))
                        d0 = depths[oidx].pop()
                        need_threshold_idx.append(oidx)
                        need_threshold_meta.append((oidx, f_idx, d0))

                # -------- decision 2: THRESHOLD --------
                if need_threshold_idx:
                    sub_pad = torch.nn.utils.rnn.pad_sequence(
                        [torch.as_tensor(seqs[i], device=device, dtype=torch.long) for i in need_threshold_idx],
                        batch_first=True, padding_value=v.PAD
                    )
                    sub_logits, _ = self.pf(sub_pad)
                    last_th = sub_logits[:, -1, :]                                # (M, V)

                    M = len(need_threshold_idx)
                    mask2 = torch.zeros((M, V), dtype=torch.bool, device=device)

                    lo_mat = torch.stack([lo_stacks[i][-1] for i in need_threshold_idx], dim=0)  # (M, F)
                    hi_mat = torch.stack([hi_stacks[i][-1] for i in need_threshold_idx], dim=0)  # (M, F)
                    f_ids = torch.tensor([f for (_, f, _) in need_threshold_meta],
                                         device=device, dtype=torch.long)                         # (M,)
                    lo_f = lo_mat.gather(1, f_ids.view(-1, 1)).squeeze(1)                         # (M,)
                    hi_f = hi_mat.gather(1, f_ids.view(-1, 1)).squeeze(1)                         # (M,)

                    ar = torch.arange(v.num_th, device=device, dtype=torch.long).view(1, -1)      # (1, #th_bins)
                    allowed_block = (ar >= lo_f.view(-1, 1)) & (ar < hi_f.view(-1, 1))            # (M, #th_bins)
                    mask2[:, th_ids_block] = allowed_block

                    # Disallow EOS/LEAF at threshold decision
                    mask2[:, END_TOKEN] = False
                    mask2[:, LEAF_TOKEN] = False

                    # HARD GUARANTEE: non-empty mask rows at THRESH stage
                    empty_rows = ~mask2.any(dim=1)
                    if empty_rows.any():
                        t_fb = torch.minimum(lo_f, hi_f - 1).clamp_min(0).clamp_max(v.num_th - 1)
                        mask2[empty_rows] = False
                        tf = t_fb[empty_rows].view(-1, 1)
                        mask2[empty_rows, th_ids_block] = (ar == tf).squeeze(1)

                    toks2 = _safe_sample(last_th, mask2, temp, eos_id=END_TOKEN)   # (M,)

                    # Apply thresholds and push children
                    for si, oidx in enumerate(need_threshold_idx):
                        t_tok = int(toks2[si].item())
                        seqs[oidx].append(t_tok)
                        if ras_counts is not None:
                            key = tuple(seqs[oidx]); ras_counts[key] = ras_counts.get(key, 0) + 1

                        _, t_idx = self.tokenizer.decode_one(t_tok)
                        t_idx = int(t_idx)

                        rows_rel = row_stacks[oidx].pop()
                        f_idx = need_threshold_meta[si][1]
                        fv = Xb_cache[oidx].index_select(0, rows_rel)[:, f_idx]
                        m = fv <= t_idx
                        rows_L = rows_rel[m]
                        rows_R = rows_rel[~m]

                        lo_top = lo_stacks[oidx].pop()
                        hi_top = hi_stacks[oidx].pop()
                        lo_L, hi_L = lo_top.clone(), hi_top.clone()
                        hi_L[f_idx] = torch.minimum(hi_L[f_idx], torch.as_tensor(t_idx, device=device))
                        lo_R, hi_R = lo_top.clone(), hi_top.clone()
                        lo_R[f_idx] = torch.maximum(lo_R[f_idx], torch.as_tensor(t_idx + 1, device=device))

                        d0 = need_threshold_meta[si][2]
                        depths[oidx].append(d0 + 1); lo_stacks[oidx].append(lo_R); hi_stacks[oidx].append(hi_R); row_stacks[oidx].append(rows_R)
                        depths[oidx].append(d0 + 1); lo_stacks[oidx].append(lo_L); hi_stacks[oidx].append(hi_L); row_stacks[oidx].append(rows_L)

                        envs[oidx].step(('th', t_idx))
                        if depths[oidx]:
                            still_for_round.append(oidx)
                        else:
                            envs[oidx].done = True

                # advance
                active = still_for_round

        # finalize outputs
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
            for fn in tqdm(self.boosting_ensemble, desc="Ensemble Prediction", leave=False):
                test_preds += c.boosting_lr * fn(X_te)
            return torch.softmax(test_preds, dim=1) if c.task == "classification" else test_preds

        tqdm.write("--- Generating Boosting Ensemble with Policy (Sequential Inference) ---")
        total_trees = policy_inference_trees if policy_inference_trees is not None else c.updates
        num_batches = math.ceil(total_trees / c.num_parallel)

        if c.task == "classification":
            residuals = (
                torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float)
                - torch.softmax(train_preds, dim=1)
            )
        else:
            residuals = y_tr - train_preds

        env_template.y = residuals.clone().to(device)

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

        infer_reward = (c.infer_reward_function if infer_reward is None else infer_reward)
        for seq in tqdm(candidate_trees, desc="Sequential Boosting Prediction", leave=False):
            if c.task == "classification":
                residuals = (
                    torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float)
                    - torch.softmax(train_preds, dim=1)
                )
            else:
                residuals = y_tr - train_preds

            pred = get_tree_predictor(
                seq, X_tr, residuals, self.tokenizer,
                min_child_size=c.min_child_size, min_gain=c.min_gain,
                predictor_mode=(policy_predictor_mode or c.policy_predictor_mode)
            )
            contrib_tr = pred(X_tr)
            contrib_te = pred(X_te)

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

    # sklearn compat
    def get_params(self, deep=True): return asdict(self.cfg)
    def set_params(self, **params):
        for k, v in params.items(): setattr(self.cfg, k, v)
        return self
