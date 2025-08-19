
## `src/trainer.py`
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Deque, Dict
from collections import deque, defaultdict
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
    # optional: uniform backward surrogate (not strictly required below)
    uniform_backward_log_prob,
)

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

    # === Redundancy & STOP controls ===
    redundancy_aware: bool = True                  # master switch
    redundancy_lambda_intra: float = 1.0           # penalty per local (intra-trajectory) prefix hit
    redundancy_lambda_inter: float = 0.25          # penalty per global (inter-trajectory) prefix hit
    redundancy_decay: float = 0.995                # exponential decay each policy update
    redundancy_ngram: int = 4                      # how many-prefix length to track (1..N)
    dedup_sequences: bool = True                   # skip adding exact-duplicate sequences to buffer

    allow_early_stop: bool = True                  # global STOP (EOS) may be chosen mid-build
    min_decisions_before_stop: int = 1             # require at least this many splits before STOP
    stop_bias: float = 0.0                         # +ve encourages STOP; -ve discourages

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

    # NEW knobs
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
    show_best_tree_acc: bool = True
    viz_every=10
    viz_dir="runs/trees"
    viz_format="png"

    tb_reward_temperature: float = 10.0
    tb_reward_standardize: bool = True



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

        # === Global redundancy tracking ===
        # counts over token-prefixes observed historically across *all* trajectories
        # keys are tuples of token IDs (prefixes); values are float counts with decay
        self.global_prefix_counts: Dict[Tuple[int, ...], float] = defaultdict(float)
        # cache of fully built sequences to deduplicate (exact match)
        self._seen_sequences: set[Tuple[int, ...]] = set()

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

        self.replay_buffer = ReplayBuffer(capacity=100000)

        if c.beta is None:
            c.beta = math.log(4) + math.log(len(c.feature_cols))
            tqdm.write(f"[trainer] β (structure prior) = {c.beta:.4f}")

        if c.random_forest:
            self._fit_dt_gfn_random_forest(env_template, y_true, X_binned, optimizers, schedulers)
        else:
            self._fit_boost_gfn(env_template, y_true, X_binned, optimizers, schedulers)

        return self

    # ========================================================
    # Reward helpers (unchanged)
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
    # Redundancy helpers
    # ========================================================
    def _decay_global_prefix_counts(self):
        """Exponential decay of inter-trajectory prefix counts to keep exploration fresh."""
        if not self.global_prefix_counts:
            return
        decay = self.cfg.redundancy_decay
        to_del = []
        for k, v in self.global_prefix_counts.items():
            v *= decay
            if v < 1e-3:
                to_del.append(k)
            else:
                self.global_prefix_counts[k] = v
        for k in to_del:
            del self.global_prefix_counts[k]

    def _register_sequence_prefixes(self, seq: List[int]):
        """Update global prefix counts for all prefixes up to `redundancy_ngram`."""
        if not self.cfg.redundancy_aware:
            return
        L = min(len(seq), max(1, self.cfg.redundancy_ngram))
        for t in range(1, L + 1):
            pref = tuple(seq[:t])
            self.global_prefix_counts[pref] += 1.0

    # ========================================================
    # Batched policy update (one step) — unchanged core logic
    # ========================================================
    def _update_policy(
        self,
        all_tuples_with_targets: List,
        env_template: TabularEnv,
        optimizers: List,
        ensemble_reward_override: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Uses log-reward in TB loss, with optional temperature + standardization to stabilize training.
        Keeps FL loss unchanged for non-Bayesian rewards.
        """
        if not all_tuples_with_targets:
            return 0.0, 0.0

        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        # knobs (safe defaults if not in Config)
        temp = float(getattr(c, "tb_reward_temperature", 10.0))
        standardize = bool(getattr(c, "tb_reward_standardize", True))

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
                log_pb = torch.zeros_like(log_pf)
            else:
                log_pb = torch.zeros_like(log_pf)

            logF = self.pf.log_F(padded)

            # ---------- Build log-reward (log_r) ----------
            if ensemble_reward_override is not None:
                # scalar R for entire batch → log, scale, (optionally) standardize
                R = ensemble_reward_override.expand(len(seqs)).to(device)
                log_r = torch.log(R.clamp_min(1e-9))
                log_r = log_r / max(temp, 1e-9)
                if standardize:
                    log_r = (log_r - log_r.mean()) / (log_r.std() + 1e-6)

                l_tb = tb_loss(log_pf, log_pb, self.log_z, log_r, priors_tensor)
                loss = l_tb
                tb_val, fl_val = l_tb, None

            else:
                # per-tree rewards
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

                R = torch.stack(R_list, dim=0)  # (B,)

                # stabilized log-reward
                log_r = torch.log(R.clamp_min(1e-9))
                log_r = log_r / max(temp, 1e-9)
                if standardize:
                    log_r = (log_r - log_r.mean()) / (log_r.std() + 1e-6)

                if self.cfg.reward_function == 'bayesian':
                    l_tb = tb_loss(log_pf, log_pb, self.log_z, log_r, priors_tensor)
                    loss = l_tb
                    tb_val, fl_val = l_tb, None
                else:
                    # keep FL as before; just switch TB to log_r
                    logR_for_shape = torch.log(R.clamp_min(1e-9))
                    if dR_list:
                        gains = torch.nn.utils.rnn.pad_sequence(dR_list, batch_first=True, padding_value=0.0)
                        gains = torch.relu(gains)
                        gsum = gains.sum(1, keepdim=True).clamp_min(1e-9)
                        dR_shaped = gains * (logR_for_shape.unsqueeze(1) / gsum)
                    else:
                        dR_shaped = torch.zeros_like(log_pf)

                    Tm1 = log_pf.size(1)
                    if dR_shaped.size(1) < Tm1:
                        pad = torch.zeros((dR_shaped.size(0), Tm1 - dR_shaped.size(1)), device=device)
                        dR_shaped = torch.cat([dR_shaped, pad], dim=1)
                    elif dR_shaped.size(1) > Tm1:
                        dR_shaped = dR_shaped[:, :Tm1]

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
        # (optional) decay inter-trajectory redundancy stats if you kept that feature
        if hasattr(self, "_decay_global_prefix_counts"):
            self._decay_global_prefix_counts()

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
        Render the best-known tree (or first tree if best not tracked) during training using
        the current env_template tensors (X_full, y_full). Saves to save_dir if provided and
        returns a graphviz.Digraph object.
        """
        import os
        import torch
        import numpy as np
        try:
            import graphviz
        except Exception as e:
            # If graphviz not installed, just skip silently
            return None

        # Pick a sequence to render
        seq = self._best_tree_seq or (self.ensemble[0] if self.ensemble else None)
        if seq is None:
            return None

        actions = list(self.tokenizer.decode(seq[1:-1]))
        Xb = env_template.X_full
        y  = env_template.y_full
        is_cls = (self.cfg.task == "classification")
        n_classes = int(self.cfg.n_classes) if is_cls and self.cfg.n_classes is not None else None

        # Class names
        if is_cls:
            if self.classes_ is not None:
                class_names = [str(c) for c in self.classes_.tolist()]
            else:
                class_names = [f"C{i}" for i in range(n_classes or 0)]
        else:
            class_names = None

        dot = graphviz.Digraph(comment="Best Decision Tree")
        dot.attr("node", shape="box", style="rounded")
        node_counter = 0

        def add_subtree(parent_id, edge_label, idxs, pos):
            nonlocal node_counter
            if pos >= len(actions):
                return pos

            kind, val = actions[pos]
            if kind == "leaf":
                node_id = str(node_counter); node_counter += 1
                n_leaf = int(idxs.numel())
                if is_cls:
                    if n_leaf > 0:
                        counts = torch.bincount(y.index_select(0, idxs), minlength=n_classes).to(torch.int64)
                        total = int(counts.sum().item())
                        if total > 0:
                            maj = int(torch.argmax(counts).item())
                            probs = (counts.float() / total).cpu().numpy()
                            prob_str = ", ".join([f"{class_names[i]}:{probs[i]:.2f}" for i in range(len(probs))])
                            label = f"Leaf • n={n_leaf}\nmajority={class_names[maj]}\n{prob_str}"
                        else:
                            label = f"Leaf • n={n_leaf}\n(no samples)"
                    else:
                        label = f"Leaf • n=0"
                else:
                    if n_leaf > 0:
                        vals = y.index_select(0, idxs).float()
                        mu = float(vals.mean().item())
                        sd = float(vals.std(unbiased=False).item())
                        label = f"Leaf • n={n_leaf}\nmean={mu:.4f} ± {sd:.4f}"
                    else:
                        label = f"Leaf • n=0\nmean=0.0"

                dot.node(node_id, label, style="rounded,filled", fillcolor="lightblue")
                if parent_id is not None:
                    dot.edge(parent_id, node_id, label=edge_label)
                return pos + 1

            if kind == "feat":
                assert pos + 1 < len(actions) and actions[pos + 1][0] == "th", "Malformed trajectory: FEAT not followed by TH."
                f_idx = int(val)
                _, t_val = actions[pos + 1]
                t = int(t_val)

                if idxs.numel() > 0:
                    col = Xb.index_select(0, idxs)[:, f_idx]
                    left = idxs[col <= t]
                    right = idxs[col >  t]
                else:
                    left = idxs
                    right = idxs

                node_id = str(node_counter); node_counter += 1
                fname = self.cfg.feature_cols[f_idx] if 0 <= f_idx < len(self.cfg.feature_cols) else f"X[{f_idx}]"
                label = f"{fname} ≤ bin {t}"
                dot.node(node_id, label)
                if parent_id is not None:
                    dot.edge(parent_id, node_id, label=edge_label)

                pos = add_subtree(node_id, "True",  left,  pos + 2)
                pos = add_subtree(node_id, "False", right, pos)
                return pos

            return pos + 1

        root = torch.arange(Xb.size(0), device=Xb.device)
        add_subtree(None, "", root, 0)

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
            all_tuples_last = all_tuples
            # Auto-visualize every N updates (set viz_every via config or set_params)
            viz_every = int(getattr(self.cfg, "viz_every", 0) or 0)
            if viz_every and (upd % viz_every == 0):
                out_dir = getattr(self.cfg, "viz_dir", "runs/trees")
                fmt = getattr(self.cfg, "viz_format", "png")
                self._visualize_best_tree_live(env_template, save_dir=out_dir, step=upd, format=fmt)


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
    # Rollouts (policy can *STOP* globally via EOS; LEAF still closes a branch)
    # ========================================================
    def _collect_rollouts(self, env_template, temp, residuals, beta):
        forward_tuples, done = [], 0
        ras_counts = {} if self.cfg.redundancy_aware else None  # intra-trajectory prefix counts (per rollout round)

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
                    # optional de-dup across rounds
                    if self.cfg.dedup_sequences:
                        key = tuple(seq)
                        if key in self._seen_sequences:
                            continue
                        self._seen_sequences.add(key)

                    # register inter-trajectory prefix stats
                    self._register_sequence_prefixes(seq)

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
            # optional de-dup on read
            out = []
            seen = set()
            for i in range(k):
                seq = tuple(entries[i][1])
                if self.cfg.dedup_sequences and seq in seen:
                    continue
                seen.add(seq)
                out.append((entries[i][1], entries[i][2]))
            return out

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

        # optional de-dup as we output
        out = []
        seen = set()
        for i in idxs:
            seq = tuple(entries[i][1])
            if self.cfg.dedup_sequences and seq in seen:
                continue
            seen.add(seq)
            out.append((entries[i][1], entries[i][2]))
        return out

    def batched_rollout(self, envs, temp, residuals, beta, ras_counts: Optional[dict] = None):
        """
        Rollouts with *feasible* actions only and redundancy-aware sampling.

        New vs. your previous version:
          • **Global STOP**: EOS is now a valid action at any decision step if `allow_early_stop` holds.
            Choosing EOS finalizes the trajectory immediately (remaining open leaves are treated as implicit leaves).
          • **Intra-trajectory redundancy**: `ras_counts` discourages repeating prefixes **within** the same rollout round.
          • **Inter-trajectory redundancy**: `self.global_prefix_counts` discourages sampling prefixes we have built often **across** rounds.
          • **Immediate token-repeat guard**: disallow picking the exact same token as the last action (simple intra-pattern curb).
        """
        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        num = len(envs)
        END_TOKEN  = v.EOS   # (also used as global STOP during construction)
        LEAF_TOKEN = self.tokenizer._leaf(0)

        for env in envs:
            env.y = residuals
            env.reset(c.batch_size)

        seqs = [[v.BOS] for _ in range(num)]
        decisions = [0 for _ in range(num)]  # number of 'feat' decisions taken so far
        depths: List[Deque[int]] = [deque([0]) for _ in range(num)]
        lo_stacks: List[Deque[torch.Tensor]] = [deque([torch.zeros(v.num_feat, dtype=torch.long, device=device)]) for _ in range(num)]
        hi_stacks: List[Deque[torch.Tensor]] = [deque([torch.full((v.num_feat,), v.num_th - 1, dtype=torch.long, device=device)]) for _ in range(num)]
        # track which training rows belong to each open leaf (indices *relative to env.idxs*)
        row_stacks: List[Deque[torch.Tensor]] = [deque([torch.arange(envs[i].idxs.numel(), device=device)]) for i in range(num)]

        def _mark_done_if_finished(ti: int):
            if not depths[ti]:
                envs[ti].done = True

        active = [i for i in range(num) if not envs[i].done]
        out = [None] * num

        @torch.no_grad()
        def apply_redundancy_penalty(last_logits: torch.Tensor, masks: torch.Tensor, prefixes: List[List[int]]):
            """Subtract λ_intra * local_count + λ_inter * global_count for each allowed next token.
            Works in-place on `last_logits`.
            """
            if not c.redundancy_aware:
                return
            lam_intra = float(c.redundancy_lambda_intra)
            lam_inter = float(c.redundancy_lambda_inter)
            for bi, pref in enumerate(prefixes):
                allowed = torch.nonzero(masks[bi], as_tuple=False).flatten().tolist()
                if not allowed:
                    continue
                base = tuple(pref)
                for tok in allowed:
                    cand = base + (int(tok),)
                    intra = 0.0 if ras_counts is None else float(ras_counts.get(cand, 0.0))
                    inter = float(self.global_prefix_counts.get(cand, 0.0))
                    pen = lam_intra * intra + lam_inter * inter
                    if pen != 0.0:
                        last_logits[bi, tok] -= pen

        with torch.no_grad():
            while active:
                pad = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(seqs[i], device=device) for i in active],
                    batch_first=True, padding_value=v.PAD
                )
                logits, _ = self.pf(pad)
                last = logits[:, -1, :]

                # ---------------- first decision: FEAT / LEAF / STOP(EOS) ----------------
                mask1 = torch.zeros((len(active), v.size()), dtype=torch.bool, device=device)
                prefixes = []
                for bi, oidx in enumerate(active):
                    prefixes.append(seqs[oidx].copy())
                    if not depths[oidx]:
                        continue
                    d = depths[oidx][-1]
                    can_split = (d < c.max_depth)

                    rows_rel = row_stacks[oidx][-1]
                    Xb = envs[oidx].X_full[envs[oidx].idxs]  # [B, F]
                    Xleaf = Xb.index_select(0, rows_rel)     # [n_leaf, F]

                    # always allow LEAF to close current branch
                    mask1[bi, LEAF_TOKEN] = True

                    # allow STOP (EOS) early to end tree globally if enough decisions already
                    if c.allow_early_stop and decisions[oidx] >= c.min_decisions_before_stop:
                        mask1[bi, v.EOS] = True

                    valid_feats = []
                    if can_split and Xleaf.size(0) > 1:
                        n_leaf = Xleaf.size(0)
                        for f in range(Xleaf.size(1)):
                            bf = Xleaf[:, f]
                            uniq, counts = torch.unique(bf, return_counts=True)
                            if uniq.numel() < 2:
                                continue
                            if c.min_child_size and c.min_child_size > 1:
                                csum = counts.cumsum(0)[:-1]
                                left_ok  = csum >= c.min_child_size
                                right_ok = (n_leaf - csum) >= c.min_child_size
                                if not bool((left_ok & right_ok).any()):
                                    continue
                            valid_feats.append(f)

                    if len(valid_feats) > 0:
                        feat_ids = v.split_start + torch.as_tensor(valid_feats, device=device, dtype=torch.long)
                        mask1[bi, feat_ids] = True

                    # immediate token-repeat guard (avoid toggling the same token repeatedly)
                    last_tok = seqs[oidx][-1]
                    if last_tok < mask1.size(1):
                        mask1[bi, last_tok] = False

                # apply redundancy penalties to logits
                apply_redundancy_penalty(last, mask1, prefixes)

                # bias STOP a bit if requested
                if c.stop_bias != 0.0:
                    for bi, oidx in enumerate(active):
                        if mask1[bi, v.EOS]:
                            last[bi, v.EOS] += float(c.stop_bias)

                toks1 = _safe_sample(last, mask1, temp)

                need_threshold: List[Tuple[int,int,int,torch.Tensor,torch.Tensor,torch.Tensor]] = []
                still_for_round: List[int] = []
                for bi, oidx in enumerate(active):
                    tok = toks1[bi].item()
                    if not mask1[bi].any():
                        envs[oidx].done = True
                        continue

                    # Global STOP → finalize
                    if tok == v.EOS:
                        # Append EOS and finish trajectory immediately
                        seqs[oidx].append(tok)
                        # update redundancy maps for this prefix as well
                        if ras_counts is not None:
                            key = tuple(seqs[oidx])
                            ras_counts[key] = ras_counts.get(key, 0) + 1
                        envs[oidx].done = True
                        _mark_done_if_finished(oidx)
                        continue

                    seqs[oidx].append(tok)
                    # update intra-trajectory counts for prefix
                    if ras_counts is not None:
                        key = tuple(seqs[oidx])
                        ras_counts[key] = ras_counts.get(key, 0) + 1

                    if tok == LEAF_TOKEN:
                        envs[oidx].step(('leaf', 0))
                        depths[oidx].pop(); lo_stacks[oidx].pop(); hi_stacks[oidx].pop(); row_stacks[oidx].pop()
                        _mark_done_if_finished(oidx)
                        if not envs[oidx].done:
                            still_for_round.append(oidx)
                        continue

                    kind, f_idx = self.tokenizer.decode_one(tok)  # 'feat'
                    envs[oidx].step((kind, f_idx))
                    decisions[oidx] += 1
                    d0 = depths[oidx].pop()
                    lo_top, hi_top = lo_stacks[oidx].pop(), hi_stacks[oidx].pop()
                    rows_rel = row_stacks[oidx].pop()
                    need_threshold.append((oidx, f_idx, d0, lo_top.clone(), hi_top.clone(), rows_rel.clone()))

                # ---------------- second decision: THRESHOLD ----------------
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

                    prefixes2 = []
                    for si, (oidx, f_idx, d0, lo_top, hi_top, rows_rel) in enumerate(need_threshold):
                        prefixes2.append(seqs[oidx].copy())
                        Xb = envs[oidx].X_full[envs[oidx].idxs]
                        bf = Xb.index_select(0, rows_rel)[:, f_idx]

                        if bf.numel() == 0:
                            continue

                        uniq, counts = torch.unique(bf, sorted=True, return_counts=True)
                        if uniq.numel() < 2:
                            continue

                        # candidate thresholds are exactly the observed bins except the max one
                        cand_t = uniq[:-1]
                        if c.min_child_size and c.min_child_size > 1:
                            csum = counts.cumsum(0)[:-1]
                            keep = (csum >= c.min_child_size) & ((bf.numel() - csum) >= c.min_child_size)
                            cand_t = cand_t[keep]

                        # also respect the current feasible [lo_f, hi_f] window enforced by grammar
                        lo_f = int(lo_top[f_idx].item())
                        hi_f = int(hi_top[f_idx].item())
                        if cand_t.numel() > 0:
                            cand_t = cand_t[(cand_t >= lo_f) & (cand_t <= hi_f)]

                        if cand_t.numel() > 0:
                            th_ids = th_base + cand_t.to(device=device, dtype=torch.long)
                            mask2[si, th_ids] = True

                        # never directly put EOS or LEAF here
                        mask2[si, v.EOS] = False
                        mask2[si, self.tokenizer._leaf(0)] = False

                        # immediate token-repeat guard
                        last_tok = seqs[oidx][-1]
                        if last_tok < mask2.size(1):
                            mask2[si, last_tok] = False

                    # penalties for redundancy on threshold step, too
                    apply_redundancy_penalty(last_th, mask2, prefixes2)

                    toks2 = _safe_sample(last_th, mask2, temp)

                    for si, (oidx, f_idx, d0, lo_top, hi_top, rows_rel) in enumerate(need_threshold):
                        t_tok = toks2[si].item()
                        seqs[oidx].append(t_tok)
                        if ras_counts is not None:
                            key = tuple(seqs[oidx])
                            ras_counts[key] = ras_counts.get(key, 0) + 1

                        _, t_idx = self.tokenizer.decode_one(t_tok)

                        # split rows by chosen t
                        Xb = envs[oidx].X_full[envs[oidx].idxs]
                        fv = Xb.index_select(0, rows_rel)[:, f_idx]
                        m = fv <= t_idx
                        rows_L = rows_rel[m]
                        rows_R = rows_rel[~m]

                        # update feasible bin windows for children
                        lo_L, hi_L = lo_top.clone(), hi_top.clone()
                        hi_L[f_idx] = torch.minimum(hi_L[f_idx], torch.as_tensor(t_idx, device=device))
                        lo_R, hi_R = lo_top.clone(), hi_top.clone()
                        lo_R[f_idx] = torch.maximum(lo_R[f_idx], torch.as_tensor(t_idx + 1, device=device))

                        # push children (R then L for LIFO)
                        depths[oidx].append(d0 + 1); lo_stacks[oidx].append(lo_R); hi_stacks[oidx].append(hi_R); row_stacks[oidx].append(rows_R)
                        depths[oidx].append(d0 + 1); lo_stacks[oidx].append(lo_L); hi_stacks[oidx].append(hi_L); row_stacks[oidx].append(rows_L)

                        envs[oidx].step(('th', int(t_idx)))
                        if depths[oidx]:
                            still_for_round.append(oidx)
                        else:
                            envs[oidx].done = True

                active = still_for_round

        for i in range(num):
            if envs[i].done:
                if seqs[i][-1] != END_TOKEN:
                    seqs[i].append(END_TOKEN)
                out[i] = (seqs[i], envs[i].get_prior(beta).item(), envs[i].idxs.clone())
            else:
                out[i] = None

        return out

    # ========================================================
    # Predict (unchanged except for passing redundancy flag during policy gen)
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

        tqdm.write("--- Generating Boosting Ensemble with Policy (Sequential Inference / v12 style) ---")
        total_trees = policy_inference_trees if policy_inference_trees is not None else c.updates
        num_batches = math.ceil(total_trees / c.num_parallel)

        # residuals for candidate generation (computed once)
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