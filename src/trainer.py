# src/trainer.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Deque, Dict, Set
from collections import deque, defaultdict
import copy
import math

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
    _build_tree_by_data,
    uniform_backward_log_prob,  # uniform PB when requested
)

# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # Data / Task
    feature_cols: List[str]
    target_col: str = "target"
    task: str = "classification"              # "classification" | "regression"
    reward_function: str = "bayesian"         # "bayesian" | "gini" | "variance" | "sse"
    n_classes: Optional[int] = None
    n_bins: int = 255
    binning_strategy: str = "global_uniform"
    device: str = "cuda"

    # Mode
    random_forest: bool = True                # RF if True, Boosting if False

    # Budgets
    updates: int = 50
    rollouts: int = 60                        # target *unique* rollouts per update (if enforced)
    batch_size: int = 8192
    max_depth: int = 7
    top_k_trees: int = 10
    boosting_lr: float = 0.1

    # ---------- Redundancy & STOP (toggles + knobs) ----------
    redundancy_aware: bool = True
    redundancy_lambda_intra: float = 1.0      # (reserved)
    redundancy_lambda_inter: float = 0.25
    redundancy_decay: float = 0.995
    redundancy_ngram: int = 8
    enforce_unique_rollouts: bool = True
    unique_rollouts_max_rounds_factor: int = 50

    dedup_sequences: bool = True
    allow_early_stop: bool = False
    min_decisions_before_stop: int = 1
    stop_bias: float = 0.0

    # Replay sampling (optional extras)
    replay_novelty_bonus: float = 0.10
    replay_metric_mode: str = "off"           # "off" | "acc_corr"
    replay_metric_alpha: float = 1.0
    replay_metric_power: float = 1.0
    replay_metric_refresh: int = 1000

    # Policy network
    lstm_hidden: int = 256
    mlp_layers: int = 3
    mlp_width: int = 256
    lr: float = 1e-4

    # Backward policy choice
    backward_policy: str = "uniform"          # "uniform" | "network" | "none"

    # TB/FL update path
    policy_update_mode: str = "single"       # "batched" | "single"

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
    metric_sample_size: int = 20000
    eval_batch_size: int = 16384

    # Predictor/build guards
    rollout_temperature: float = 0.0
    min_child_size: int = 20
    min_gain: float = 0.0

    # Reward scope
    training_reward_scope: str = "per_tree"   # "per_tree" | "ensemble"
    ensemble_reward_metric: str = "mse"       # (regression-only)

    # Inference-time weighting reward (RF & Boost)
    infer_reward_function: Optional[str] = None  # None->use training; "none"->equal weights

    # Policy predictor mode
    policy_predictor_mode: str = "dirichlet_sample"  # "dirichlet_sample" | "dirichlet" | "mean"

    # Best-tree viz tracker
    show_best_tree_acc: bool = True
    viz_every: int = 0
    viz_dir: str = "runs/trees"
    viz_format: str = "png"

    # TB loss stabilization
    tb_reward_temperature: float = 10.0
    tb_reward_standardize: bool = True

    # --- Leaf token penalty (discourage early LEAF) ---
    leaf_penalty_strength: float = 4.0
    leaf_penalty_decay: float = 0.85
    leaf_penalty_min_depth: int = 2
    leaf_bias: float = 0.0
    leaf_cooldown_steps: int = 1

    # Threshold balance penalty (optional)
    threshold_balance_gamma: float = 0.0


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

        self.pf = None         # forward policy
        self.pb = None         # backward policy (optional)
        self.log_z: Optional[torch.Tensor] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.le: Optional[LabelEncoder] = None
        self.classes_: Optional[np.ndarray] = None
        self.scaler = GradScaler(enabled=cfg.amp)

        # Best-single-tree tracker
        self._best_tree_seq: Optional[List[int]] = None
        self._best_tree_acc: float = 0.0

        # Global redundancy tracking (inter-trajectory)
        self.global_prefix_counts: Dict[Tuple[int, ...], float] = defaultdict(float)

        # Lightweight novelty memory (full sequences)
        self._seen_sequences: Set[Tuple[int, ...]] = set()

        # Cached tensors for train-metric evaluation
        self._metric_X = None
        self._metric_y = None
        self._metric_target_full = None
        self._train_metric_cache: Dict[Tuple[int, ...], Tuple[float, int]] = {}

        self._pb_replay_scores: Dict[Tuple[int, ...], float] = {}  # cache: seq tuple -> PB score


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

        # cache tensors for metric computation
        self._metric_X = X_binned.cpu() if c.eval_on_cpu else X_binned
        self._metric_y = y_true.cpu() if c.eval_on_cpu else y_true
        if c.task == "classification":
            mt = torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float)
        else:
            mt = y_true
        self._metric_target_full = mt.cpu() if c.eval_on_cpu else mt

        # forward policy
        self.pf = torch.jit.script(
            PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device)
        )

        # backward policy (optional)
        if c.backward_policy == "network":
            self.pb = torch.jit.script(
                PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device)
            )
        else:
            self.pb = None

        # logZ
        self.log_z = torch.nn.Parameter(torch.tensor(0.0, device=c.device))

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

        # global logZ optimizer (unchanged)
        optim_z = torch.optim.Adam([self.log_z], lr=c.lr / 10)
        sched_z = SequentialLR(
            optim_z,
            [LambdaLR(optim_z, lambda u: min(1.0, u / max(1, 10))),
            CosineAnnealingLR(optim_z, T_max=max(1, c.updates - 10))],
            milestones=[10],
        )
        opt_list.append(optim_z)
        sch_list.append(sched_z)

        # --- NEW: PB optimizer if we train a network backward policy ---
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


        # replay
        self.replay_buffer = ReplayBuffer(capacity=100000)

        if c.beta is None:
            c.beta = math.log(4) + math.log(len(c.feature_cols))
            tqdm.write(f"[trainer] β (structure prior) = {c.beta:.4f}")

        if c.random_forest:
            self._fit_dt_gfn_random_forest(env_template, y_true, X_binned, opt_list, sch_list)
        else:
            self._fit_boost_gfn(env_template, y_true, X_binned, opt_list, sch_list)

        return self

    # ========================================================
    # Visualize (unchanged)
    # ========================================================
    def _visualize_best_tree_live(
        self,
        env_template,
        *,
        save_dir: str = "runs/trees",
        step: int | None = None,
        format: str = "png",
    ):
        import os
        try:
            import graphviz
        except Exception:
            return None

        seq = self._best_tree_seq or (self.ensemble[0] if self.ensemble else None)
        if not seq:
            return None

        Xb = env_template.X_full
        y  = env_template.y_full
        device = Xb.device

        tree = None
        try:
            tok_tensor = torch.tensor(seq, device=device, dtype=torch.long)
            idxs_map  = torch.arange(Xb.size(0), device=device, dtype=torch.long)
            root, _ = _build_tree_by_data(tok_tensor, self.tokenizer, Xb, idxs_map)

            def is_split(n): return n.get("type") == "split"
            def feat(n):     return int(n["f"])
            def thr(n):      return int(n["t"])
            def left(n):     return n["L"]
            def right(n):    return n["R"]
            def leaf_indices(n): return n.get("idxs", None)

            tree = (root, is_split, feat, thr, left, right, leaf_indices)
        except Exception:
            try:
                from src.utils import decode_tree_from_seq
                def _wrap_simple(root_dict):
                    def is_split(n): return n.get("kind") == "split"
                    def feat(n):     return int(n["feat"])
                    def thr(n):      return int(n["thr"])
                    def left(n):     return n["left"]
                    def right(n):    return n["right"]
                    def leaf_indices(n): return None
                    return (root_dict, is_split, feat, thr, left, right, leaf_indices)
                tree = _wrap_simple(decode_tree_from_seq(seq, self.tokenizer))
            except Exception:
                return None

        root, is_split, feat, thr, left, right, leaf_indices = tree

        is_cls = (self.cfg.task == "classification")
        n_classes = int(self.cfg.n_classes) if is_cls and self.cfg.n_classes is not None else None
        if is_cls:
            if getattr(self, "classes_", None) is not None:
                class_names = [str(c) for c in self.classes_.tolist()]
            else:
                class_names = [f"C{i}" for i in range(n_classes or 0)]

        def leaf_label(indices_tensor) -> str:
            import torch
            if indices_tensor is None:
                return "Leaf"
            idxs_local = indices_tensor
            n = int(idxs_local.numel())
            if is_cls:
                if n == 0:
                    return "Leaf • n=0"
                counts = torch.bincount(y.index_select(0, idxs_local), minlength=n_classes)
                total = int(counts.sum().item())
                maj = int(torch.argmax(counts).item()) if total > 0 else 0
                probs = (counts.float() / max(1, total)).cpu().numpy()
                prob_str = ", ".join([f"{class_names[i]}:{probs[i]:.2f}" for i in range(len(probs))])
                return f"Leaf • n={n}\nmajority={class_names[maj]}\n{prob_str}"
            else:
                if n == 0:
                    return "Leaf • n=0\nmean=0.0"
                vals = y.index_select(0, idxs_local).float()
                mu = float(vals.mean().item())
                sd = float(vals.std(unbiased=False).item())
                return f"Leaf • n={n}\nmean={mu:.4f} ± {sd:.4f}"

        dot = graphviz.Digraph(comment="Best Decision Tree")
        dot.attr("node", shape="box", style="rounded")
        nid = 0
        feat_names = self.cfg.feature_cols

        def walk(node) -> str:
            nonlocal nid
            if not is_split(node):
                lid = str(nid); nid += 1
                dot.node(lid, leaf_label(leaf_indices(node)), style="rounded,filled", fillcolor="lightblue")
                return lid

            f = feat(node); t = thr(node)
            fname = feat_names[f] if 0 <= f < len(feat_names) else f"X[{f}]"
            myid = str(nid); nid += 1
            dot.node(myid, f"{fname} ≤ bin {t}")

            lid = walk(left(node))
            rid = walk(right(node))
            dot.edge(myid, lid, label="True")
            dot.edge(myid, rid, label="False")
            return myid

        walk(root)
        os.makedirs(save_dir, exist_ok=True)
        fname = f"best_tree_step_{step}" if step is not None else "best_tree"
        dot.render(os.path.join(save_dir, fname), format=format, cleanup=True)
        return dot

    # ========================================================
    # Reward helpers
    # ========================================================
    def _per_tree_reward(self, tok: torch.Tensor, reward_env: TabularEnv) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        c = self.cfg
        is_residual_matrix = (
            hasattr(reward_env, "y") and isinstance(reward_env.y, torch.Tensor) and reward_env.y.dim() == 2
            and not torch.allclose(reward_env.y.sum(1), torch.ones_like(reward_env.y.sum(1)), atol=1e-3, rtol=0.0)
        )
        if c.reward_function == "bayesian" and not is_residual_matrix:
            fn = calculate_bayesian_reward if c.task == "classification" else calculate_bayesian_reward_regression
            R_t = fn(tok, self.tokenizer, reward_env, c.beta)
            return R_t, None

        if is_residual_matrix or c.reward_function in ("variance", "sse"):
            dR = deltaE_split_gain_sse(tok, self.tokenizer, reward_env)
            R_t = torch.clamp(dR.sum(), min=1e-9)
            return R_t, dR

        if c.task == "classification":
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
    # Global redundancy helpers
    # ========================================================
    def _decay_global_prefix_counts(self):
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
        if not self.cfg.redundancy_aware:
            return
        L = min(len(seq), max(1, self.cfg.redundancy_ngram))
        for t in range(1, L + 1):
            pref = tuple(seq[:t])
            self.global_prefix_counts[pref] += 1.0

    # ========================================================
    # Train-metric helper (unchanged)
    # ========================================================
    @torch.no_grad()
    def _train_metric_for_seq(self, seq: List[int]) -> float:
        if self._metric_X is None or self._metric_y is None or self._metric_target_full is None:
            return 0.5
        X = self._metric_X
        y = self._metric_y
        y_target = self._metric_target_full
        c = self.cfg

        N = X.size(0)
        if c.metric_sample_size and c.metric_sample_size > 0 and c.metric_sample_size < N:
            idx = torch.randperm(N, device=X.device)[:c.metric_sample_size]
            Xs = X.index_select(0, idx)
            ys = y.index_select(0, idx)
            yts = y_target.index_select(0, idx) if y_target.dim() == 2 else y_target.index_select(0, idx)
        else:
            Xs, ys, yts = X, y, y_target

        pred_fn = get_tree_predictor(
            seq, X, y_target, self.tokenizer,
            min_child_size=c.min_child_size, min_gain=c.min_gain,
            predictor_mode=c.policy_predictor_mode
        )
        preds = self._predict_in_batches(
            pred_fn, Xs, c.eval_batch_size, device=("cpu" if c.eval_on_cpu else c.device)
        )

        if c.task == "classification":
            acc = (preds.argmax(1).cpu() == ys.cpu()).float().mean().item()
            return float(max(0.0, min(1.0, acc)))
        else:
            pv = preds.squeeze().cpu()
            yv = ys.squeeze().cpu().float()
            if pv.numel() < 2 or pv.std(unbiased=False) == 0 or yv.std(unbiased=False) == 0:
                return 0.5
            corr = torch.corrcoef(torch.stack([pv, yv]))[0, 1].item()
            corr = max(-1.0, min(1.0, corr))
            return 0.5 * (corr + 1.0)

    # ========================================================
    # Backward network helper
    # ========================================================
    @torch.no_grad()
    def _network_backward_log_prob(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Compute per-step backward log-probabilities with a learnable PB network.

        Given padded batch `seq` (B×T), for each row b with length L_b, we want:
          log p_b( remove last token at step t | state s_t ) for t=1..L_b-1.

        Implementation:
          1) Reverse each sequence up to its actual length (keep PAD at the end).
          2) Run PB.forward on the reversed batch and use `log_prob` (teacher forcing).
          3) Flip the (B×(T-1)) result back along time so it aligns with forward steps.

        Returns: (B×(T-1)) float tensor; zeros where next token is PAD.
        """
        assert self.pb is not None, "PB network not initialized"
        v = self.tokenizer.v
        pad_id = v.PAD
        device = seq.device
        B, T = seq.shape

        lengths = (seq != pad_id).sum(dim=1)  # includes BOS..EOS (no PAD)
        rev_rows = []
        for b in range(B):
            L = int(lengths[b].item())
            sb = seq[b, :L]
            rb = torch.flip(sb, dims=[0])  # reverse up to length
            if L < T:
                pad_tail = torch.full((T - L,), pad_id, dtype=seq.dtype, device=device)
                rb = torch.cat([rb, pad_tail], dim=0)
            rev_rows.append(rb)
        rev_seq = torch.stack(rev_rows, dim=0)  # (B, T)

        # teacher-forced next-token log-prob on reversed sequence
        rev_logp = self.pb.log_prob(rev_seq)  # (B, T-1); zeros where next PAD

        # map back: for each b, reverse the valid part (L_b-1) along time
        out = torch.zeros_like(rev_logp)
        for b in range(B):
            L = int(lengths[b].item())
            if L >= 2:
                out[b, :L-1] = torch.flip(rev_logp[b, :L-1], dims=[0])
        return out

    # ========================================================
    # Policy update (batched & single)
    # ========================================================
    def _update_policy(
        self,
        all_tuples_with_targets: List,
        env_template: TabularEnv,
        optimizers: List,
        ensemble_reward_override: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Batched policy update. Uses teacher-forced log-probs, and one of:
          • uniform backward (analytic),
          • network backward (learned),
          • none (zeros).
        """
        if not all_tuples_with_targets:
            return 0.0, 0.0

        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        temp = float(getattr(c, "tb_reward_temperature", 10.0))
        standardize = bool(getattr(c, "tb_reward_standardize", True))

        seqs, priors, targets = zip(*all_tuples_with_targets)
        toks = [torch.tensor(s, device=device, dtype=torch.long) for s in seqs]
        padded = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=v.PAD)
        priors_tensor = torch.as_tensor(priors, device=device, dtype=torch.float32)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        with autocast(enabled=c.amp):
            log_pf = self.pf.log_prob(padded)  # (B, T-1)

            # backward policy
            if c.backward_policy == "network" and self.pb is not None:
                log_pb = self._network_backward_log_prob_train(padded)  # requires grad -> trains PB
            elif c.backward_policy == "uniform":
                log_pb = uniform_backward_log_prob(padded, self.tokenizer, c.max_depth)
            else:  # "none"
                log_pb = torch.zeros_like(log_pf)

            logF = self.pf.log_F(padded)

            if ensemble_reward_override is not None:
                R = ensemble_reward_override.expand(len(seqs)).to(device)
                log_r = torch.log(R.clamp_min(1e-9)) / max(temp, 1e-9)
                if standardize:
                    log_r = (log_r - log_r.mean()) / (log_r.std() + 1e-6)
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
                    reward_env.reset(len(target))
                    R_t, dR = self._per_tree_reward(tok_i, reward_env)
                    R_list.append(R_t.squeeze())
                    if dR is not None:
                        dR_list.append(dR.squeeze(0))

                R = torch.stack(R_list, dim=0)
                log_r = torch.log(R.clamp_min(1e-9)) / max(temp, 1e-9)
                if standardize:
                    log_r = (log_r - log_r.mean()) / (log_r.std() + 1e-6)

                if self.cfg.reward_function == "bayesian":
                    l_tb = tb_loss(log_pf, log_pb, self.log_z, log_r, priors_tensor)
                    loss = l_tb
                    tb_val, fl_val = l_tb, None
                else:
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
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 1.0)
            self.scaler.step(opt)
        self.scaler.update()

        self.replay_buffer.mark_policy_update()
        self._decay_global_prefix_counts()

        tb_loss_acc = float(tb_val.item())
        fl_loss_acc = float(fl_val.item()) if fl_val is not None else 0.0
        return tb_loss_acc, fl_loss_acc

    def _update_policy_single(
        self,
        all_tuples_with_targets: List,
        env_template: TabularEnv,
        optimizers: List,
        ensemble_reward_override: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """
        Non-batched policy update: compute TB/FL per-sample and average.
        Useful to rule out batching/masking artifacts.
        """
        if not all_tuples_with_targets:
            return 0.0, 0.0

        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        temp = float(getattr(c, "tb_reward_temperature", 10.0))
        standardize = bool(getattr(c, "tb_reward_standardize", True))

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        tb_vals, fl_vals = [], []
        losses = []
        with autocast(enabled=c.amp):
            if ensemble_reward_override is not None:
                R_scalar = ensemble_reward_override.to(device)

            for (seq, prior, target) in all_tuples_with_targets:
                t = torch.tensor(seq, device=device, dtype=torch.long).unsqueeze(0)
                prior_t = torch.as_tensor([prior], device=device, dtype=torch.float32)

                log_pf_i = self.pf.log_prob(t)
              
                if c.backward_policy == "network" and self.pb is not None:
                    log_pb_i = self._network_backward_log_prob_train(t)  # requires grad -> trains PB
                elif c.backward_policy == "uniform":
                    log_pb_i = uniform_backward_log_prob(t, self.tokenizer, c.max_depth)
                else:  # "none"
                    log_pb_i = torch.zeros_like(log_pf_i)


                logF_i = self.pf.log_F(t)

                if ensemble_reward_override is not None:
                    log_r = torch.log(R_scalar.clamp_min(1e-9)) / max(temp, 1e-9)
                    if standardize:
                        log_r = log_r - log_r.detach()
                    ltb = tb_loss(log_pf_i, log_pb_i, self.log_z, log_r, prior_t)
                    losses.append(ltb)
                    tb_vals.append(ltb.detach())
                else:
                    reward_env = copy.copy(env_template)
                    reward_env.y = target
                    reward_env.reset(len(target))
                    R_t, dR = self._per_tree_reward(t, reward_env)
                    log_r = torch.log(R_t.clamp_min(1e-9)) / max(temp, 1e-9)
                    if standardize:
                        log_r = log_r - log_r.detach()
                    ltb = tb_loss(log_pf_i, log_pb_i, self.log_z, log_r, prior_t)

                    if self.cfg.reward_function == "bayesian" or dR is None:
                        losses.append(ltb)
                        tb_vals.append(ltb.detach())
                    else:
                        gains = torch.relu(dR.squeeze(0))
                        if gains.numel() == 0:
                            lfl = torch.zeros_like(ltb)
                        else:
                            Tm1 = log_pf_i.size(1)
                            if gains.numel() < Tm1:
                                pad = torch.zeros((Tm1 - gains.numel(),), device=device)
                                shaped = torch.cat([gains, pad], dim=0).unsqueeze(0)
                            else:
                                shaped = gains[:Tm1].unsqueeze(0)
                            lfl = fl_loss(logF_i, log_pf_i, log_pb_i, shaped)
                        losses.append(ltb + lfl)
                        tb_vals.append(ltb.detach())
                        fl_vals.append(lfl.detach())

            loss = torch.stack(losses).mean()

        self.scaler.scale(loss).backward()
        for opt in optimizers:
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 1.0)
            self.scaler.step(opt)
        self.scaler.update()

        self.replay_buffer.mark_policy_update()
        self._decay_global_prefix_counts()

        tb_loss_acc = float(torch.stack(tb_vals).mean().item() if tb_vals else 0.0)
        fl_loss_acc = float(torch.stack(fl_vals).mean().item() if fl_vals else 0.0)
        return tb_loss_acc, fl_loss_acc

    # ========================================================
    # RF training (policy + gen)
    # ========================================================
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

        all_tuples_last: List[Tuple[List[int], float]] = []

        for upd in tqdm(range(1, c.updates + 1), desc="Policy Training & Tree Generation"):
            forward_tuples = self._collect_rollouts(env_template, temp=c.rollout_temperature, residuals=y_true, beta=c.beta)
            replay_tuples = self.sample_replay(c.top_k_trees)
            all_tuples = forward_tuples + replay_tuples
            if not all_tuples:
                for sch in schedulers:
                    sch.step()
                continue

            ensemble_R_override = None
            if c.training_reward_scope == "ensemble" and c.task == "regression" and len(all_tuples) > 0:
                seqs = [seq for seq, _ in all_tuples]
                R_scalar = self._ensemble_reward_mse(seqs, X_binned, y_true)
                ensemble_R_override = R_scalar

            all_tuples_with_targets = [(seq, prior, y_target_for_reward) for seq, prior in all_tuples]

            if c.policy_update_mode == "single":
                avg_tb_loss, avg_fl_loss = self._update_policy_single(
                    all_tuples_with_targets, env_template, optimizers, ensemble_reward_override=ensemble_R_override
                )
            else:
                avg_tb_loss, avg_fl_loss = self._update_policy(
                    all_tuples_with_targets, env_template, optimizers, ensemble_reward_override=ensemble_R_override
                )
            for sch in schedulers:
                sch.step()

            # FULL DATA quick metrics (reward-weighted)
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
            viz_every = int(getattr(self.cfg, "viz_every", 0) or 0)
            if viz_every and (upd % viz_every == 0):
                out_dir = getattr(self.cfg, "viz_dir", "runs/trees")
                fmt = getattr(self.cfg, "viz_format", "png")
                self._visualize_best_tree_live(env_template, save_dir=out_dir, step=upd, format=fmt)

            all_tuples_last = all_tuples

        self.ensemble = [seq for seq, _ in all_tuples_last if seq] if all_tuples_last else []
        tqdm.write(f"--- RF finished. Final forest size: {len(self.ensemble)} ---")

    # ========================================================
    # Boosting (unchanged structurally)
    # ========================================================
    def _fit_boost_gfn(self, env_template, y_true, X_binned, optimizers, schedulers):
        c = self.cfg
        tqdm.write("--- Starting Boost-GFN Training ---")

        if c.task == "classification":
            class_counts = torch.bincount(y_true, minlength=c.n_classes).float()
            class_probs = class_counts / class_counts.sum()
            base_pred = torch.log(class_probs + 1e-9).unsqueeze(0).repeat(len(y_true), 1)
        else:
            self.y_mean = y_true.mean().item()
            base_pred = torch.full_like(y_true, self.y_mean, dtype=torch.float32)

        for upd in tqdm(range(1, c.updates + 1), desc="Boost Updates"):
            if c.task == "classification":
                class_counts = torch.bincount(y_true, minlength=c.n_classes).float()
                class_probs = class_counts / class_counts.sum()
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
                for sch in schedulers:
                    sch.step()
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

            if c.policy_update_mode == "single":
                avg_tb_loss, avg_fl_loss = self._update_policy_single(
                    tuples, env_template, optimizers, ensemble_reward_override=ensemble_R_override
                )
            else:
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
                    tqdm.write(f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_tb_loss:.4f} | Corr: nan")
            self.ensemble = [seq for seq, _ in candidates if seq]
            viz_every = int(getattr(self.cfg, "viz_every", 0) or 0)
            if viz_every and (upd % viz_every == 0):
                out_dir = getattr(self.cfg, "viz_dir", "runs/trees")
                fmt = getattr(self.cfg, "viz_format", "png")
                self._visualize_best_tree_live(env_template, save_dir=out_dir, step=upd, format=fmt)

    # ========================================================
    # Rollouts (LEAF → expand? → feature → threshold)
    # ========================================================
    def _collect_rollouts(self, env_template, temp, residuals, beta):
        c = self.cfg
        forward_tuples: List[Tuple[List[int], float]] = []

        target = int(max(1, c.rollouts))
        attempts = 0
        max_rounds = max(1, c.unique_rollouts_max_rounds_factor) * target

        pbar = tqdm(total=target, desc="Rollouts (unique)" if c.enforce_unique_rollouts else "Rollouts", leave=False)

        seen_this_round: Set[Tuple[int, ...]] = set()

        while (len(seen_this_round) if c.enforce_unique_rollouts else len(forward_tuples)) < target and (attempts < max_rounds):
            attempts += 1
            remaining = (target - (len(seen_this_round) if c.enforce_unique_rollouts else len(forward_tuples)))
            batch = min(c.num_parallel, remaining)

            ras_counts: Optional[dict] = {} if c.redundancy_aware else None

            envs = [copy.copy(env_template) for _ in range(batch)]
            idx_batches = [env_template.draw_indices(c.batch_size) for _ in range(batch)]
            for env, idxs in zip(envs, idx_batches):
                env.y = residuals
                env.paths = []
                env.open_leaves = 1
                env.done = False
                env.idxs = idxs

            results = self.batched_rollout(envs, temp, residuals, beta, ras_counts)

            new_count = 0
            for res in results:
                if not res:
                    continue
                seq, prior, idxs = res
                key = tuple(seq)
                if c.enforce_unique_rollouts:
                    if key in seen_this_round:
                        continue
                    seen_this_round.add(key)
                forward_tuples.append((seq, prior))
                new_count += 1

                if c.dedup_sequences:
                    self._seen_sequences.add(key)
                self._register_sequence_prefixes(seq)

                reward_env = copy.copy(env_template)
                reward_env.idxs = idxs.to(c.device)
                reward_env.y_full = env_template.y_full
                reward_env.X_full = env_template.X_full
                reward_env.y = residuals
                r = self._weight_for_tree(seq, reward_env, mode=c.reward_function)
                self.replay_buffer.add(r, seq, prior, idxs.cpu())

            if new_count > 0:
                pbar.update(new_count)

        pbar.close()

        if c.enforce_unique_rollouts:
            uniq = []
            seen = set()
            for s, p in forward_tuples:
                t = tuple(s)
                if t in seen:
                    continue
                seen.add(t)
                uniq.append((s, p))
            forward_tuples = uniq[:target]
        else:
            forward_tuples = forward_tuples[:target]

        return forward_tuples

    def sample_replay(self, k: int) -> List[Tuple[List[int], float]]:
        """
        Top-k from replay by backward policy *score*.
        - If backward_policy=="network" and PB exists: rank by sum_t log p_B(s_{t-1}|s_t).
        - Else: fallback to reward top-k (first element in replay entries).
        Never trains PB here (eval+no_grad only). Caches PB scores and only
        evaluates new sequences.
        Returns: List[(seq, prior)]
        """
        buf = self.replay_buffer
        if not buf or not getattr(buf, "data", None):
            return []

        entries = list(buf.data)

        # --- Extract (key, seq, prior) robustly from entries ---
        triples: List[Tuple[Tuple[int, ...], List[int], float]] = []
        for e in entries:
            # Common layout from `add(r, seq, prior, idxs)`; tolerate extras.
            if isinstance(e, (list, tuple)) and len(e) >= 3:
                seq = e[1]
                prior = float(e[2])
            elif isinstance(e, dict):
                seq = e.get("seq", None)
                prior = float(e.get("prior", 0.0))
            else:
                continue
            if not isinstance(seq, (list, tuple)) or len(seq) == 0:
                continue
            key = tuple(int(x) for x in seq)
            triples.append((key, list(seq), prior))

        if not triples:
            return []

        v = self.tokenizer.v
        device = self.cfg.device

        # --- If PB network is active, score by PB in eval/no_grad and cache ---
        if self.cfg.backward_policy == "network" and (self.pb is not None):
            # figure out which sequences are new to the cache
            to_eval = [(key, seq) for (key, seq, _prior) in triples if key not in self._pb_replay_scores]
            if to_eval:
                seq_tensors = [torch.tensor(seq, device=device, dtype=torch.long) for _, seq in to_eval]
                padded = torch.nn.utils.rnn.pad_sequence(seq_tensors, batch_first=True, padding_value=v.PAD)

                # eval-only, no grads
                was_training = self.pb.training
                self.pb.eval()
                with torch.no_grad():
                    # reuse the same routine used during training (OK for eval)
                    log_pb = self._network_backward_log_prob_train(padded)  # (B, T-1)
                    scores = log_pb.sum(dim=1)  # scalar per seq
                if was_training:
                    self.pb.train()

                for i, (key, _seq) in enumerate(to_eval):
                    self._pb_replay_scores[key] = float(scores[i].item())

            # gather cached scores
            scored = [ (self._pb_replay_scores.get(key, float("-inf")), seq, prior)
                      for (key, seq, prior) in triples ]
            scored.sort(key=lambda x: x[0], reverse=True)

            out: List[Tuple[List[int], float]] = []
            for _score, seq, prior in scored[: max(0, k)]:
                out.append((seq, prior))
            return out

        # --- Fallback: reward top-k (assumes buf stores reward at index 0) ---
        try:
            entries_sorted = sorted(entries, key=lambda e: float(e[0]), reverse=True)
        except Exception:
            entries_sorted = entries  # if not sortable, just take in-order

        out: List[Tuple[List[int], float]] = []
        for e in entries_sorted:
            if isinstance(e, (list, tuple)) and len(e) >= 3:
                seq, prior = e[1], float(e[2])
            elif isinstance(e, dict):
                seq, prior = e.get("seq", None), float(e.get("prior", 0.0))
            else:
                continue
            if not isinstance(seq, (list, tuple)) or len(seq) == 0:
                continue
            out.append((list(seq), prior))
            if len(out) >= k:
                break
        return out

    def _network_backward_log_prob_train(self, padded: torch.Tensor) -> torch.Tensor:
        """
        Compute log p_B(s_{t-1}|s_t) for *forward* steps using a backward network by
        teacher-forcing on the reversed non-PAD sequence and flipping back.
        Returns: (B, T-1) with zeros where next token is PAD.
        """
        assert self.pb is not None, "PB network not initialized"
        v = self.tokenizer.v
        device = padded.device
        B, T = padded.shape
        out = torch.zeros((B, T - 1), device=device, dtype=torch.float32)

        for b in range(B):
            row = padded[b]
            L = int((row != v.PAD).sum().item())   # length incl. BOS..EOS
            if L < 2:
                continue

            s = row[:L]               # [L]
            # reversed teacher forcing: predict rev[1:] from rev[:-1]
            rev_in  = s[:-1].flip(0).unsqueeze(0)   # (1, L-1)
            rev_tgt = s[1: ].flip(0).unsqueeze(0)   # (1, L-1)

            logits, _ = self.pb(rev_in)             # (1, L-1, V)
            logp = torch.log_softmax(logits, dim=-1)
            gathered = logp.gather(-1, rev_tgt.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

            # map back to forward order
            gathered_fwd = gathered.flip(1).squeeze(0)  # (L-1,)
            out[b, :L-1] = gathered_fwd

        mask = (padded[:, 1:] != v.PAD).to(out.dtype)
        return out * mask





    def batched_rollout(self, envs, temp, residuals, beta, ras_counts: Optional[dict] = None):
        """
        Roll out trees with feasible actions only, making decisions as:
            leaf -> choose whether to expand -> (if expand) sample feature -> sample threshold.
        """
        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        num = len(envs)
        END_TOKEN  = v.EOS
        LEAF_TOKEN = self.tokenizer._leaf(0)

        leaf_cd_steps   = int(getattr(c, "leaf_cooldown_steps", 0) or 0)
        th_imbal_gamma  = float(getattr(c, "threshold_balance_gamma", 0.0) or 0.0)
        leaf_bias_const = float(getattr(c, "leaf_bias", 0.0) or 0.0)

        def _leaf_logit_penalty(depth: int) -> float:
            strength = float(getattr(c, "leaf_penalty_strength", 0.0) or 0.0)
            if strength <= 0.0:
                return 0.0
            decay   = float(getattr(c, "leaf_penalty_decay", 0.85))
            min_d   = int(getattr(c, "leaf_penalty_min_depth", 2))
            pen = strength * (decay ** max(0, depth))
            if depth < min_d:
                pen *= 3.0
            return float(pen)

        for env in envs:
            env.y = residuals
            env.reset(c.batch_size)

        seqs = [[v.BOS] for _ in range(num)]
        depths: List[Deque[int]] = [deque([0]) for _ in range(num)]
        lo_stacks: List[Deque[torch.Tensor]] = [deque([torch.zeros(v.num_feat, dtype=torch.long, device=device)]) for _ in range(num)]
        hi_stacks: List[Deque[torch.Tensor]] = [deque([torch.full((v.num_feat,), v.num_th - 1, dtype=torch.long, device=device)]) for _ in range(num)]
        row_stacks: List[Deque[torch.Tensor]] = [deque([torch.arange(envs[i].idxs.numel(), device=device)]) for i in range(num)]
        cd_stacks:  List[Deque[int]]         = [deque([0]) for _ in range(num)]  # LEAF cooldown

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
                last = logits[:, -1, :]  # (B_active, V)

                if ras_counts is not None:
                    for bi, oidx in enumerate(active):
                        pref = tuple(seqs[oidx])
                        if pref in ras_counts:
                            last[bi, :] -= 1e9

                expand_decisions: List[Tuple[bool, Optional[int]]] = []
                valid_feat_lists: List[List[int]] = []
                still_for_round: List[int] = []

                for bi, oidx in enumerate(active):
                    if not depths[oidx]:
                        expand_decisions.append((False, None))
                        continue

                    d = depths[oidx][-1]
                    can_split = (d < c.max_depth)

                    rows_rel = row_stacks[oidx][-1]
                    Xb = envs[oidx].X_full[envs[oidx].idxs]
                    Xleaf = Xb.index_select(0, rows_rel)
                    n_leaf = int(Xleaf.size(0))

                    valid_feats = []
                    if can_split and n_leaf > 1:
                        mcs = int(c.min_child_size or 0)
                        for f in range(Xleaf.size(1)):
                            bf = Xleaf[:, f]
                            uniq, counts = torch.unique(bf, return_counts=True)
                            if uniq.numel() < 2:
                                continue
                            if mcs > 1:
                                csum = counts.cumsum(0)[:-1]
                                if not bool(((csum >= mcs) & ((n_leaf - csum) >= mcs)).any()):
                                    continue
                            valid_feats.append(f)

                    valid_feat_lists.append(valid_feats)

                    leaf_allowed = True
                    if leaf_cd_steps > 0 and can_split and cd_stacks[oidx][-1] > 0:
                        leaf_allowed = False

                    if not valid_feats and leaf_allowed:
                        choose_expand = False
                        expand_decisions.append((choose_expand, None))
                        if cd_stacks[oidx]:
                            cd_stacks[oidx][-1] = max(0, cd_stacks[oidx][-1] - 1)
                        continue

                    if not valid_feats and not leaf_allowed:
                        leaf_allowed = True

                    if valid_feats:
                        feat_ids = v.split_start + torch.as_tensor(valid_feats, device=device, dtype=torch.long)
                        feat_logits = last[bi, feat_ids]
                        if temp > 1e-9:
                            feat_logits = feat_logits / temp
                        logit_close = last[bi, LEAF_TOKEN]
                        if temp > 1e-9:
                            logit_close = logit_close / temp
                        if leaf_allowed:
                            if leaf_bias_const != 0.0:
                                logit_close = logit_close + leaf_bias_const
                            pen = _leaf_logit_penalty(d)
                            if pen != 0.0:
                                logit_close = logit_close - pen
                        logit_expand = torch.logsumexp(feat_logits, dim=0)
                        two = torch.stack([logit_close, logit_expand], dim=0)
                        probs_two = torch.softmax(two, dim=0)
                        choose_expand = (torch.multinomial(probs_two.unsqueeze(0), 1).item() == 1)
                    else:
                        choose_expand = False

                    if choose_expand:
                        feat_ids = v.split_start + torch.as_tensor(valid_feats, device=device, dtype=torch.long)
                        feat_logits = last[bi, feat_ids]
                        if temp > 1e-9:
                            feat_logits = feat_logits / temp
                        probs_feat = torch.softmax(feat_logits, dim=0)
                        choice_idx = int(torch.multinomial(probs_feat.unsqueeze(0), 1).item())
                        chosen_feat = valid_feats[choice_idx]
                        expand_decisions.append((True, chosen_feat))
                    else:
                        expand_decisions.append((False, None))

                    if cd_stacks[oidx]:
                        cd_stacks[oidx][-1] = max(0, cd_stacks[oidx][-1] - 1)

                need_threshold: List[Tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
                for bi, oidx in enumerate(active):
                    will_expand, chosen_feat = expand_decisions[bi]

                    if not depths[oidx]:
                        envs[oidx].done = True
                        continue

                    if not will_expand:
                        seqs[oidx].append(LEAF_TOKEN)
                        if ras_counts is not None:
                            ras_counts[tuple(seqs[oidx])] = ras_counts.get(tuple(seqs[oidx]), 0) + 1

                        envs[oidx].step(("leaf", 0))
                        depths[oidx].pop(); lo_stacks[oidx].pop(); hi_stacks[oidx].pop(); row_stacks[oidx].pop(); cd_stacks[oidx].pop()
                        if not depths[oidx]:
                            envs[oidx].done = True
                        else:
                            still_for_round.append(oidx)
                        continue

                    f_idx = int(chosen_feat)
                    feat_tok = int(v.split_start + f_idx)
                    seqs[oidx].append(feat_tok)
                    if ras_counts is not None:
                        ras_counts[tuple(seqs[oidx])] = ras_counts.get(tuple(seqs[oidx]), 0) + 1

                    envs[oidx].step(("feat", f_idx))

                    d0 = depths[oidx].pop()
                    lo_top = lo_stacks[oidx].pop()
                    hi_top = hi_stacks[oidx].pop()
                    rows_rel = row_stacks[oidx].pop()
                    cd_top = cd_stacks[oidx].pop()
                    need_threshold.append((oidx, f_idx, d0, lo_top.clone(), hi_top.clone(), rows_rel.clone(), cd_top))

                if need_threshold:
                    sub_idx = [oidx for (oidx, *_) in need_threshold]
                    sub_pad = torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(seqs[i], device=device) for i in sub_idx],
                        batch_first=True, padding_value=v.PAD
                    )
                    sub_logits, _ = self.pf(sub_pad)
                    last_th = sub_logits[:, -1, :]

                    th_base = v.split_start + v.num_feat
                    for si, (oidx, f_idx, d0, lo_top, hi_top, rows_rel, cd_top) in enumerate(need_threshold):
                        Xb = envs[oidx].X_full[envs[oidx].idxs]
                        bf = Xb.index_select(0, rows_rel)[:, f_idx]

                        if bf.numel() == 0:
                            envs[oidx].done = True
                            continue

                        uniq, counts = torch.unique(bf, sorted=True, return_counts=True)
                        if uniq.numel() < 2:
                            depths[oidx].append(d0)
                            lo_stacks[oidx].append(lo_top)
                            hi_stacks[oidx].append(hi_top)
                            row_stacks[oidx].append(rows_rel)
                            cd_stacks[oidx].append(max(0, cd_top - 1))
                            still_for_round.append(oidx)
                            continue

                        th_all_bins = uniq[:-1]
                        th_pos_all  = torch.arange(th_all_bins.numel(), device=device)

                        mcs = int(c.min_child_size or 0)
                        mask_mc = torch.ones_like(th_pos_all, dtype=torch.bool)
                        if mcs > 1:
                            csum_all = counts.cumsum(0)[:-1]
                            n_leaf = int(bf.numel())
                            left_ok  = csum_all >= mcs
                            right_ok = (n_leaf - csum_all) >= mcs
                            mask_mc = left_ok & right_ok

                        lo_f = int(lo_top[f_idx].item())
                        hi_f = int(hi_top[f_idx].item())
                        mask_win = (th_all_bins >= lo_f) & (th_all_bins <= hi_f)

                        mask_final = mask_mc & mask_win
                        if not mask_final.any():
                            depths[oidx].append(d0)
                            lo_stacks[oidx].append(lo_top)
                            hi_stacks[oidx].append(hi_top)
                            row_stacks[oidx].append(rows_rel)
                            cd_stacks[oidx].append(max(0, cd_top - 1))
                            still_for_round.append(oidx)
                            continue

                        cand_bins = th_all_bins[mask_final]
                        cand_pos  = th_pos_all[mask_final]
                        th_ids = th_base + cand_bins.to(device=device, dtype=torch.long)

                        if th_imbal_gamma > 0.0 and mcs > 0:
                            csum_all = counts.cumsum(0)[:-1].float()
                            n_leaf = float(bf.numel())
                            left_counts = csum_all.index_select(0, cand_pos)
                            imbal = (2.0 * (left_counts / max(1.0, n_leaf)) - 1.0).abs()
                            for jj, bval in enumerate(cand_bins.tolist()):
                                last_th[si, th_base + int(bval)] -= th_imbal_gamma * float(imbal[jj].item())

                        logits_th = last_th[si, th_ids]
                        if temp > 1e-9:
                            logits_th = logits_th / temp
                        probs_th = torch.softmax(logits_th, dim=0)
                        choice_idx = int(torch.multinomial(probs_th.unsqueeze(0), 1).item())
                        t_bin = int(cand_bins[choice_idx].item())
                        t_tok = int(th_base + t_bin)
                        seqs[oidx].append(t_tok)
                        if ras_counts is not None:
                            ras_counts[tuple(seqs[oidx])] = ras_counts.get(tuple(seqs[oidx]), 0) + 1

                        fv = Xb.index_select(0, rows_rel)[:, f_idx]
                        m = fv <= t_bin
                        rows_L = rows_rel[m]
                        rows_R = rows_rel[~m]

                        lo_L, hi_L = lo_top.clone(), hi_top.clone()
                        hi_L[f_idx] = torch.minimum(hi_L[f_idx], torch.as_tensor(t_bin, device=device))
                        lo_R, hi_R = lo_top.clone(), hi_top.clone()
                        lo_R[f_idx] = torch.maximum(lo_R[f_idx], torch.as_tensor(t_bin + 1, device=device))

                        depths[oidx].append(d0 + 1); lo_stacks[oidx].append(lo_R); hi_stacks[oidx].append(hi_R); row_stacks[oidx].append(rows_R)
                        depths[oidx].append(d0 + 1); lo_stacks[oidx].append(lo_L); hi_stacks[oidx].append(hi_L); row_stacks[oidx].append(rows_L)

                        if leaf_cd_steps > 0:
                            cd_stacks[oidx].append(leaf_cd_steps)
                            cd_stacks[oidx].append(leaf_cd_steps)
                        else:
                            cd_stacks[oidx].append(0)
                            cd_stacks[oidx].append(0)

                        envs[oidx].step(("th", int(t_bin)))
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
    # Predict
    # ========================================================
    @torch.no_grad()
    def _bin_test_like_train(
        self,
        env_template: TabularEnv,
        df_test: pd.DataFrame,
        df_train: pd.DataFrame,
    ) -> torch.Tensor:
        if hasattr(env_template, "transform_df"):
            X_te = env_template.transform_df(df_test)
            return X_te.to(env_template.device)

        if hasattr(env_template, "transform"):
            X_te = env_template.transform(df_test)
            return X_te.to(env_template.device)

        if hasattr(env_template, "featurise"):
            X_te = env_template.featurise(df_test)
            return X_te.to(env_template.device)

        if hasattr(env_template, "_featurise"):
            X_te = env_template._featurise(
                df_test, df_train, self.cfg.feature_cols, self.cfg.n_bins
            )
            return X_te.to(env_template.device)

        raise RuntimeError(
            "TabularEnv does not expose a known transform/featurise method."
        )

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
            df_train,
            feature_cols=c.feature_cols,
            target_col=c.target_col,
            n_bins=c.n_bins,
            task=c.task,
            binning_strategy=c.binning_strategy,
            device=c.device,
            min_data_in_bin=getattr(c, "min_data_in_bin", None),
            subsample_for_bin=getattr(c, "subsample_for_bin", None),
            per_feature_binning=getattr(c, "per_feature_binning", None),
        )

        if hasattr(env_template, "transform"):
            X_te = env_template.transform(df_test)
        elif hasattr(env_template, "_featurise"):
            try:
                X_te = env_template._featurise(df_test, is_train=False)  # type: ignore
            except TypeError:
                X_te = env_template._featurise(df_test, df_train, c.feature_cols, c.n_bins)  # type: ignore
        else:
            raise RuntimeError("TabularEnv has no transform/_featurise method for test data.")

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
            test_preds = torch.zeros((len(X_te), c.n_classes), device=device)
            train_preds = torch.zeros((len(y_tr), c.n_classes), device=device)
        else:
            test_preds = torch.full((len(X_te),), self.y_mean, device=device, dtype=torch.float32)
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
                envs, temp=c.rollout_temperature, residuals=residuals, beta=c.beta,
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
            test_preds += c.boosting_lr * (w * contrib_te)

        return torch.softmax(test_preds, dim=1) if c.task == "classification" else test_preds

    # sklearn compat
    def get_params(self, deep=True): return asdict(self.cfg)
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self.cfg, k, v)
        return self