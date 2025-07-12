from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import torch
import math
import random
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

# Assuming these imports are in the correct path
from src.tokenizer import Tokenizer, Vocab
from src.env import TabularEnv
from src.policy import PolicyPaperMLP
from src.utils import (
    ReplayBuffer,
    tb_loss,
    fl_loss,
    _safe_sample,
    get_tree_predictor,
    deltaE_split_gain_GH,
    create_gain_bias
)

@dataclass
class Config:
    feature_cols: List[str]
    target_col: str = "target"
    n_bins: int = 255
    device: str = "cuda"

    # Boost-GFN parameters
    updates: int = 50
    rollouts: int = 60
    batch_size: int = 8192  # Note: This parameter is not used in the corrected training loop.
    max_depth: int = 7
    top_k_trees: int = 10
    boosting_lr: float = 0.1

    # Policy network architecture
    lstm_hidden: int = 512
    mlp_layers: int = 12
    mlp_width: int = 256
    lr: float = 5e-5

    # Priors & annealing
    beta_start: float = 0.35
    beta_end: float = math.log(4)
    prior_scale: float = 0.5

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer: Optional[Tokenizer] = None
        self.ensemble: list[list[list[int]]] = []
        self.y_mean: float = 0.0
        self.pf: Optional[PolicyPaperMLP] = None
        self.pb: Optional[PolicyPaperMLP] = None
        self.log_z: Optional[torch.Tensor] = None
        self.replay_buffer: Optional[ReplayBuffer] = None

    def fit(self, df_train: pd.DataFrame) -> "Trainer":
        c, device = self.cfg, self.cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available → falling back to CPU.")
            device = "cpu"

        # ── 1. Tokenizer & environment ----------------------------------
        v = Vocab(len(c.feature_cols), c.n_bins, 1)
        self.tokenizer = Tokenizer(v)
        env = TabularEnv(df_train, c.feature_cols, c.target_col,
                         c.n_bins, device=device)
        y_true, X_bin = env.y_full, env.X_full

        # ── 2. Policy nets ----------------------------------------------
        self.pf = torch.jit.script(
            PolicyPaperMLP(v.size(), c.lstm_hidden,
                           c.mlp_layers, c.mlp_width).to(device))
        self.pb = torch.jit.script(
            PolicyPaperMLP(v.size(), c.lstm_hidden,
                           c.mlp_layers, c.mlp_width).to(device))

        # ── 3. Normaliser & optim ---------------------------------------
        self.log_z = torch.zeros((), device=device, requires_grad=True)
        optf = torch.optim.AdamW(self.pf.parameters(), lr=c.lr)
        optb = torch.optim.AdamW(self.pb.parameters(), lr=c.lr)
        optz = torch.optim.Adam([self.log_z], lr=c.lr / 10)
        optimizers = [optf, optb, optz]

        warm, tmax = 10, max(1, c.updates - 10)
        schedulers = [SequentialLR(opt, [LambdaLR(opt, lambda u: min(1,u/warm)),
                                         CosineAnnealingLR(opt, T_max=tmax)],
                                   milestones=[warm])
                      for opt in optimizers]

        # ── 4. Boosting-state -------------------------------------------
        self.replay_buffer = ReplayBuffer(100_000)
        self.y_mean = y_true.mean().item()
        f_ensemble = torch.full_like(y_true, self.y_mean)   # current pred
        BA, TA, FA = 20.0, 20.0, 20.0                       # anneal

        print(f"--- GFN-Boost training on {device} ---")
        for upd in tqdm(range(1, c.updates + 1), desc="Boost"):
            # residuals + GH stats for this round
            resid = (y_true - f_ensemble).detach()
            g = -2.0 * resid
            h = 2.0 * torch.ones_like(g)

            env.y = resid

            # β / temperature schedules
            beta = c.beta_start + min(1.0, (upd-1)/BA)*(c.beta_end-c.beta_start)
            temp = max(1.0, 5.0 - (upd-1)*(4.0/TA))
            lam_fl = min(1.0, upd / FA)

            # ---------- gather trajectories -----------------------------
            self.replay_buffer.data.clear()
            fwd_trajs = []
            
            for _ in range(c.rollouts):
                seq = self.rollout(env, temp)
                if seq:
                    R_sum = deltaE_split_gain_GH(
                        torch.tensor([seq], device=device),
                        self.tokenizer, env, g, h).sum().item()
                    self.replay_buffer.add(R_sum, seq, 0.0, None)
                    fwd_trajs.append((seq, R_sum, 0.0))

            rep_trajs = self.sample_replay(c.top_k_trees) if upd > 1 else []
            all_trajs = fwd_trajs + rep_trajs
            if not all_trajs:
                continue

            # ---------- optimise ---------------------------------------
            for opt in optimizers: opt.zero_grad()
            tb_total = fl_total = 0.0

            for seq, R_sum, prior in all_trajs:
                tok = torch.tensor([seq], device=device)
                log_pf = self.pf.log_prob(tok)
                log_pb = self.pb.log_prob(tok.flip(1))
                dR = deltaE_split_gain_GH(tok, self.tokenizer, env,
                                              g, h) + 1e-8
                pr_t = torch.tensor([prior], device=device)

                # FIX: Pad dR tensor if its length doesn't match log_pf
                if dR.shape[1] < log_pf.shape[1]:
                    padding_size = log_pf.shape[1] - dR.shape[1]
                    # Pad with 0s on the right side (for the final transitions)
                    dR = torch.nn.functional.pad(dR, (0, padding_size))

                l_tb = tb_loss(log_pf, log_pb, self.log_z, dR.sum(), pr_t)
                l_fl = fl_loss(self.pf.log_F(tok), log_pf, log_pb, dR)

                tb_total += l_tb.item()
                fl_total += l_fl.item()
                (l_tb + lam_fl * l_fl).backward()

            for opt in optimizers:
                torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
                opt.step()
            for sch in schedulers: sch.step()

            # ---------- add top-k trees to ensemble ---------------------
            topk = self.sample_replay(c.top_k_trees)
            if topk:
                self.ensemble.append([seq for seq, _, _ in topk])
                delta_pred = torch.zeros_like(f_ensemble)
                for seq, _, _ in topk:
                    delta_pred += get_tree_predictor(
                        seq, X_bin, resid, self.tokenizer)(X_bin)
                f_ensemble += c.boosting_lr * delta_pred / len(topk)

            # ---------- stats ------------------------------------------
            train_corr = torch.corrcoef(torch.stack([f_ensemble, y_true]))[0,1].item()
            n_traj = len(all_trajs)
            tqdm.write(f"Upd {upd:02d}  TB {tb_total/n_traj:.4f}  "
                       f"FL {fl_total/n_traj:.4f}  Corr {train_corr:+.4f}")

        return self

    def sample_replay(self, k: int) -> list[tuple[list[int], float, float]]:
        buf = self.replay_buffer
        if not buf or not buf.data:
            return []
        entries = buf.data
        weights = []
        with torch.no_grad():
            for R, seq, pr, _ in entries:
                tok = torch.tensor([seq], device=self.cfg.device)
                tok_bwd = torch.flip(tok, dims=[1])
                log_pb = self.pb.log_prob(tok_bwd)
                weights.append(log_pb.sum().exp().item())

        total = sum(weights)
        k = min(k, len(entries))
        if total > 0:
            probs = [w / total for w in weights]
            idxs = random.choices(range(len(entries)), weights=probs, k=k)
        else:
            idxs = random.sample(range(len(entries)), k)

        return [(entries[i][1], entries[i][0], entries[i][2]) for i in idxs]

    def rollout(self, env: TabularEnv, temp: float) -> Optional[list[int]]:
        c = self.cfg
        v = self.tokenizer.v
        seq: list[int] = [v.BOS]
        depths = deque([0])
        
        # CORRECTED: Reset environment for a single trajectory.
        env.reset(1)

        with torch.no_grad():
            while not env.done and depths:
                x = torch.tensor([seq], device=c.device)
                logits, _ = self.pf(x)
                logits = logits[0, -1]
                mask = torch.zeros_like(logits, dtype=torch.bool)
                d = depths[-1]

                if env.open_leaves > 0 and d < c.max_depth:
                    mask[v.split_start : v.split_start + v.num_feat] = True
                if env.open_leaves > 0:
                    mask[v.split_start + v.num_feat + v.num_th :] = True
                if not mask.any():
                    break

                tok1 = _safe_sample(logits, mask, temp)
                kind, idx = self.tokenizer.decode_one(tok1)
                
                if kind == 'feat':
                    d0 = depths.pop()
                    depths.extend([d0 + 1, d0 + 1])
                else: # stop
                    depths.pop()

                env.step((kind, idx))
                seq.append(tok1)

                if kind == 'feat':
                    x2 = torch.tensor([seq], device=c.device)
                    logits2, _ = self.pf(x2)
                    th = logits2[0, -1]
                    m2 = torch.zeros_like(th, dtype=torch.bool)
                    start = v.split_start + v.num_feat
                    m2[start : start + v.num_th] = True
                    tok2 = _safe_sample(th, m2, temp)
                    env.step(self.tokenizer.decode_one(tok2))
                    seq.append(tok2)

            seq.append(v.EOS)
            return seq if env.open_leaves == 0 else None

    def predict(self, df_test: pd.DataFrame, df_train: pd.DataFrame) -> np.ndarray:
        if self.tokenizer is None or not self.ensemble:
            raise RuntimeError("Call fit() first.")
        
        c = self.cfg
        device = c.device

        # --- Setup environment for featurization ---
        env = TabularEnv(df_train, feature_cols=c.feature_cols, target_col=c.target_col, n_bins=c.n_bins, device=device)
        X_te = env._featurise(df_test, df_train, c.feature_cols, c.n_bins)
        X_tr, y_tr = env.X_full.clone(), env.y_full.clone()

        # --- PREDICTION LOGIC ---
        test_predictions = torch.full((len(df_test),), self.y_mean, device=device)
        train_predictions = torch.full_like(y_tr, self.y_mean)

        for trees_in_round in tqdm(self.ensemble, desc="Ensemble Prediction", leave=False):
            if not trees_in_round:
                continue

            residuals_for_this_round = y_tr - train_predictions

            avg_pred_on_test = torch.zeros_like(test_predictions)
            avg_pred_on_train = torch.zeros_like(train_predictions)

            for seq in trees_in_round:
                predictor = get_tree_predictor(seq, X_tr, residuals_for_this_round, self.tokenizer)
                avg_pred_on_test += predictor(X_te)
                avg_pred_on_train += predictor(X_tr)

            avg_pred_on_test /= len(trees_in_round)
            avg_pred_on_train /= len(trees_in_round)

            test_predictions += c.boosting_lr * avg_pred_on_test
            train_predictions += c.boosting_lr * avg_pred_on_train

        return test_predictions.cpu().numpy()

    def get_params(self, deep=True):
        return asdict(self.cfg)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self.cfg, k, v)
        return self