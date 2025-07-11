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

from src.tokenizer import Tokenizer, Vocab
from src.env import TabularEnv
from src.policy import PolicyPaperMLP
from src.utils import (
    ReplayBuffer,
    tb_loss,
    fl_loss,
    _safe_sample,
    get_tree_predictor,
    deltaE_split_gain,
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
    batch_size: int = 8192
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
        c = self.cfg
        device = c.device

        # --- Setup tokenizer and environment ---
        v = Vocab(len(c.feature_cols), c.n_bins, 1)
        self.tokenizer = Tokenizer(v)
        env = TabularEnv(
            df_train,
            feature_cols=c.feature_cols,
            target_col=c.target_col,
            n_bins=c.n_bins,
            device=device
        )
        y_true, X_binned = env.y_full.clone(), env.X_full.clone()

        # --- Policy networks ---
        pf = PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(device)
        pb = PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(device)
        pf = torch.jit.script(pf)
        pb = torch.jit.script(pb)
        self.pf, self.pb = pf, pb

        # --- Normalizer ---
        log_z = torch.zeros((), device=device, requires_grad=True)
        self.log_z = log_z

        # --- Optimizers and schedulers ---
        optf = torch.optim.AdamW(pf.parameters(), lr=c.lr)
        optb = torch.optim.AdamW(pb.parameters(), lr=c.lr)
        optz = torch.optim.Adam([log_z], lr=c.lr / 10)
        optimizers = [optf, optb, optz]

        warmup = 10
        tmax = max(1, c.updates - warmup)
        schedulers = []
        for opt in optimizers:
            ws = LambdaLR(opt, lr_lambda=lambda u: min(1.0, u / warmup))
            ds = CosineAnnealingLR(opt, T_max=tmax)
            schedulers.append(SequentialLR(opt, schedulers=[ws, ds], milestones=[warmup]))

        # --- Replay buffer and baseline ---
        buf = ReplayBuffer(capacity=100000)
        self.replay_buffer = buf
        self.y_mean = y_true.mean().item()
        base_pred = torch.full_like(y_true, self.y_mean)
        
        # --- Annealing constants ---
        BA, TA, FA = 20.0, 20.0, 20.0

        print("--- Starting GFN-Boost Training ---")
        for upd in tqdm(range(1, c.updates + 1), desc="Boost Updates"):
            # Compute residuals and set environment target
            residuals = y_true - base_pred
            env.y = residuals.clone()

            tb_acc, fl_acc, complete = 0.0, 0.0, 0
            prog = min(1.0, (upd - 1) / BA)
            beta = 0.01  # c.beta_start + prog * (c.beta_end - c.beta_start)
            temp = max(1.0, 5.0 - (upd - 1) * (4.0 / TA))
            lam_fl = 0.01  # min(1.0, upd / FA)

            # Zero gradients
            for opt in optimizers:
                opt.zero_grad()

            # --- Forward rollouts ---
            forward_seqs: list[tuple[list[int], float, float]] = []
            for _ in tqdm(range(c.rollouts), desc=f"Rollouts {upd:02d}", leave=False):
                seq = self.rollout(env, temp)
                if seq is None:
                    continue
                complete += 1
                R, prior, _, _ = env.evaluate(beta)
                buf.add(R, seq, prior.item(), env.idxs.clone())
                forward_seqs.append((seq, R, prior.item()))

            # --- Backward (replay) samples ---
            replay_seqs: list[tuple[list[int], float, float]] = []
            if buf.data and upd > 1:
                replay_seqs = self.sample_replay(c.top_k_trees)

            # --- Compute losses and backpropagate ---
            for seq, R, prior in forward_seqs + replay_seqs:
                tok = torch.tensor([seq], device=device)
                log_pf = pf.log_prob(tok)
                log_pb = pb.log_prob(torch.flip(tok, dims=[1]))
                
                # Use split gain on the current residuals as the reward
                R_t = deltaE_split_gain(tok, self.tokenizer, env) + 1e-8
                pr_t = torch.tensor([prior], device=device)

                # Trajectory balance loss
                l_tb = tb_loss(log_pf, log_pb, log_z, R_t.sum(), pr_t)
                tb_acc += l_tb.item()

                # (Optional) flow loss
                l_fl = fl_loss(pf.log_F(tok), log_pf, log_pb, R_t)
                fl_acc += l_fl.item()

                (l_tb + l_fl * lam_fl).backward()

            # --- Optimizer step ---
            if complete > 0:
                for opt in optimizers:
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
                    opt.step()
                for sch in schedulers:
                    sch.step()

            # --- Update ensemble and baseline prediction ---
            if buf.data:
                topk = self.sample_replay(c.top_k_trees)
                self.ensemble.append([seq for seq, _, _ in topk])
                
                avg_pred_on_residuals = torch.zeros_like(base_pred)
                for seq in topk:
                    # Predictor is trained on the current set of residuals
                    tree_predictor = get_tree_predictor(seq[0], X_binned, residuals, self.tokenizer)
                    avg_pred_on_residuals += tree_predictor(X_binned)
                
                avg_pred_on_residuals /= len(topk)
                base_pred += c.boosting_lr * avg_pred_on_residuals

            # --- Print stats ---
            corr = torch.corrcoef(torch.stack([base_pred, y_true]))[0, 1].item()
            avg_tb = tb_acc / max(1, complete)
            avg_fl = fl_acc / max(1, complete)
            print(f"Update {upd}/{c.updates} | TB Loss: {avg_tb:.4f} | FL Loss: {avg_fl:.4f} | Train Corr: {corr:+.4f}")

            # Clear buffer for the next boosting iteration
            buf.data.clear()

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
        env.reset(c.batch_size)

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

        # --- CORRECTED PREDICTION LOGIC ---
        # Start with the initial baseline prediction
        test_predictions = torch.full((len(df_test),), self.y_mean, device=device)
        train_predictions = torch.full_like(y_tr, self.y_mean)

        # Iterate through the stored ensemble, which contains one set of trees for each boosting round
        for trees_in_round in tqdm(self.ensemble, desc="Ensemble Prediction", leave=False):
            if not trees_in_round:
                continue

            # 1. Determine the residuals these trees were trained on
            # This is the crucial step: we use the state of the ensemble *before* this round
            residuals_for_this_round = y_tr - train_predictions

            # 2. Get the average prediction from this round's trees
            avg_pred_on_test = torch.zeros_like(test_predictions)
            avg_pred_on_train = torch.zeros_like(train_predictions)

            for seq in trees_in_round:
                # The tree predictor needs the correct residual target it was trained on
                predictor = get_tree_predictor(seq, X_tr, residuals_for_this_round, self.tokenizer)
                avg_pred_on_test += predictor(X_te)
                avg_pred_on_train += predictor(X_tr)

            avg_pred_on_test /= len(trees_in_round)
            avg_pred_on_train /= len(trees_in_round)

            # 3. Add this round's predictions to the total
            # This is a stateless update to the final prediction
            test_predictions += c.boosting_lr * avg_pred_on_test
            
            # 4. Update the training predictions to calculate residuals for the *next* round
            train_predictions += c.boosting_lr * avg_pred_on_train

        return test_predictions.cpu().numpy()

    def get_params(self, deep=True):
        return asdict(self.cfg)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self.cfg, k, v)
        return self