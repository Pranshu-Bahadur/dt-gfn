# src/trainer.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


# --------------------------------------------------------------------------- #
# 1.  Hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    feature_cols: List[str]
    target_col: str = "target"
    n_bins: int = 255
    device: str = "cuda"

    # Boost-GFN
    updates: int = 50
    rollouts: int = 60
    batch_size: int = 8192
    max_depth: int = 7
    top_k_trees: int = 10
    boosting_lr: float = 0.1

    # Policy net
    lstm_hidden: int = 512
    mlp_layers: int = 12
    mlp_width: int = 256
    lr: float = 5e-5

    # Priors & Annealing
    beta_start: float = 0.35
    beta_end: float = math.log(4)
    prior_scale: float = 0.5


# --------------------------------------------------------------------------- #
# 2.  Trainer (fit + predict)
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer: Optional[Tokenizer] = None
        self.ensemble: list[list[list[int]]] = []
        self.y_mean: float = 0.0

    # --------------------------------------------------------------------- #
    # fit
    # --------------------------------------------------------------------- #
    def fit(self, df_train: pd.DataFrame) -> "Trainer":
        c = self.cfg
        device = c.device

        # ---- Tokenizer and Vocab -----------------------------------------
        v = Vocab(len(c.feature_cols), c.n_bins, 1)
        self.tokenizer = Tokenizer(v)

        # ---- Environment --------------------------------------------------
        env = TabularEnv(
            df_train,
            feature_cols=c.feature_cols,
            target_col=c.target_col,
            n_bins=c.n_bins,
            device=device,
        )
        
        y_tr_true, X_tr_binned = env.y_full.clone(), env.X_full.clone()

        # ---- Gain-Based Prior ---------------------------------------------
        prior_bias = create_gain_bias(df_train, c.feature_cols, c.target_col, self.tokenizer, c.n_bins, c.prior_scale).to(device)

        # ---- Policy Nets --------------------------------------------------
        pf = PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(device)
        pb = PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(device)
        pf = torch.jit.script(pf); pb = torch.jit.script(pb)
        with torch.no_grad():
            pf.head_tok.bias.copy_(prior_bias)
            pb.head_tok.bias.copy_(prior_bias)
        

        log_z = torch.zeros((), device=device, requires_grad=True)
        optf, optb, optz = torch.optim.AdamW(pf.parameters(), lr=c.lr), torch.optim.AdamW(pb.parameters(), lr=c.lr), torch.optim.Adam([log_z], lr=c.lr/10)
        optimizers = [optf, optb, optz]
        
        # ---- LR Schedulers ------------------------------------------------
        warmup_updates, t_max_val = 10, max(1, c.updates - 10)
        warmup_schedulers = [LambdaLR(opt, lr_lambda=lambda upd: min(1.0, upd / warmup_updates)) for opt in optimizers]
        decay_schedulers = [CosineAnnealingLR(opt, T_max=t_max_val) for opt in optimizers]
        schedulers = [SequentialLR(opt, schedulers=[ws, ds], milestones=[warmup_updates]) for opt, ws, ds in zip(optimizers, warmup_schedulers, decay_schedulers)]
        
        # ---- Replay Buffer & Baseline -------------------------------------
        buf = ReplayBuffer(capacity=10000)
        base_prediction = torch.full_like(y_tr_true, y_tr_true.mean())
        self.y_mean = y_tr_true.mean().item()
        
        BETA_ANNEAL_UPDATES, TEMP_ANNEAL_UPDATES, FL_LOSS_ANNEAL_UPDATES = 20.0, 20.0, 20.0

        # ------------------------------------------------------------------ #
        # Boost loop
        # ------------------------------------------------------------------ #
        print("--- Starting GFN-Boost Training ---")
        for upd in range(1, c.updates + 1):
            residuals, env.y = y_tr_true - base_prediction, y_tr_true - base_prediction
            progress = min(1.0, (upd - 1) / BETA_ANNEAL_UPDATES)
            current_beta = 0.01#c.beta_start + progress * (c.beta_end - c.beta_start)
            temperature = max(1.0, 5.0 - (upd - 1) * (4.0 / TEMP_ANNEAL_UPDATES))
            lam_fl = 0.1#min(1.0, upd / FL_LOSS_ANNEAL_UPDATES)
            tb_loss_acc, fl_loss_acc, complete_rollouts_this_update = 0, 0, 0
            
            for opt in optimizers: opt.zero_grad()
            
            pbar = tqdm(range(c.rollouts), desc=f"Boosting Round {upd:02d}", leave=False)
            for _ in pbar:
                env.reset(c.batch_size)
                seq = [v.BOS]
                open_leaf_depths = deque([0]) 
                with torch.no_grad():
                    while not env.done:
                        if not open_leaf_depths: break
                        x = torch.tensor([seq], device=device)
                        logits, _ = pf.forward(x)
                        logits = logits[0, -1]
                        mask = torch.zeros_like(logits, dtype=torch.bool)
                        current_depth = open_leaf_depths[-1]
                        if env.open_leaves > 0 and current_depth < c.max_depth:
                            mask[v.split_start : v.split_start + v.num_feat] = True
                        if env.open_leaves > 0:
                            mask[v.split_start + v.num_feat + v.num_th:] = True
                        if not mask.any(): break
                        tok1 = _safe_sample(logits, mask, temperature)
                        kind, idx = self.tokenizer.decode_one(tok1)
                        if kind == 'feat':
                            d = open_leaf_depths.pop(); open_leaf_depths.append(d + 1); open_leaf_depths.append(d + 1)
                        else: open_leaf_depths.pop()
                        env.step((kind, idx)); seq.append(tok1)
                        if kind == 'feat':
                            x2, _ = pf.forward(torch.tensor([seq], device=device))
                            logits_th = x2[0, -1]
                            th_mask = torch.zeros_like(logits_th, dtype=torch.bool)
                            th_mask[v.split_start+v.num_feat : v.split_start+v.num_feat+v.num_th] = True
                            tok2 = _safe_sample(logits_th, th_mask, temperature)
                            env.step(self.tokenizer.decode_one(tok2)); seq.append(tok2)
                seq.append(v.EOS)

                if env.open_leaves != 0: continue
                complete_rollouts_this_update += 1
                R, prior, _, _ = env.evaluate(current_beta)
                buf.add(R, seq, prior.item(), env.idxs.clone())
                
                tokens_fwd = torch.tensor([seq], device=device)
                log_pf_fwd = pf.log_prob(tokens_fwd)
                logF = pf.log_F(tokens_fwd)
                tokens_bwd = torch.flip(tokens_fwd, dims=[1])
                log_pb = pb.log_prob(tokens_bwd)
                dE = deltaE_split_gain(tokens_fwd, self.tokenizer, env)
                
                l_tb = tb_loss(log_pf_fwd, log_pb, log_z, torch.tensor([R], device=device).unsqueeze(0), prior)
                l_fl = fl_loss(logF, log_pf_fwd, log_pb, dE)
                loss = l_tb + lam_fl * l_fl
                loss.backward()
                tb_loss_acc += l_tb.item(); fl_loss_acc += l_fl.item()

            did_any_backward = complete_rollouts_this_update > 0
            if did_any_backward:
                denom = complete_rollouts_this_update
                torch.nn.utils.clip_grad_norm_(pf.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(pb.parameters(), 1.0)
                torch.nn.utils.clip_grad_value_([log_z], 1.0)
                for opt in optimizers:
                    for group in opt.param_groups:
                        for param in group['params']:
                            if param.grad is not None: param.grad /= denom
                for opt in optimizers: opt.step()
            
            if did_any_backward:
                for scheduler in schedulers: scheduler.step()
            
            if buf.data:
                top_k_trajectories = buf.data[:c.top_k_trees]
                top_k_sequences = [t[1] for t in top_k_trajectories]
                self.ensemble.append(top_k_sequences)
                avg_residual_preds = torch.zeros_like(base_prediction)
                for best_seq in top_k_sequences:
                    predictor = get_tree_predictor(best_seq, X_tr_binned, residuals, self.tokenizer)
                    avg_residual_preds += predictor(X_tr_binned)
                if len(top_k_sequences) > 0:
                    avg_residual_preds /= len(top_k_sequences)
                
                base_prediction += c.boosting_lr * avg_residual_preds
                
                final_corr = torch.corrcoef(torch.stack([base_prediction, y_tr_true]))[0,1].item()
                print(f"Boosting Round {upd:02d} | Comp: {complete_rollouts_this_update}/{c.rollouts} | "
                      f"TB: {tb_loss_acc/denom if did_any_backward else 0:.4f} | FL: {fl_loss_acc/denom if did_any_backward else 0:.4f} | Train ρ: {final_corr:+.3f}")
            else:
                print(f"Boosting Round {upd:02d} | No complete trees generated.")
            
            buf.data.clear()

        return self

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #
    def predict(self, df_test: pd.DataFrame, df_train: pd.DataFrame) -> np.ndarray:
        if self.tokenizer is None:
            raise RuntimeError("Call fit() first.")

        device = self.cfg.device
        
        env = TabularEnv(
            df_train,
            feature_cols=self.cfg.feature_cols,
            target_col=self.cfg.target_col,
            n_bins=self.cfg.n_bins,
            device=device,
        )
        
        X_te_binned = env._featurise(df_test, df_train, self.cfg.feature_cols, self.cfg.n_bins)
        final_test_preds = torch.full((len(df_test),), self.y_mean, device=device)
        
        X_tr_binned = env.X_full.clone()
        y_tr_true = env.y_full.clone()
        inference_residuals = y_tr_true.clone()
        
        print("\n--- Training finished. Starting Boosted Inference. ---")
        for i, top_k_in_round in enumerate(tqdm(self.ensemble, desc="Building boosted ensemble")):
            avg_train_preds_for_round, avg_test_preds_for_round = torch.zeros_like(y_tr_true), torch.zeros_like(final_test_preds)
            if not top_k_in_round: continue
            
            for tree_seq in top_k_in_round:
                predictor = get_tree_predictor(tree_seq, X_tr_binned, inference_residuals, self.tokenizer)
                avg_train_preds_for_round += predictor(X_tr_binned)
                avg_test_preds_for_round += predictor(X_te_binned)
                
            avg_train_preds_for_round /= len(top_k_in_round)
            avg_test_preds_for_round /= len(top_k_in_round)

            final_test_preds += self.cfg.boosting_lr * avg_test_preds_for_round
            inference_residuals -= self.cfg.boosting_lr * avg_train_preds_for_round
            
        return final_test_preds.cpu().numpy()

    # ------------------------------------------------------------------ #
    # sklearn helpers
    # ------------------------------------------------------------------ #
    def get_params(self, deep=True):
        return asdict(self.cfg)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self.cfg, k, v)
        return self