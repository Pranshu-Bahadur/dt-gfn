from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Deque
from collections import deque
import copy

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

    # Parallelism
    num_parallel: int = 10  # Adjust based on VRAM; 8 should fit within 40GB if current is 25GB

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

        # --- Setup tokenizer and environment template ---
        v = Vocab(len(c.feature_cols), c.n_bins, 1)
        self.tokenizer = Tokenizer(v)
        env_template = TabularEnv(
            df_train,
            feature_cols=c.feature_cols,
            target_col=c.target_col,
            n_bins=c.n_bins,
            device=device
        )
        y_true, X_binned = env_template.y_full.clone(), env_template.X_full.clone()

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
            env_template.y = residuals.clone()

            tb_acc, fl_acc, complete = 0.0, 0.0, 0
            prog = min(1.0, (upd - 1) / BA)
            beta = 0.01  # c.beta_start + prog * (c.beta_end - c.beta_start)
            temp = max(1.0, 5.0 - (upd - 1) * (4.0 / TA))
            lam_fl = 0.01  # min(1.0, upd / FA)

            # Zero gradients
            for opt in optimizers:
                opt.zero_grad()

            # --- Forward rollouts with GPU batching ---
            forward_tuples: list[tuple[list[int], float]] = []
            rollouts_done = 0
            with tqdm(total=c.rollouts, desc=f"Batch Rollouts {upd:02d}", leave=False) as pbar:
                while rollouts_done < c.rollouts:
                    batch_size = min(c.num_parallel, c.rollouts - rollouts_done)
                    
                    envs = [env_template] * batch_size
                    batch_results = self.batched_rollout(envs, temp, residuals, beta)
                    
                    for result in batch_results:
                        if result is None:
                            continue
                        seq, prior, idxs = result
                        complete += 1
                        buf.add(0.0, seq, prior, idxs)
                        forward_tuples.append((seq, prior))
                    
                    rollouts_done += batch_size
                    pbar.update(batch_size)

            # --- Backward (replay) samples ---
            replay_tuples: list[tuple[list[int], float]] = []
            if buf.data and upd > 1:
                replay_tuples = self.sample_replay(c.top_k_trees)

            # --- Compute losses and backpropagate (no sorting here to avoid extra R_t calc) ---
            
            # FIX: Reset the template environment. This initializes `env_template.idxs` to
            # the full range of training samples, which is the required state for
            # deltaE_split_gain to evaluate trajectories starting from the root.
            env_template.reset(c.batch_size)
            
            for seq, prior in forward_tuples + replay_tuples:
                tok = torch.tensor([seq], device=device)
                log_pf = pf.log_prob(tok)
                log_pb = pb.log_prob(torch.flip(tok, dims=[1]))
                
                # Use split gain on the current residuals as the reward
                R_t = deltaE_split_gain(tok, self.tokenizer, env_template) + 1e-8
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
                # sample_replay returns (seq, prior) tuples
                topk_tuples = self.sample_replay(c.top_k_trees)
                topk_seqs = [seq for seq, _ in topk_tuples]
                if not topk_seqs: continue

                # Sort top-k by reward (descending split gain) right before boosting part
                tree_rewards = []
                for seq in topk_seqs:
                    tok = torch.tensor([seq], device=device)
                    R_t = deltaE_split_gain(tok, self.tokenizer, env_template).sum().item()
                    tree_rewards.append(R_t)

                sorted_indices = sorted(range(len(topk_seqs)), key=lambda i: tree_rewards[i], reverse=True)
                topk_seqs = [topk_seqs[i] for i in sorted_indices]

                self.ensemble.append(topk_seqs)
                
                avg_pred_on_residuals = torch.zeros_like(base_pred)
                for seq in topk_seqs:
                    # Predictor is trained on the current set of residuals
                    tree_predictor = get_tree_predictor(seq, X_binned, residuals, self.tokenizer)
                    avg_pred_on_residuals += tree_predictor(X_binned)
                
                if topk_seqs:
                    avg_pred_on_residuals /= len(topk_seqs)

                base_pred += c.boosting_lr * avg_pred_on_residuals

            # --- Print stats ---
            corr = torch.corrcoef(torch.stack([base_pred, y_true]))[0, 1].item()
            avg_tb = tb_acc / max(1, complete)
            avg_fl = fl_acc / max(1, complete)
            print(f"Update {upd}/{c.updates} | TB Loss: {avg_tb:.4f} | FL Loss: {avg_fl:.4f} | Train Corr: {corr:+.4f}")

            # Clear buffer for the next boosting iteration
            buf.data.clear()

        return self

    def sample_replay(self, k: int) -> list[tuple[list[int], float]]:
        buf = self.replay_buffer
        if not buf or not buf.data:
            return []
        entries = list(buf.data) # entries are (R, seq, pr, idxs)
        weights = []
        with torch.no_grad():
            for _, seq, _, _ in entries:
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

        # Return list of (sequence, prior)
        return [(entries[i][1], entries[i][2]) for i in idxs]

    def batched_rollout(
        self, envs: List[TabularEnv], temp: float, residuals: torch.Tensor, beta: float
    ) -> List[Optional[Tuple[list[int], float, torch.Tensor]]]:
        c = self.cfg
        v = self.tokenizer.v
        num = len(envs)
        device = c.device
        
        # Create independent shallow copies of the environment for each rollout
        local_envs = [copy.copy(env) for env in envs]
        for env in local_envs:
            env.reset(c.batch_size)
            env.y = residuals  # Share residuals tensor

        seqs: List[List[int]] = [[v.BOS] for _ in range(num)]
        depths: List[Deque[int]] = [deque([0]) for _ in range(num)]
        
        active_indices = list(range(num))
        final_results: List[Optional[Tuple[list[int], float, torch.Tensor]]] = [None] * num

        with torch.no_grad():
            while active_indices:
                # --- Step 1: Sample Feature or Leaf action ---
                
                # Batch preparation
                batch_seqs_tensors = [torch.tensor(seqs[i], device=device) for i in active_indices]
                x_batch = torch.nn.utils.rnn.pad_sequence(batch_seqs_tensors, batch_first=True, padding_value=v.PAD)

                logits_batch, _ = self.pf(x_batch)
                last_logits = logits_batch[:, -1, :]  # [num_active, vocab_size]

                # Create action masks for the active environments
                masks = []
                for i in active_indices:
                    env, d_q = local_envs[i], depths[i]
                    d = d_q[-1] if d_q else c.max_depth
                    mask = torch.zeros(v.size(), dtype=torch.bool, device=device)
                    if env.open_leaves > 0 and d < c.max_depth:
                        mask[v.split_start : v.split_start + v.num_feat] = True
                    if env.open_leaves > 0:
                        mask[v.split_start + v.num_feat + v.num_th :] = True
                    masks.append(mask)
                batch_mask = torch.stack(masks)

                # Sample actions for all active environments at once
                toks1 = _safe_sample(last_logits, batch_mask, temp)

                # --- Update states and identify which rollouts need a threshold ---
                
                needs_threshold_map = {} # {sub_batch_idx: original_active_idx}
                
                still_active_indices = []
                for i, original_idx in enumerate(active_indices):
                    tok1 = toks1[i].item()
                    kind, idx = self.tokenizer.decode_one(tok1)

                    seqs[original_idx].append(tok1)
                    local_envs[original_idx].step((kind, idx))

                    if kind == 'feat':
                        d0 = depths[original_idx].pop()
                        depths[original_idx].extend([d0 + 1, d0 + 1])
                        needs_threshold_map[len(needs_threshold_map)] = i
                    else: # kind == 'leaf'
                        depths[original_idx].pop()
                    
                    if not depths[original_idx]:
                        local_envs[original_idx].done = True
                    
                    if not local_envs[original_idx].done:
                        still_active_indices.append(original_idx)

                # --- Step 2: Batched Threshold Sampling ---

                if needs_threshold_map:
                    sub_batch_active_indices = [active_indices[i] for i in needs_threshold_map.values()]
                    
                    sub_batch_seq_tensors = [torch.tensor(seqs[i], device=device) for i in sub_batch_active_indices]
                    x_sub_batch = torch.nn.utils.rnn.pad_sequence(sub_batch_seq_tensors, batch_first=True, padding_value=v.PAD)

                    sub_logits, _ = self.pf(x_sub_batch)
                    sub_last_logits = sub_logits[:, -1, :]

                    th_mask = torch.zeros_like(sub_last_logits, dtype=torch.bool, device=device)
                    th_start = v.split_start + v.num_feat
                    th_end = th_start + v.num_th
                    th_mask[:, th_start:th_end] = True
                    
                    toks2 = _safe_sample(sub_last_logits, th_mask, temp)

                    for i, original_idx in enumerate(sub_batch_active_indices):
                        tok2 = toks2[i].item()
                        seqs[original_idx].append(tok2)
                        local_envs[original_idx].step(self.tokenizer.decode_one(tok2))

                active_indices = still_active_indices

        # --- Finalize: Mark valid trajectories and evaluate ---
        for i in range(num):
            env = local_envs[i]
            if env.done and env.open_leaves == 0:
                seqs[i].append(v.EOS)
                # Evaluate to get the prior; reward is recalculated in the main loop
                _, prior, _, _ = env.evaluate(beta)
                final_results[i] = (seqs[i], prior.item(), env.idxs.clone())

        return final_results


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
            
            if trees_in_round:
                avg_pred_on_test /= len(trees_in_round)
                avg_pred_on_train /= len(trees_in_round)

            # 3. Add this round's predictions to the total (for test set)
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