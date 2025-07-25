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
    create_gain_bias
)

@dataclass
class Config:
    feature_cols: List[str]
    target_col: str = "target"
    task: str = "classification"
    reward_function: str = "bayesian"  # "gini" or "bayesian" for classification
    n_classes: Optional[int] = None
    n_bins: int = 255
    device: str = "cuda"

    # Boost-GFN parameters
    updates: reinstated = 50
    rollouts: int = 60
    batch_size: int = 8192
    max_depth: int = 7
    top_k_trees: int = 10
    boosting_lr: float = 0.1

    # Policy network architecture
    lstm_hidden: int = 512
    mlp_layers: int = 3
    mlp_width: int = 256
    lr: float = 1e-2

    # Priors & annealing
    beta: Optional[float] = None # Will be set based on the paper's formula
    prior_scale: float = 0.5

    # Parallelism
    num_parallel: int = 10
    
    # Inference
    policy_inference_trees: int = 500

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
        self.le: Optional[LabelEncoder] = None
        self.best_trees_so_far: set = set() # Track unique best trees

    def fit(self, df_train: pd.DataFrame) -> "Trainer":
        c = self.cfg
        device = c.device

        v = Vocab(len(c.feature_cols), c.n_bins, 1)
        self.tokenizer = Tokenizer(v)
        env_template = TabularEnv(
            df_train,
            feature_cols=c.feature_cols,
            target_col=c.target_col,
            n_bins=c.n_bins,
            task=c.task,
            device=device
        )
        if c.task == "classification":
            self.le = env_template.le
            c.n_classes = env_template.n_classes

        y_true, X_binned = env_template.y_full.clone(), env_template.X_full.clone()

        pf = PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(device)
        pb = PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(device)
        self.pf, self.pb = torch.jit.script(pf), torch.jit.script(pb)

        self.log_z = torch.nn.Parameter(torch.tensor(150.0 / 64, device=device))
        
        optimizers = [
            torch.optim.AdamW(self.pf.parameters(), lr=c.lr),
            torch.optim.AdamW(self.pb.parameters(), lr=c.lr),
            torch.optim.Adam([self.log_z], lr=c.lr / 10)
        ]

        warmup, tmax = 10, max(1, c.updates - 10)
        schedulers = [
            SequentialLR(opt, schedulers=[LambdaLR(opt, lr_lambda=lambda u: min(1.0, u / warmup)), CosineAnnealingLR(opt, T_max=tmax)], milestones=[warmup])
            for opt in optimizers
        ]

        self.replay_buffer = ReplayBuffer(capacity=100000)
        if c.task == "regression":
            self.y_mean = y_true.mean().item()
            base_pred = torch.full_like(y_true, self.y_mean, dtype=torch.float32)
        else:
            base_pred = torch.zeros((len(y_true), c.n_classes), device=device) if c.n_classes > 2 else torch.zeros(len(y_true), device=device, dtype=torch.float32)
        
        if c.beta is None and c.task == 'classification' and c.reward_function == 'bayesian':
            num_features = len(c.feature_cols)
            num_thresholds = c.n_bins
            c.beta = math.log(4) + math.log(num_features) + math.log(num_thresholds)
            print(f"Using beta derived from the paper's formula: {c.beta:.4f}")

        print("--- Starting GFN-Boost Training ---")
        for upd in tqdm(range(1, c.updates + 1), desc="Boost Updates"):
            
            if c.task == "regression":
                residuals = y_true - base_pred
            else: # classification
                if c.n_classes > 2:
                    probs = torch.softmax(base_pred, dim=1)
                    residuals = torch.nn.functional.one_hot(y_true, num_classes=c.n_classes) - probs
                else: # binary
                    probs = torch.sigmoid(base_pred)
                    residuals = y_true - probs
            
            env_template.y = residuals.clone()

            beta = c.beta if c.beta is not None else 0.1
            temp = max(1.0, 5.0 - (upd - 1) * (4.0 / 20.0))
            lam_fl = 0.1

            for opt in optimizers: opt.zero_grad()
            
            forward_tuples = self._collect_rollouts(env_template, temp, residuals, beta)
            
            reward_env = copy.copy(env_template)
            reward_env.reset(len(y_true))

            all_tuples = forward_tuples + self.sample_replay(c.top_k_trees)
            
            tb_loss_acc = 0.0
            fl_loss_acc = 0.0
            
            for seq, prior in all_tuples:
                tok = torch.tensor([seq], device=device)
                log_pf = self.pf.log_prob(tok)
                log_pb = self.pb.log_prob(torch.flip(tok, dims=[1]))
                
                R_t_per_step = None 
                if c.task == "classification":
                    if c.reward_function == "bayesian":
                        R_t = calculate_bayesian_reward(tok, self.tokenizer, reward_env, beta)
                    else: # gini
                        R_t_per_step = deltaE_split_gain_classification(tok, self.tokenizer, reward_env)
                        R_t = R_t_per_step.sum(1) + 1e-8
                else: # regression
                    R_t_per_step = deltaE_split_gain_regression(tok, self.tokenizer, env_template)
                    R_t = R_t_per_step.sum(1) + 1e-8

                l_tb = tb_loss(log_pf, log_pb, self.log_z, R_t, torch.tensor([prior], device=device))
                total_loss = l_tb
                tb_loss_acc += l_tb.item()
                
                if R_t_per_step is not None:
                    l_fl = fl_loss(self.pf.log_F(tok), log_pf, log_pb, R_t_per_step)
                    total_loss += lam_fl * l_fl
                    fl_loss_acc += l_fl.item()
                
                total_loss.backward()

            if forward_tuples:
                for opt in optimizers: torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0); opt.step()
                for sch in schedulers: sch.step()

            base_pred = self._update_ensemble(base_pred, X_binned, residuals, beta, reward_env)
            
            n_trajs = len(all_tuples)
            avg_tb_loss = tb_loss_acc / n_trajs if n_trajs > 0 else 0
            avg_fl_loss = fl_loss_acc / n_trajs if n_trajs > 0 else 0
            log_str = f"Update {upd}/{c.updates} | TB Loss: {avg_tb_loss:.4f} | FL Loss: {avg_fl_loss:.4f}"

            if c.task == "regression":
                corr = torch.corrcoef(torch.stack([base_pred.squeeze(), y_true.squeeze()]))[0, 1].item()
                tqdm.write(f"{log_str} | Train Corr: {corr:+.4f}")
            else:
                if c.n_classes > 2:
                  acc = (base_pred.argmax(1) == y_true).float().mean().item()
                else:
                  acc = ((torch.sigmoid(base_pred) > 0.5).long() == y_true).float().mean().item()
                tqdm.write(f"{log_str} | Train Acc: {acc:.4f}")

            self.replay_buffer.data.clear()
        return self

    def _collect_rollouts(self, env_template, temp, residuals, beta):
        forward_tuples = []
        rollouts_done = 0
        with tqdm(total=self.cfg.rollouts, desc=f"Rollouts", leave=False) as pbar:
            while rollouts_done < self.cfg.rollouts:
                batch_size = min(self.cfg.num_parallel, self.cfg.rollouts - rollouts_done)
                envs = [copy.copy(env_template) for _ in range(batch_size)]
                batch_results = self.batched_rollout(envs, temp, residuals, beta)
                for result in batch_results:
                    if result:
                        seq, prior, idxs = result
                        self.replay_buffer.add(0.0, seq, prior, idxs)
                        forward_tuples.append((seq, prior))
                rollouts_done += batch_size
                pbar.update(batch_size)
        return forward_tuples

    def _update_ensemble(self, base_pred, X_binned, residuals, beta, reward_env):
        if not self.replay_buffer.data: return base_pred
        
        # Get all unique trees from the replay buffer for this update
        all_available_trees = [list(t) for t in {tuple(e[1]) for e in self.replay_buffer.data}]

        if not all_available_trees: return base_pred

        # Evaluate all available trees to find the new top-k set
        tree_rewards = []
        for seq in all_available_trees:
            tok = torch.tensor([seq], device=self.cfg.device)
            if self.cfg.task == "classification":
                if self.cfg.reward_function == "bayesian":
                    R_t = calculate_bayesian_reward(tok, self.tokenizer, reward_env, beta).item()
                else:
                    R_t = deltaE_split_gain_classification(tok, self.tokenizer, reward_env).sum().item()
            else:
                R_t = deltaE_split_gain_regression(tok, self.tokenizer, reward_env).sum().item()
            tree_rewards.append(R_t)

        # Sort trees by reward and select top-k
        sorted_indices = sorted(range(len(all_available_trees)), key=lambda i: tree_rewards[i], reverse=True)
        new_top_k_trees = [tuple(all_available_trees[i]) for i in sorted_indices[:self.cfg.top_k_trees]]
        
        # Convert to set for comparison, ensuring tuples are used
        new_top_k_set = {tuple(seq) for seq in new_top_k_trees}

        # Only update if we found new unique trees
        if new_top_k_set != self.best_trees_so_far:
            print(f"Found {len(new_top_k_set - self.best_trees_so_far)} new unique top-{self.cfg.top_k_trees} trees. Updating residuals.")
            self.best_trees_so_far = new_top_k_set
            
            # Use only the new top-k trees for the boosting update
            avg_pred_on_residuals = torch.zeros_like(base_pred)
            for seq_tuple in new_top_k_trees:
                seq = list(seq_tuple)
                predictor = get_tree_predictor(seq, X_binned, residuals, self.tokenizer)
                avg_pred_on_residuals += predictor(X_binned)
            
            if new_top_k_trees:
                avg_pred_on_residuals /= len(new_top_k_trees)
            base_pred += self.cfg.boosting_lr * avg_pred_on_residuals
            
            # Store this new best set in the main ensemble for prediction
            self.ensemble.append([list(t) for t in new_top_k_trees])
        
        return base_pred

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

    def batched_rollout(self, envs, temp, residuals, beta):
        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        num = len(envs)
        for env in envs: env.reset(c.batch_size); env.y = residuals
        seqs, depths = [[v.BOS] for _ in range(num)], [deque([0]) for _ in range(num)]
        active_indices, final_results = list(range(num)), [None] * num

        with torch.no_grad():
            while active_indices:
                batch_seqs_tensors = torch.nn.utils.rnn.pad_sequence([torch.tensor(seqs[i], device=device) for i in active_indices], batch_first=True, padding_value=v.PAD)
                logits_batch, _ = self.pf(batch_seqs_tensors)
                last_logits = logits_batch[:, -1, :]
                
                masks = []
                for i in active_indices:
                    mask = torch.zeros(v.size(), dtype=torch.bool, device=device)
                    d = depths[i][-1] if depths[i] else c.max_depth
                    if envs[i].open_leaves > 0 and d < c.max_depth: mask[v.split_start : v.split_start + v.num_feat] = True
                    if envs[i].open_leaves > 0: mask[v.split_start + v.num_feat + v.num_th :] = True
                    masks.append(mask)
                
                toks1 = _safe_sample(last_logits, torch.stack(masks), temp)
                needs_threshold, still_active = {}, []
                for i, original_idx in enumerate(active_indices):
                    kind, idx = self.tokenizer.decode_one(toks1[i].item())
                    seqs[original_idx].append(toks1[i].item())
                    envs[original_idx].step((kind, idx))
                    if kind == 'feat':
                        d0 = depths[original_idx].pop()
                        depths[original_idx].extend([d0 + 1, d0 + 1])
                        needs_threshold[len(needs_threshold)] = i
                    else:
                        if depths[original_idx]: depths[original_idx].pop()
                    if not depths[original_idx]: envs[original_idx].done = True
                    if not envs[original_idx].done: still_active.append(original_idx)

                if needs_threshold:
                    sub_batch_indices = [active_indices[i] for i in needs_threshold.values()]
                    sub_batch_seqs = torch.nn.utils.rnn.pad_sequence([torch.tensor(seqs[i], device=device) for i in sub_batch_indices], batch_first=True, padding_value=v.PAD)
                    sub_logits, _ = self.pf(sub_batch_seqs)
                    th_mask = torch.zeros_like(sub_logits[:, -1, :], dtype=torch.bool, device=device)
                    th_mask[:, v.split_start + v.num_feat : v.split_start + v.num_feat + v.num_th] = True
                    toks2 = _safe_sample(sub_logits[:, -1, :], th_mask, temp)
                    for i, original_idx in enumerate(sub_batch_indices):
                        seqs[original_idx].append(toks2[i].item())
                        envs[original_idx].step(self.tokenizer.decode_one(toks2[i].item()))
                active_indices = still_active

        for i in range(num):
            if envs[i].done and envs[i].open_leaves == 0:
                seqs[i].append(v.EOS)
                final_results[i] = (seqs[i], envs[i].get_prior(beta).item(), envs[i].idxs.clone())
        return final_results

    def predict(self, df_test, df_train, use_policy=False, policy_inference_trees=None):
        if not self.tokenizer or (not self.ensemble and not use_policy):
            raise RuntimeError("Fit first.")
        c, device = self.cfg, self.cfg.device
        env_template = TabularEnv(df_train, c.feature_cols, c.target_col, c.n_bins, c.task, device)
        X_te = env_template._featurise(df_test, df_train, c.feature_cols, c.n_bins)
        X_tr, y_tr = env_template.X_full.clone(), env_template.y_full.clone()

        if c.task == "regression":
            test_preds = torch.full((len(df_test),), self.y_mean, device=device, dtype=torch.float32)
            train_preds = torch.full_like(y_tr, self.y_mean, dtype=torch.float32)
        else:  # classification
            if c.n_classes > 2:
                test_preds = torch.zeros((len(df_test), c.n_classes), device=device)
                train_preds = torch.zeros((len(y_tr), c.n_classes), device=device)
            else:  # binary
                test_preds = torch.zeros(len(df_test), device=device)
                train_preds = torch.zeros_like(y_tr, device=device, dtype=torch.float32)

        if not use_policy:
            for trees_in_round in tqdm(self.ensemble, desc="Ensemble Prediction", leave=False):
                if not trees_in_round:
                    continue

                if c.task == "regression":
                    residuals = y_tr - train_preds
                else:
                    if c.n_classes > 2:
                        probs = torch.softmax(train_preds, dim=1)
                        residuals = torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes) - probs
                    else:
                        probs = torch.sigmoid(train_preds)
                        residuals = y_tr - probs

                avg_test, avg_train = torch.zeros_like(test_preds), torch.zeros_like(train_preds)
                for seq in trees_in_round:
                    predictor = get_tree_predictor(seq, X_tr, residuals, self.tokenizer)
                    avg_test += predictor(X_te)
                    avg_train += predictor(X_tr)
                if trees_in_round:
                    avg_test /= len(trees_in_round)
                    avg_train /= len(trees_in_round)
                test_preds += c.boosting_lr * avg_test
                train_preds += c.boosting_lr * avg_train
        else:
            total_trees = policy_inference_trees if policy_inference_trees is not None else c.policy_inference_trees
            num_batches = math.ceil(total_trees / c.num_parallel)

            for i in tqdm(range(num_batches), desc="Policy-based Prediction", leave=False):
                if c.task == "regression":
                    residuals_for_batch = y_tr - train_preds
                else:
                    if c.n_classes > 2:
                        probs = torch.softmax(train_preds, dim=1)
                        residuals_for_batch = torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes) - probs
                    else:
                        probs = torch.sigmoid(train_preds)
                        residuals_for_batch = y_tr - probs
                
                env_template.y = residuals_for_batch.clone()

                trees_in_batch = min(c.num_parallel, total_trees - (i * c.num_parallel))
                if trees_in_batch <= 0:
                    break

                batch_results = self.batched_rollout(
                    [copy.copy(env_template) for _ in range(trees_in_batch)],
                    temp=1.0,
                    residuals=residuals_for_batch,
                    beta=c.beta if c.beta is not None else 0.1
                )
                
                generated_seqs = [res[0] for res in batch_results if res is not None]
                if not generated_seqs:
                    continue

                avg_test_batch = torch.zeros_like(test_preds)
                avg_train_batch = torch.zeros_like(train_preds)

                for seq in generated_seqs:
                    predictor = get_tree_predictor(seq, X_tr, residuals_for_batch, self.tokenizer)
                    avg_test_batch += predictor(X_te)
                    avg_train_batch += predictor(X_tr)
                
                if generated_seqs:
                    avg_test_batch /= len(generated_seqs)
                    avg_train_batch /= len(generated_seqs)

                test_preds += c.boosting_lr * avg_test_batch
                train_preds += c.boosting_lr * avg_train_batch
                
        return test_preds.cpu().numpy()

    def get_params(self, deep=True): return asdict(self.cfg)
    def set_params(self, **params):
        for k, v in params.items(): setattr(self.cfg, k, v)
        return self