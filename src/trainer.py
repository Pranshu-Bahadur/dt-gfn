from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Deque, Dict
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

# Assuming these imports are in your project structure
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

@dataclass
class Config:
    feature_cols: List[str]
    target_col: str = "target"
    task: str = "classification"
    reward_function: str = "bayesian"
    n_classes: Optional[int] = None
    n_bins: int = 255
    binning_strategy: str = "global_uniform"
    device: str = "cuda"
    random_forest: bool = True
    updates: int = 50
    rollouts: int = 60
    batch_size: int = 8192
    max_depth: int = 7
    top_k_trees: int = 10
    boosting_lr: float = 0.1
    redundancy_aware: bool = False
    lstm_hidden: int = 256
    mlp_layers: int = 3
    mlp_width: int = 256
    lr: float = 1e-4
    beta: Optional[float] = None
    prior_scale: float = 0.5
    num_parallel: int = 10
    policy_inference_trees: int = 500

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer: Optional[Tokenizer] = None
        self.ensemble: list[list[int]] = []
        self.boosting_ensemble: list = []
        self.y_mean: float = 0.0
        self.pf: Optional[PolicyPaperMLP] = None
        self.pb: Optional[PolicyPaperMLP] = None
        self.log_z: Optional[torch.Tensor] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.le: Optional[LabelEncoder] = None
        self.tree_log_posteriors: Dict[tuple[int, ...], float] = {}

    def fit(self, df_train: pd.DataFrame) -> "Trainer":
        c = self.cfg
        v = Vocab(len(c.feature_cols), c.n_bins, 1)
        self.tokenizer = Tokenizer(v)
        env_template = TabularEnv(
            df_train, feature_cols=c.feature_cols, target_col=c.target_col,
            n_bins=c.n_bins, task=c.task, binning_strategy=c.binning_strategy, device=c.device
        )
        if c.task == "classification":
            self.le, c.n_classes = env_template.le, env_template.n_classes
        y_true, X_binned = env_template.y_full.clone(), env_template.X_full.clone()
        self.pf = torch.jit.script(PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device))
        self.pb = torch.jit.script(PolicyPaperMLP(v.size(), c.lstm_hidden, c.mlp_layers, c.mlp_width).to(c.device))
        self.log_z = torch.nn.Parameter(torch.tensor(1.0, device=c.device))
        optimizers = [
            torch.optim.AdamW(self.pf.parameters(), lr=c.lr),
            torch.optim.AdamW(self.pb.parameters(), lr=c.lr),
            torch.optim.Adam([self.log_z], lr=c.lr / 10)
        ]
        warmup, tmax = 10, max(1, c.updates - 10)
        schedulers = [
            SequentialLR(opt, schedulers=[LambdaLR(opt, lambda u: min(1.0, u / warmup)), CosineAnnealingLR(opt, T_max=tmax)], milestones=[warmup])
            for opt in optimizers
        ]
        self.replay_buffer = ReplayBuffer(capacity=100000)
        if c.beta is None:
            c.beta = math.log(4) + math.log(len(c.feature_cols)) + math.log(c.n_bins)
            print(f"Using beta derived from the paper's formula: {c.beta:.4f}")
        if c.random_forest:
            self._fit_dt_gfn_random_forest(env_template, y_true, X_binned, optimizers, schedulers)
        else:
            self._fit_boost_gfn(env_template, y_true, X_binned, optimizers, schedulers)
        return self

    def _calculate_reward(self, tok: torch.Tensor, reward_env: TabularEnv) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        c = self.cfg
        if c.reward_function == 'bayesian':
            reward_func = calculate_bayesian_reward if c.task == 'classification' else calculate_bayesian_reward_regression
            R_t = reward_func(tok, self.tokenizer, reward_env, c.beta)
            return R_t, None
        else:
            if c.task == 'classification':
                reward_func = deltaE_split_gain_classification
            else:
                reward_func = deltaE_split_gain_regression
            R_t_per_step = reward_func(tok, self.tokenizer, reward_env)
            R_t = torch.clamp(R_t_per_step.sum(), min=1e-9)
            return R_t, R_t_per_step

    def _update_policy(self, all_tuples: List, reward_env: TabularEnv, optimizers: List) -> Tuple[float, float]:
        c = self.cfg
        tb_loss_acc, fl_loss_acc = 0.0, 0.0
        for opt in optimizers: opt.zero_grad()
        for seq, prior in all_tuples:
            tok = torch.tensor([seq], device=c.device)
            prior_tensor = torch.tensor([prior], device=c.device)
            log_pf, log_pb = self.pf.log_prob(tok), self.pb.log_prob(torch.flip(tok, dims=[1]))
            R_t, R_t_per_step = self._calculate_reward(tok, reward_env)
            if c.reward_function == 'bayesian':
                total_loss = tb_loss(log_pf, log_pb, self.log_z, R_t, prior_tensor)
                tb_loss_acc += total_loss.item()
            else:
                l_tb = tb_loss(log_pf, log_pb, self.log_z, R_t, prior_tensor)
                l_fl = fl_loss(self.pf.log_F(tok), log_pf, log_pb, R_t_per_step)
                total_loss = l_tb + l_fl
                tb_loss_acc += l_tb.item()
                fl_loss_acc += l_fl.item()
            total_loss.backward()
        for opt in optimizers:
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
            opt.step()
        self.replay_buffer.mark_policy_update()
        avg_tb = tb_loss_acc / len(all_tuples) if all_tuples and tb_loss_acc > 0 else 0
        avg_fl = fl_loss_acc / len(all_tuples) if all_tuples and fl_loss_acc > 0 else 0
        return avg_tb, avg_fl

    def _fit_dt_gfn_random_forest(self, env_template, y_true, X_binned, optimizers, schedulers):
        c = self.cfg
        print("--- Starting DT-GFN (Random Forest) Training ---")
        current_ensemble_for_metrics = []
        env_template.y = y_true.clone()
        reward_env = copy.copy(env_template)
        reward_env.reset(len(y_true))
        if c.task == 'classification':
             reward_env.y = torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float)

        for upd in tqdm(range(1, c.updates + 1), desc="Policy Training & Tree Generation"):
            forward_tuples = self._collect_rollouts(env_template, temp=1.0, residuals=y_true, beta=c.beta)
            replay_tuples = self.sample_replay(c.top_k_trees)
            all_tuples_for_policy_update = forward_tuples + replay_tuples
            if not all_tuples_for_policy_update: continue

            current_ensemble_for_metrics = [seq for seq, _ in all_tuples_for_policy_update if seq]
            avg_tb_loss, avg_fl_loss = self._update_policy(all_tuples_for_policy_update, reward_env, optimizers)
            for sch in schedulers: sch.step()

            log_str = f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f} | Step Forest Size: {len(current_ensemble_for_metrics)}"
            if current_ensemble_for_metrics:
                predictors = [get_tree_predictor(seq, X_binned, y_true, self.tokenizer, task=c.task, n_classes=c.n_classes) for seq in current_ensemble_for_metrics]
                all_preds = torch.stack([p(X_binned) for p in predictors])
                train_preds = all_preds.mean(dim=0)
                if c.task == "classification":
                    acc = (train_preds.argmax(1) == y_true).float().mean().item()
                    log_str += f" | Train Acc: {acc:.4f}"
                else:
                    if train_preds.std() > 0 and y_true.std() > 0:
                        corr = torch.corrcoef(torch.stack([train_preds.squeeze(), y_true.squeeze()]))[0, 1].item()
                        log_str += f" | Train Corr: {corr:.4f}"
            tqdm.write(log_str)

        print(f"\n--- Training Finished. Computing final posterior weights for {len(current_ensemble_for_metrics)} trees. ---")
        self.ensemble = current_ensemble_for_metrics
        self.tree_log_posteriors.clear()
        for seq in tqdm(self.ensemble, desc="Calculating Posteriors", leave=False):
            if tuple(seq) not in self.tree_log_posteriors:
                 with torch.no_grad():
                    tok_tensor = torch.tensor([seq], device=c.device)
                    log_posterior = self._calculate_reward(tok_tensor, reward_env)[0].item()
                    self.tree_log_posteriors[tuple(seq)] = log_posterior
        print(f"--- RF Finalization Complete. Forest size: {len(self.ensemble)} trees. ---")

    def _fit_boost_gfn(self, env_template, y_true, X_binned, optimizers, schedulers):
        c = self.cfg
        print("--- Starting Boost-GFN Training ---")

        if c.task == "classification":
            class_counts = torch.bincount(y_true, minlength=c.n_classes).float()
            class_probs = class_counts / class_counts.sum()
            initial_logits = torch.log(class_probs + 1e-9)
            base_pred = initial_logits.unsqueeze(0).repeat(len(y_true), 1)
            reward_env_for_sorting = copy.copy(env_template)
            reward_env_for_sorting.y = torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float)
        else: # Regression
            self.y_mean = y_true.mean().item()
            base_pred = torch.full_like(y_true, self.y_mean, dtype=torch.float32)
            reward_env_for_sorting = copy.copy(env_template)
            reward_env_for_sorting.y = y_true.clone()

        # Main boosting loop
        for upd in tqdm(range(1, c.updates + 1), desc="Boost Updates"):
            # --- 1. Generate a batch of candidate trees ---
            if c.task == "classification":
                probs = torch.softmax(base_pred, dim=1)
                residuals = torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float) - probs
            else:
                residuals = y_true - base_pred
            
            env_template.y = residuals.clone()
            # Generate one batch of rollouts for this update step
            candidate_seqs = [r[0] for r in self._collect_rollouts(env_template, 1.0, residuals, c.beta) if r]
            if not candidate_seqs: continue

            # --- 2. Calculate Bayesian reward for each candidate and sort them ---
            candidate_rewards = []
            with torch.no_grad():
                for seq in candidate_seqs:
                    tok_tensor = torch.tensor([seq], device=c.device)
                    reward = self._calculate_reward(tok_tensor, reward_env_for_sorting)[0].item()
                    candidate_rewards.append((reward, seq))
            
            candidate_rewards.sort(key=lambda x: x[0], reverse=True)
            
            # --- 3. Sequentially add the *sorted* trees to the ensemble ---
            for _, seq in candidate_rewards:
                if c.task == "classification":
                    probs = torch.softmax(base_pred, dim=1)
                    current_residuals = torch.nn.functional.one_hot(y_true, num_classes=c.n_classes).to(torch.float) - probs
                else:
                    current_residuals = y_true - base_pred
                
                predictor = get_tree_predictor(seq, X_binned, current_residuals, self.tokenizer, task=c.task, n_classes=c.n_classes)
                base_pred += c.boosting_lr * predictor(X_binned)
                self.boosting_ensemble.append(predictor)

            # --- 4. Update the GFlowNet policy ---
            all_tuples_for_policy_update = self.sample_replay(c.top_k_trees) + [(seq, 0.0) for _, seq in candidate_rewards]
            policy_update_env = env_template
            policy_update_env.reset(len(y_true))
            avg_tb_loss, avg_fl_loss = self._update_policy(all_tuples_for_policy_update, policy_update_env, optimizers)
            for sch in schedulers: sch.step()

            # --- Logging ---
            log_str = f"Update {upd}/{c.updates} | TB: {avg_tb_loss:.4f} | FL: {avg_fl_loss:.4f}"
            if c.task == "classification":
                acc = (base_pred.argmax(1) == y_true).float().mean().item()
                tqdm.write(f"{log_str} | Train Acc: {acc:.4f}")
            else:
                if base_pred.std() > 0 and y_true.std() > 0:
                    corr = torch.corrcoef(torch.stack([base_pred.squeeze(), y_true.squeeze()]))[0, 1].item()
                    tqdm.write(f"{log_str} | Train Corr: {corr:+.4f}")

    def _collect_rollouts(self, env_template, temp, residuals, beta):
        forward_tuples = []
        rollouts_done = 0
        ras_counts = {} if self.cfg.redundancy_aware else None
        with tqdm(total=self.cfg.rollouts, desc="Rollouts", leave=False) as pbar:
            while rollouts_done < self.cfg.rollouts:
                if ras_counts is not None: ras_counts.clear()
                batch_size = min(self.cfg.num_parallel, self.cfg.rollouts - rollouts_done)
                envs = [copy.copy(env_template) for _ in range(batch_size)]
                batch_results = self.batched_rollout(envs, temp, residuals, beta, ras_counts)
                for result in batch_results:
                    if result:
                        seq, prior, idxs = result
                        self.replay_buffer.add(0.0, seq, prior, idxs)
                        forward_tuples.append((seq, prior))
                rollouts_done += batch_size
                pbar.update(batch_size)
        return forward_tuples

    def sample_replay(self, k: int, REFRESH_INTERVAL: int = 5) -> list[tuple[list[int], float]]:
        buf = self.replay_buffer
        if not buf or not buf.data: return []
        stale_indices = [i for i, entry in enumerate(buf.data) if entry[4] is None or buf.step - entry[5] >= REFRESH_INTERVAL]
        if stale_indices:
            stale_seqs = [buf.data[i][1] for i in stale_indices]
            with torch.no_grad():
                flipped_seqs = [torch.tensor(seq, device=self.cfg.device).flip(dims=[0]) for seq in stale_seqs]
                padded_bwd_seqs = torch.nn.utils.rnn.pad_sequence(flipped_seqs, batch_first=True, padding_value=self.tokenizer.v.PAD)
                log_probs = self.pb.log_prob(padded_bwd_seqs)
                mask = (padded_bwd_seqs != self.tokenizer.v.PAD).float()
                if log_probs.shape[1] != mask.shape[1]: mask = mask[:, :-1]
                new_weights = (log_probs * mask).sum(dim=1).exp()
                for i, weight in zip(stale_indices, new_weights):
                    r, t, p, idxs, _, _ = buf.data[i]
                    buf.data[i] = (r, t, p, idxs, weight.item(), buf.step)
        entries = list(buf.data)
        weights = np.array([e[4] for e in entries], dtype=np.float32)
        total_weight = weights.sum()
        k = min(k, len(entries))
        if total_weight > 0:
            probabilities = weights / total_weight
            sampled_indices = np.random.choice(len(entries), size=k, p=probabilities, replace=True)
        else:
            sampled_indices = np.random.randint(0, len(entries), size=k)
        return [(entries[i][1], entries[i][2]) for i in sampled_indices]

    def batched_rollout(self, envs, temp, residuals, beta, ras_counts: Optional[dict] = None):
        c, v, device = self.cfg, self.tokenizer.v, self.cfg.device
        num = len(envs)
        END_TOKEN = 2
        for env in envs:
            env.y = residuals
            env.reset(c.batch_size)
        seqs, depths = [[v.BOS] for _ in range(num)], [deque([0]) for _ in range(num)]
        active_indices, final_results = list(range(num)), [None] * num
        with torch.no_grad():
            while active_indices:
                batch_seqs_tensors = torch.nn.utils.rnn.pad_sequence([torch.tensor(seqs[i], device=device) for i in active_indices], batch_first=True, padding_value=v.PAD)
                logits_batch, _ = self.pf(batch_seqs_tensors)
                last_logits = logits_batch[:, -1, :]
                if ras_counts is not None:
                    for i, original_idx in enumerate(active_indices):
                        path_tuple = tuple(seqs[original_idx])
                        if path_tuple in ras_counts: last_logits[i, :] -= ras_counts[path_tuple] * 1e9
                masks = torch.zeros((len(active_indices), v.size()), dtype=torch.bool, device=device)
                for i, original_idx in enumerate(active_indices):
                    d = depths[original_idx][-1] if depths[original_idx] else c.max_depth
                    if envs[original_idx].open_leaves > 0 and d < c.max_depth: masks[i, v.split_start : v.split_start + v.num_feat] = True
                    if envs[original_idx].open_leaves > 0: masks[i, v.split_start + v.num_feat + v.num_th :] = True
                toks1 = _safe_sample(last_logits, masks, temp)
                needs_threshold, still_active = {}, []
                for i, original_idx in enumerate(active_indices):
                    token = toks1[i].item()
                    seqs[original_idx].append(token)
                    if ras_counts is not None:
                        path_tuple = tuple(seqs[original_idx])
                        ras_counts[path_tuple] = ras_counts.get(path_tuple, 0) + 1
                    if token == END_TOKEN:
                        envs[original_idx].done = True
                        continue
                    kind, idx = self.tokenizer.decode_one(token)
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
                        token = toks2[i].item()
                        seqs[original_idx].append(token)
                        if token == END_TOKEN:
                            envs[original_idx].done = True
                            continue
                        envs[original_idx].step(self.tokenizer.decode_one(token))
                active_indices = still_active
        for i in range(num):
            if envs[i].done and envs[i].open_leaves == 0:
                if seqs[i][-1] != v.EOS: seqs[i].append(v.EOS)
                final_results[i] = (seqs[i], envs[i].get_prior(beta).item(), envs[i].idxs.clone())
        return final_results

    def predict(self, df_test, df_train, use_policy=False, policy_inference_trees=None, sample_leaf_params=False):
        ensemble_exists = self.ensemble or self.boosting_ensemble
        if not self.tokenizer or (not ensemble_exists and not use_policy):
            raise RuntimeError("Fit the model before predicting.")
        c = self.cfg
        env_template = TabularEnv(df_train, c.feature_cols, c.target_col, c.n_bins, c.task, binning_strategy=c.binning_strategy, device=c.device)
        X_te = env_template._featurise(df_test, df_train, c.feature_cols, c.n_bins)
        X_tr, y_tr = env_template.X_full.clone(), env_template.y_full.clone()
        if c.random_forest:
            preds = self._predict_random_forest(X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees, sample_leaf_params)
        else:
            preds = self._predict_boosting(X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees)
        return preds.cpu().numpy()


    def _predict_random_forest(self, X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees, sample_leaf_params):
        c = self.cfg
        trees_to_use = []
        if use_policy:
            total_trees = policy_inference_trees if policy_inference_trees is not None else c.policy_inference_trees
            num_batches = math.ceil(total_trees / c.num_parallel)
            env_template.y = y_tr.clone()
            for _ in tqdm(range(num_batches), desc="Policy-based Tree Generation", leave=False):
                trees_in_batch = min(c.num_parallel, total_trees - len(trees_to_use))
                if trees_in_batch <= 0: break
                ras_counts = {} if c.redundancy_aware else None
                batch_results = self.batched_rollout([copy.copy(env_template) for _ in range(trees_in_batch)], temp=1.0, residuals=y_tr, beta=c.beta, ras_counts=ras_counts)
                trees_to_use.extend([res[0] for res in batch_results if res])

            log_posteriors = []
            reward_env = copy.copy(env_template)
            if c.task == 'classification':
                reward_env.y = torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float)
            else:
                reward_env.y = y_tr.clone()
            for seq in tqdm(trees_to_use, desc="Calculating Posteriors", leave=False):
                with torch.no_grad():
                    log_posteriors.append(self._calculate_reward(torch.tensor([seq], device=c.device), reward_env)[0])
            weights = torch.softmax(torch.tensor(log_posteriors, device=c.device), dim=0)

        else:
            trees_to_use = self.ensemble
            if not trees_to_use: raise RuntimeError("The fitted forest is empty.")
            log_posteriors = torch.tensor([self.tree_log_posteriors[tuple(seq)] for seq in trees_to_use], device=c.device)
            weights = torch.softmax(log_posteriors, dim=0)

        sum_preds = torch.zeros((X_te.shape[0], c.n_classes), device=c.device) if c.task == "classification" else torch.zeros(X_te.shape[0], device=c.device, dtype=torch.float32)
        for i, seq in enumerate(tqdm(trees_to_use, desc="RF Bayesian Prediction", leave=False)):
            predictor = get_tree_predictor(
                seq, X_tr, y_tr, self.tokenizer,
                task=c.task, n_classes=c.n_classes, alpha=c.prior_scale, sample_leaf_params=sample_leaf_params
            )
            sum_preds += weights[i] * predictor(X_te)
        return sum_preds

    def _predict_boosting(self, X_te, X_tr, y_tr, env_template, use_policy, policy_inference_trees):
        c = self.cfg
        # --- Setup initial predictions ---
        if c.task == "classification":
            class_counts = torch.bincount(y_tr, minlength=c.n_classes).float()
            class_probs = class_counts / class_counts.sum()
            initial_logits = torch.log(class_probs + 1e-9)
            base_pred = initial_logits.unsqueeze(0).repeat(len(X_te), 1)
            train_base_pred = initial_logits.unsqueeze(0).repeat(len(y_tr), 1)
        else: # Regression
            base_pred = torch.full((len(X_te),), self.y_mean, device=c.device, dtype=torch.float32)
            train_base_pred = torch.full_like(y_tr, self.y_mean, dtype=torch.float32)

        if use_policy:
            print("--- Generating Boosting Ensemble with Policy (Batch-Sort-Sequential-Apply) ---")
            total_trees = policy_inference_trees if policy_inference_trees is not None else c.updates * c.rollouts
            
            # --- Setup reward env for sorting ---
            reward_env_for_sorting = copy.copy(env_template)
            if c.task == 'classification':
                reward_env_for_sorting.y = torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float)
            else:
                reward_env_for_sorting.y = y_tr.clone()

            # --- 1. Generate a large pool of candidates ---
            if c.task == "classification":
                probs = torch.softmax(train_base_pred, dim=1)
                initial_residuals = torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float) - probs
            else:
                initial_residuals = y_tr - train_base_pred
            env_template.y = initial_residuals.clone()

            num_rollout_batches = math.ceil(total_trees / c.rollouts)
            candidate_seqs = []
            for _ in tqdm(range(num_rollout_batches), desc="Generating Candidate Pool", leave=False):
                candidate_seqs.extend([r[0] for r in self._collect_rollouts(env_template, 1.0, env_template.y, c.beta) if r])

            # --- 2. Calculate rewards and sort the entire pool ---
            candidate_rewards = []
            with torch.no_grad():
                for seq in tqdm(candidate_seqs, desc="Ranking Candidate Pool", leave=False):
                    reward = self._calculate_reward(torch.tensor([seq], device=c.device), reward_env_for_sorting)[0].item()
                    candidate_rewards.append((reward, seq))
            
            candidate_rewards.sort(key=lambda x: x[0], reverse=True)

            # --- 3. Sequentially apply the sorted trees ---
            if candidate_rewards:
              for _, best_seq in tqdm(candidate_rewards, desc="Applying Sorted Ensemble", leave=False):
                # Always calculate residuals based on the current train predictions
                if c.task == "classification":
                    probs = torch.softmax(train_base_pred, dim=1)
                    current_residuals = torch.nn.functional.one_hot(y_tr, num_classes=c.n_classes).to(torch.float) - probs
                else:
                    current_residuals = y_tr - train_base_pred
                
                # Fit the chosen tree on the current residuals
                predictor = get_tree_predictor(best_seq, X_tr, current_residuals, self.tokenizer, task=c.task, n_classes=c.n_classes)
                
                # Update both train and test predictions
                train_base_pred += c.boosting_lr * predictor(X_tr)
                base_pred += c.boosting_lr * predictor(X_te)
        else:
            # Use the pre-trained boosting ensemble
            for predictor_func in tqdm(self.boosting_ensemble, desc="Ensemble Prediction", leave=False):
                # In this case, the predictors are already fitted and ready to be applied
                base_pred += c.boosting_lr * predictor_func(X_te)
        
        return base_pred

    def get_params(self, deep=True): return asdict(self.cfg)
    def set_params(self, **params):
        for k, v in params.items(): setattr(self.cfg, k, v)
        return self