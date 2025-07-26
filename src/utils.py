from __future__ import annotations

import random
import math
from collections import deque
from typing import Callable, List, Optional, Tuple, Deque

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# --- Loss Functions ---

@torch.jit.script
def tb_loss(log_pf: torch.Tensor, log_pb: torch.Tensor, log_z: torch.Tensor, R: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Trajectory Balance (TB) loss, batched.
    """
    loss = (log_z + log_pf.sum(1) - (torch.log(R) + prior + log_pb.sum(1)))**2
    return loss.mean()

@torch.jit.script
def fl_loss(logF: torch.Tensor, log_pf: torch.Tensor, log_pb: torch.Tensor, dR: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Flow-Matching (FM) or Detailed Balance loss.
    """
    loss = (logF[:, :-1] + log_pf - (logF[:, 1:] + log_pb + dR))**2
    return loss.mean()

# --- Tree & Reward Utilities ---

def deltaE_split_gain_regression(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """
    Computes the reward (ΔR) as the MSE reduction (split gain) for each split
    action in a trajectory. Used for REGRESSION tasks.
    """
    y: torch.Tensor = env.y[env.idxs]
    N: int = y.numel()
    dR: torch.Tensor = torch.zeros(tokens.shape[1] - 1, device=y.device)

    full_mse = (y.mul(y).mean() - y.mean()**2).item() if N > 1 else 0.0
    stack_rows: Deque[torch.Tensor] = deque([torch.arange(N, device=y.device)])
    stack_mse: Deque[float] = deque([full_mse])

    action_sequence: List[int] = tokens[0, 1:-1].tolist()
    it = iter(tok.decode(action_sequence))
    token_idx = 0

    for kind, idx in it:
        if kind == "feat":
            try:
                _, th = next(it)
            except StopIteration:
                break

            if not stack_rows: continue
            parent_rows = stack_rows.pop()
            parent_mse = stack_mse.pop()

            fv = env.X_full[env.idxs[parent_rows], idx]
            mask = fv <= th
            L_rows, R_rows = parent_rows[mask], parent_rows[~mask]

            def mse(rows: torch.Tensor) -> float:
                if rows.numel() < 2: return 0.0
                yy = y[rows]
                return ((yy * yy).mean() - yy.mean()**2).item()

            mseL, mseR = mse(L_rows), mse(R_rows)
            stack_rows.extend([R_rows, L_rows])
            stack_mse.extend([mseR,  mseL])

            wL, wR = L_rows.numel(), R_rows.numel()
            parent_N = wL + wR
            if parent_N > 0:
                gain = parent_mse - (wL / parent_N * mseL + wR / parent_N * mseR)
                dR[token_idx] = gain
            token_idx += 2
        else:
            if stack_rows:
                stack_rows.pop()
                stack_mse.pop()
            token_idx += 1
    return dR.unsqueeze(0)

def deltaE_split_gain_classification(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """
    Computes the reward (ΔR) as the Gini impurity reduction for each split action
    in a trajectory. Used for CLASSIFICATION tasks.
    """
    y: torch.Tensor = env.y_full[env.idxs]
    N: int = y.numel()
    dR: torch.Tensor = torch.zeros(tokens.shape[1] - 1, device=y.device)

    def gini_impurity(rows: torch.Tensor) -> float:
        if rows.numel() == 0: return 0.0
        labels = y[rows].long()
        n_labels = len(labels)
        if n_labels == 0: return 0.0
        
        clamped_labels = torch.clamp(labels, 0, env.n_classes - 1)
        counts = torch.bincount(clamped_labels, minlength=env.n_classes)
        
        probs = counts.float() / n_labels
        return 1 - torch.sum(probs**2).item()

    full_gini = gini_impurity(torch.arange(N, device=y.device))
    stack_rows: Deque[torch.Tensor] = deque([torch.arange(N, device=y.device)])
    stack_metric: Deque[float] = deque([full_gini])

    action_sequence: List[int] = tokens[0, 1:-1].tolist()
    it = iter(tok.decode(action_sequence))
    token_idx = 0

    for kind, idx in it:
        if kind == "feat":
            try:
                _, th = next(it)
            except StopIteration:
                break

            if not stack_rows: continue
            parent_rows = stack_rows.pop()
            parent_metric = stack_metric.pop()

            fv = env.X_full[env.idxs[parent_rows], idx]
            mask = fv <= th
            L_rows, R_rows = parent_rows[mask], parent_rows[~mask]

            metricL, metricR = gini_impurity(L_rows), gini_impurity(R_rows)
            stack_rows.extend([R_rows, L_rows])
            stack_metric.extend([metricR,  metricL])

            wL, wR = L_rows.numel(), R_rows.numel()
            parent_N = wL + wR
            if parent_N > 0:
                gain = parent_metric - (wL / parent_N * metricL + wR / parent_N * metricR)
                dR[token_idx] = gain
            token_idx += 2
        else:
            if stack_rows:
                stack_rows.pop()
                stack_metric.pop()
            token_idx += 1
    return dR.unsqueeze(0)

def calculate_bayesian_reward(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv", beta: float) -> torch.Tensor:
    """
    Computes reward for a completed tree based on Bayesian marginal likelihood, as per the paper[cite: 324].
    Used for CLASSIFICATION tasks with the 'bayesian' reward function.
    """
    alpha = 0.1 # Dirichlet prior, as specified in the paper's hyperparameter table [cite: 895]
    alphas = torch.full((env.n_classes,), alpha, device=env.device)

    decoded_actions = tok.decode(tokens[0, 1:-1].tolist())
    
    leaves_indices = []
    n_decision_nodes = 0
    
    tree_nodes = {0: {'indices': env.idxs, 'children': []}}
    node_counter = 0

    action_iter = iter(decoded_actions)
    
    while True:
        try:
            leaf_node_id = -1
            for nid, node in sorted(tree_nodes.items()):
                if not node['children']:
                    leaf_node_id = nid
                    break
            
            if leaf_node_id == -1: break

            kind, val = next(action_iter)
            parent_indices = tree_nodes[leaf_node_id]['indices']

            if kind == 'feat':
                if len(parent_indices) == 0:
                    next(action_iter)
                    continue

                n_decision_nodes += 1
                _, threshold = next(action_iter)
                
                fv = env.X_full[parent_indices, val]
                mask = fv <= threshold
                
                left_indices = parent_indices[mask]
                right_indices = parent_indices[~mask]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    tree_nodes[leaf_node_id]['children'] = [-1, -1]
                    continue

                node_counter += 1; left_child_id = node_counter
                tree_nodes[left_child_id] = {'indices': left_indices, 'children': []}
                
                node_counter += 1; right_child_id = node_counter
                tree_nodes[right_child_id] = {'indices': right_indices, 'children': []}
                
                tree_nodes[leaf_node_id]['children'] = [left_child_id, right_child_id]
            else:
                tree_nodes[leaf_node_id]['children'] = [-1, -1]
                
        except StopIteration:
            break
            
    for nid, node in tree_nodes.items():
        if not node['children']:
            leaves_indices.append(node['indices'])

    # Calculate Log Marginal Likelihood from Proposition 3.2 
    log_likelihood = 0.0
    log_gamma_alpha_sum = torch.lgamma(alphas.sum())
    log_gamma_alpha_prod = torch.lgamma(alphas).sum()
    log_dirichlet_norm = len(leaves_indices) * (log_gamma_alpha_sum - log_gamma_alpha_prod)
    log_likelihood += log_dirichlet_norm

    for leaf_indices in leaves_indices:
        if len(leaf_indices) == 0: continue
        leaf_labels = env.y_full[leaf_indices]
        n_l_c = torch.bincount(leaf_labels, minlength=env.n_classes).float()
        n_l = n_l_c.sum()
        
        log_numerator = torch.lgamma(n_l_c + alphas).sum()
        log_denominator = torch.lgamma(n_l + alphas.sum())
        log_likelihood += log_numerator - log_denominator
        
    # Calculate Structure Prior from Section 4.2 
    log_prior = -beta * n_decision_nodes
    log_reward = log_likelihood + log_prior
    reward = torch.exp(log_reward) + 1e-9
    
    return reward.unsqueeze(0)


def get_tree_predictor(traj: List[int], X_binned: torch.Tensor, y_target: torch.Tensor, tok: "Tokenizer") -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Decodes a token trajectory into a callable tree predictor function.
    """
    path_iter = iter(tok.decode(traj[1:-1]))
    def build_recursive() -> Optional[dict]:
        try:
            kind, idx = next(path_iter)
        except StopIteration:
            return None
        if kind == 'feat':
            return {'type': 'split', 'f': idx, 't': next(path_iter)[1], 'L': build_recursive(), 'R': build_recursive()}
        return {'type': 'leaf', 'value': 0}

    tree_structure = build_recursive()
    if tree_structure is None:
        if y_target.ndim > 1:
            return lambda X: torch.zeros(X.size(0), y_target.size(1), device=X.device)
        return lambda X: torch.zeros(X.size(0), device=X.device)

    leaf_values = {}
    q = deque([(tree_structure, torch.arange(X_binned.size(0), device=X_binned.device))])
    leaf_idx_counter = 0
    while q:
        node, indices = q.popleft()
        if not node or not indices.numel(): continue

        if node['type'] == 'split':
            mask = X_binned[indices, node['f']] <= node['t']
            if node.get('L'): q.append((node['L'], indices[mask]))
            if node.get('R'): q.append((node['R'], indices[~mask]))
        else:
            node['leaf_idx'] = leaf_idx_counter
            if y_target.ndim > 1:
                value = y_target[indices].mean(dim=0) if indices.numel() > 0 else torch.zeros(y_target.size(1), device=y_target.device)
            else:
                value = y_target[indices].mean().item() if indices.numel() > 0 else 0.0
            leaf_values[leaf_idx_counter] = value
            leaf_idx_counter += 1

    def predict(X_test: torch.Tensor) -> torch.Tensor:
        if y_target.ndim > 1:
             preds = torch.zeros(X_test.size(0), y_target.size(1), device=X_test.device)
        else:
             preds = torch.zeros(X_test.size(0), device=X_test.device)

        q = deque([(tree_structure, torch.arange(X_test.size(0), device=X_test.device))])
        while q:
            node, indices = q.popleft()
            if not node or not indices.numel(): continue
            if node['type'] == 'leaf':
                preds[indices] = leaf_values.get(node.get('leaf_idx'), 0.0)
            else:
                mask = X_test[indices, node['f']] <= node['t']
                if node.get('L'): q.append((node['L'], indices[mask]))
                if node.get('R'): q.append((node['R'], indices[~mask]))
        return preds
    return predict

# --- Sampling & Buffer ---
class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.data: List[Tuple[float, List[int], float, torch.Tensor]] = []

    def add(self, r: float, t: List[int], p: float, idxs: torch.Tensor):
        self.data.append((r, t, p, idxs))
        self.data.sort(key=lambda x: x[0], reverse=True)
        if len(self.data) > self.capacity:
            self.data.pop()

    def sample(self, k: int) -> list:
        return random.sample(self.data, min(k, len(self.data)))

END_TOKEN = 2
EPS = 1e-9

def _safe_sample(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature > EPS:
        logits = logits / temperature
    logits = logits.masked_fill(~mask, -float("inf"))
    all_masked = (~mask).all(dim=-1)
    if all_masked.any():
        logits[all_masked, END_TOKEN] = 0
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(1)

def create_gain_bias(df_train: pd.DataFrame, feats: List[str], target: str, tok: "Tokenizer", bins: int, prior_scale: float = 0.5) -> torch.Tensor:
    df_sample = df_train.sample(n=min(len(df_train), 200_000), random_state=42)
    X_binned_list = []
    for f in feats:
        s = df_sample[f].replace([np.inf,-np.inf],np.nan).fillna(df_sample[f].median()).values
        qs = np.linspace(0,1,bins+1)
        edges = np.unique(np.quantile(s,qs))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        X_binned_list.append(np.searchsorted(edges,s,side="right")-1)
    X_binned = np.stack(X_binned_list, 1)
    y_train = df_sample[target].values

    lgb_train = lgb.Dataset(X_binned, y_train, feature_name=feats, free_raw_data=False)
    params = { 'objective': 'regression_l1', 'metric': 'l1', 'n_estimators': 100, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'num_leaves': 1024, 'max_depth': 10, 'verbose': -1, 'n_jobs': -1 }
    gbm = lgb.train(params, lgb_train)
    tree_info = gbm.dump_model()["tree_info"]
    bias = torch.zeros(tok.v.size(), dtype=torch.float32)

    def parse_node(node: dict):
        nonlocal bias
        if "split_gain" in node and node["split_gain"] > 0:
            gain = node["split_gain"]
            f_idx = node["split_feature"]
            tok_id_feat = tok._feat(f_idx)
            if tok_id_feat < tok.v.size():
                bias[tok_id_feat] += gain
            try:
                bin_idx = min(int(node["threshold"]), bins - 1)
                tok_id_th = tok._th(bin_idx)
                if tok_id_th < tok.v.size():
                    bias[tok_id_th] += gain
            except (ValueError, AttributeError):
                pass
        if "left_child" in node: parse_node(node["left_child"])
        if "right_child" in node: parse_node(node["right_child"])
    for tree in tree_info:
        parse_node(tree['tree_structure'])

    bias_std = bias.std()
    if bias_std > 1e-6:
        bias /= bias_std
    bias *= (prior_scale / 2.0)
    return bias.to(dtype=torch.float32)