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
    rhs = torch.log(R) + prior + log_pb.sum(1)
    diff = log_z + log_pf.sum(1) - rhs
    return (diff * diff).mean()

@torch.jit.script
def fl_loss(logF: torch.Tensor, log_pf: torch.Tensor, log_pb: torch.Tensor, dR: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Flow-Matching (FM) or Detailed Balance loss.
    """
    loss = (logF[:, :-1] + log_pf - (logF[:, 1:] + log_pb + dR))**2
    return loss.mean()

# --- Tree & Reward Utilities ---

def _traverse_and_get_leaves(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> Tuple[List[torch.Tensor], int]:
    """Helper function to robustly build a tree and return leaf indices and the number of decision nodes."""
    decoded_actions = tok.decode(tokens[0, 1:-1].tolist())
    
    tree_nodes = {0: {'indices': env.idxs, 'children': []}}
    node_counter = 0
    n_decision_nodes = 0
    action_iter = iter(decoded_actions)
    
    while True:
        try:
            leaf_node_id = -1
            for nid, node in sorted(tree_nodes.items()):
                if not node['children']:
                    leaf_node_id = nid
                    break
            
            if leaf_node_id == -1:
                break

            kind, val = next(action_iter)
            
            if kind == 'feat':
                n_decision_nodes += 1
                _, threshold = next(action_iter)
                parent_indices = tree_nodes[leaf_node_id]['indices']

                if len(parent_indices) == 0:
                    tree_nodes[leaf_node_id]['children'] = [-1, -1]
                    continue
                
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
    
    leaves_indices = [node['indices'] for node in tree_nodes.values() if not node['children']]
    return leaves_indices, n_decision_nodes

def calculate_bayesian_reward(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv", beta: float) -> torch.Tensor:
    """Computes reward for a completed tree based on Bayesian marginal likelihood (for CLASSIFICATION)."""
    leaves_indices, n_decision_nodes = _traverse_and_get_leaves(tokens, tok, env)
    
    alpha = 0.1 
    alphas = torch.full((env.n_classes,), alpha, device=env.device)
    
    log_likelihood = torch.tensor(0.0, device=env.device)
    log_gamma_alpha_sum = torch.lgamma(alphas.sum())
    log_gamma_alpha_prod = torch.lgamma(alphas).sum()
    
    num_leaves = len(leaves_indices)
    log_dirichlet_norm = num_leaves * (log_gamma_alpha_sum - log_gamma_alpha_prod)
    log_likelihood += log_dirichlet_norm

    for leaf_indices in leaves_indices:
        if len(leaf_indices) == 0: continue
        leaf_labels = env.y_full[leaf_indices]
        n_l_c = torch.bincount(leaf_labels, minlength=env.n_classes).float()
        n_l = n_l_c.sum()
        
        log_numerator = torch.lgamma(n_l_c + alphas).sum()
        log_denominator = torch.lgamma(n_l + alphas.sum())
        log_likelihood += log_numerator - log_denominator
        
    log_reward = (log_likelihood)/ float(env.idxs.numel() or 1)
    reward = torch.exp(log_reward).clamp(min=1e-9)
    
    return reward.unsqueeze(0)

def calculate_bayesian_reward_regression(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv", beta: float) -> torch.Tensor:
    """Computes reward for a completed tree based on Bayesian marginal likelihood (for REGRESSION)."""
    leaves_indices, n_decision_nodes = _traverse_and_get_leaves(tokens, tok, env)
    
    mu0 = 0.0
    kappa0 = 1.0
    a0 = torch.tensor(1.0, device=env.device)
    b0 = torch.tensor(1.0, device=env.device)
    
    log_marginal_likelihood = torch.tensor(0.0, device=env.device)
    
    for leaf_indices in leaves_indices:
        n_l = len(leaf_indices)
        if n_l == 0: continue
        
        y_leaf = env.y_full[leaf_indices]
        y_bar = y_leaf.mean(dim=0, keepdim=True)
        sse = ((y_leaf - y_bar)**2).sum()

        kappa_n = kappa0 + n_l
        beta_n = b0 + 0.5 * sse + (kappa0 * n_l) * (y_bar - mu0)**2 / (2 * kappa_n)

        log_ml_leaf = (
            torch.lgamma(a0 + n_l / 2) - torch.lgamma(a0) +
            a0 * torch.log(b0) - (a0 + n_l / 2) * torch.log(beta_n) +
            0.5 * (math.log(kappa0) - math.log(kappa_n)) -
            (n_l / 2) * math.log(2 * math.pi)
        )
        log_marginal_likelihood += log_ml_leaf.sum()

    log_reward = (log_marginal_likelihood) / float(env.idxs.numel() or 1)
    reward = torch.exp(log_reward).clamp(min=1e-9)
    
    return reward.unsqueeze(0)


def deltaE_split_gain_regression(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    y: torch.Tensor = env.y[env.idxs]
    N: int = y.shape[0] 
    dR: torch.Tensor = torch.zeros(tokens.shape[1] - 1, device=y.device)

    def mse(rows: torch.Tensor) -> float:
        if rows.numel() < 2: return 0.0
        yy = y[rows]
        return ((yy.float()**2).mean()).item()

    full_mse = mse(torch.arange(N, device=y.device))
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

            mseL, mseR = mse(L_rows), mse(R_rows)
            stack_rows.extend([R_rows, L_rows])
            stack_mse.extend([mseR,  mseL])

            wL, wR = L_rows.numel(), R_rows.numel()
            parent_N = wL + wR
            if parent_N > 0:
                gain = parent_mse - (wL / parent_N * mseL + wR / parent_N * mseR)
                dR[token_idx] = gain / (float(env.idxs.numel() or 1))
            token_idx += 2
        else:
            if stack_rows:
                stack_rows.pop()
                stack_mse.pop()
            token_idx += 1
    return dR.unsqueeze(0)

def deltaE_split_gain_classification(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
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
                dR[token_idx] = gain / (float(env.idxs.numel() or 1))
            token_idx += 2
        else:
            if stack_rows:
                stack_rows.pop()
                stack_metric.pop()
            token_idx += 1
    return dR.unsqueeze(0)

def get_tree_predictor(traj: List[int], X_binned: torch.Tensor, y_target: torch.Tensor, tok: "Tokenizer") -> Callable[[torch.Tensor], torch.Tensor]:
    # Detach from grad and cache device/dtype info
    y_target = y_target.detach().to(dtype=torch.float32)
    device = X_binned.device
    all_idx = torch.arange(X_binned.size(0), device=device)

    # -------- build tree --------------------------------------------------
    path_iter = iter(tok.decode(traj[1:-1]))
    def build():
        try:
            kind, idx = next(path_iter)
            if kind == 'feat':
                return {
                    'type': 'split', 'f': idx, 't': next(path_iter)[1],
                    'L': build(), 'R': build()
                }
            return {'type': 'leaf'}
        except StopIteration:
            return None # Handle malformed or truncated trajectories

    tree_root = build()
    # If the trajectory is empty or malformed, return a predictor that always predicts zeros
    if tree_root is None:
        return lambda X: torch.zeros(X.size(0), *(y_target.shape[1:]), device=X.device, dtype=y_target.dtype)

    # -------- fit leaves ---------------------------------------------------
    leaf_val = {}
    q = deque([(tree_root, all_idx)])
    # Create a zero vector with the correct dtype and device for default values
    zero_vec = torch.zeros_like(y_target[0])

    while q:
        node, idxs = q.popleft()
        
        # Handle cases where a branch might be missing after a bad split
        if node is None:
            continue

        if not idxs.numel():
            if 'leaf' not in node: # Assign a leaf_id if it doesn't have one
                node['leaf'] = len(leaf_val)
                leaf_val[node['leaf']] = zero_vec
            continue

        if node.get('type') == 'split':
            # Ensure both children exist before proceeding
            if node.get('L') is not None and node.get('R') is not None:
                m = X_binned[idxs, node['f']] <= node['t']
                q.append((node['L'], idxs[m]))
                q.append((node['R'], idxs[~m]))
            else: # If a split node is malformed, treat it as a leaf
                node['type'] = 'leaf'
                node['leaf'] = len(leaf_val)
                leaf_val[node['leaf']] = y_target[idxs].mean(dim=0, keepdim=False)
        else: # Leaf node
            node['leaf'] = len(leaf_val)
            leaf_val[node['leaf']] = y_target[idxs].mean(dim=0, keepdim=False)

    # -------- predictor ----------------------------------------------------
    def predict(X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Initialize output tensor with correct shape, device, and dtype
            out = torch.zeros(X.size(0), *leaf_val.get(0, zero_vec).shape, device=X.device, dtype=y_target.dtype)
            q = deque([(tree_root, torch.arange(X.size(0), device=X.device))])
            
            while q:
                node, idxs = q.popleft()
                if not idxs.numel() or node is None: continue
                
                if node.get('type') == 'leaf':
                    out[idxs] = leaf_val.get(node.get('leaf'), zero_vec)
                elif node.get('type') == 'split' and node.get('L') and node.get('R'):
                    m = X[idxs, node['f']] <= node['t']
                    q.append((node['L'], idxs[m]))
                    q.append((node['R'], idxs[~m]))
                else: # Fallback for any other malformed node
                    out[idxs] = zero_vec
            return out
    return predict

# --- Sampling & Buffer ---
class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.data: List[Tuple[float, List[int], float, torch.Tensor, Optional[float], int]] = []
        self.step = 0

    def add(self, r: float, t: List[int], p: float, idxs: torch.Tensor):
        if any(t == traj for _, traj, _, _, _, _ in self.data):
            return
        # Entry: (reward, traj, prior, idxs, weight, cached_step)
        self.data.append((r, t, p, idxs, None, -1))
        # Keep the buffer sorted by reward (descending)
        self.data.sort(key=lambda x: x[0], reverse=True)
        if len(self.data) > self.capacity:
            self.data.pop()

    def sample(self, k: int) -> list:
        return random.sample(self.data, min(k, len(self.data)))
    
    def mark_policy_update(self):
        """Increments the policy update counter."""
        self.step += 1

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