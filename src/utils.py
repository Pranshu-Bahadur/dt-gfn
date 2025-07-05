# src/utils.py

from __future__ import annotations

import random
import math
import torch
import lightgbm as lgb
import numpy as np
from typing import Callable, List, Optional, Tuple
from collections import deque

@torch.jit.script
def tb_loss(log_pf, log_pb, log_z, R, prior):
    """Trajectory Balance Loss, batched."""
    return ((log_z + log_pf.sum(1) - (torch.log(R) + prior + log_pb.sum(1)))**2).mean()

@torch.jit.script
def fl_loss(logF, log_pf, log_pb, dE):
    """Flow Matching Loss, batched."""
    return ((logF[:, :-1] + log_pf - (logF[:, 1:] + log_pb - dE))**2).mean()

def deltaE_split_gain(tokens, tok, env):
    """
    Compute ΔE (negative split gain) for every transition in a batch of trajectories.
    """
    y = env.y[env.idxs]; N = y.numel()
    dE = torch.zeros(tokens.shape[1] - 1, device=y.device)
    if N > 0:
        y2 = y * y; full_mse = (y2.mean() - y.mean() ** 2).item()
    else: full_mse = 0.0
    stack_rows = [torch.arange(N, device=y.device)]; stack_mse = [full_mse]
    action_sequence = tokens[0, 1:-1].tolist(); it = iter(tok.decode(action_sequence))
    token_idx = 0
    for kind, idx in it:
        if kind == "feat":
            token_idx += 1
            try: _, th = next(it)
            except StopIteration: break
            if not stack_rows: break
            parent_rows = stack_rows.pop(); parent_mse = stack_mse.pop()
            fv = env.X_full[env.idxs[parent_rows], idx]; mask = fv <= th
            L_rows = parent_rows[mask]; R_rows = parent_rows[~mask]
            def mse_fn(rows):
                if rows.numel() < 2: return 0.0
                yy = y[rows]; return ((yy*yy).mean() - yy.mean()**2).item()
            mseL, mseR = mse_fn(L_rows), mse_fn(R_rows)
            stack_rows.extend([R_rows, L_rows]); stack_mse.extend([mseR, mseL])
            wL = L_rows.numel(); wR = R_rows.numel(); parent_N = wL + wR
            if parent_N > 0:
                gain = parent_mse - (wL/parent_N * mseL + wR/parent_N * mseR)
                dE[token_idx] = -gain
            token_idx += 1
        else:
            if stack_rows: stack_rows.pop(); stack_mse.pop()
            token_idx += 1
    return dE.unsqueeze(0)

def get_tree_predictor(traj, X_binned, y_target, tok):
    """
    Decode a token-ID trajectory into a tree predictor function.
    """
    path_iter = iter(tok.decode(traj[1:-1]))
    def build_recursive():
        try: k, i = next(path_iter)
        except StopIteration: return None
        if k == 'feat': return {'type': 'split', 'f': i, 't': next(path_iter)[1], 'L': build_recursive(), 'R': build_recursive()}
        return {'type': 'leaf', 'value': 0}
    tree_structure = build_recursive()
    if tree_structure is None: return lambda X: torch.zeros(X.size(0), device=X.device)
    q = [(tree_structure, torch.arange(X_binned.size(0), device=X_binned.device))]
    while q:
        node, indices = q.pop(0)
        if node['type'] == 'split':
            if not indices.numel() or node.get('L') is None: continue
            mask = X_binned[indices, node['f']] <= node['t']
            q.append((node['L'], indices[mask])); q.append((node['R'], indices[~mask]))
        else:
            node['value'] = y_target[indices].mean().item() if indices.numel() > 0 else y_target.mean().item()
    def predict(X_test):
        out = torch.empty(X_test.size(0), device=X_test.device)
        stack = [(tree_structure, torch.arange(X_test.size(0), device=X_test.device))]
        while stack:
            node, indices = stack.pop()
            if not indices.numel() or not node: continue
            if node['type'] == 'leaf': out[indices] = node['value']
            else:
                mask = X_test[indices, node['f']] <= node['t']
                if node.get('L'): stack.append((node['L'], indices[mask]))
                if node.get('R'): stack.append((node['R'], indices[~mask]))
        return out
    return predict

class ReplayBuffer:
    """A simple, list-based replay buffer that sorts trajectories by reward."""
    def __init__(self, capacity=10000):
        self.capacity, self.data = capacity, []
    def add(self, r, t, p, idxs):
        self.data.append((r, t, p, idxs)); self.data.sort(key=lambda x:x[0], reverse=True)
        if len(self.data) > self.capacity: self.data.pop()
    def sample(self, k): return random.sample(self.data, min(k, len(self.data)))

def _safe_sample(logits, mask, temperature):
    """Masked, temperature-controlled sampling from logits."""
    logits = logits / temperature; masked = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device))
    probs = torch.softmax(masked, dim=-1); return torch.multinomial(probs, 1).item()

def create_gain_bias(df_train, feats, target, tok, bins, prior_scale=0.5) -> torch.Tensor:
    """Uses a LightGBM model to create a gain-based prior for the policy network."""
    if len(df_train) > 200_000:
        df_sample = df_train.sample(n=200_000, random_state=42)
    else:
        df_sample = df_train
    X_binned = []
    for f in feats:
        s = df_sample[f].replace([np.inf,-np.inf],np.nan).fillna(df_sample[f].median()).values
        qs = np.linspace(0,1,bins+1); edges = np.quantile(s,qs)
        edges = np.unique(edges); edges[0] -= 1e-9; edges[-1] += 1e-9
        X_binned.append(np.searchsorted(edges,s,side="right")-1)
    X_binned = np.stack(X_binned,1)
    y_train = df_sample[target].values
    lgb_train = lgb.Dataset(X_binned, y_train, feature_name=feats)
    params = { 'objective': 'regression_l1', 'metric': 'l1', 'n_estimators': 100, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'num_leaves': 1024, 'max_depth': 10, 'verbose': -1, 'n_jobs': -1 }
    gbm = lgb.train(params, lgb_train)
    tree_info = gbm.dump_model()["tree_info"]
    bias = torch.zeros(tok.v.size(), dtype=torch.float32)
    def parse_node(node):
        nonlocal bias
        if "split_gain" in node and node["split_gain"] > 0:
            f_idx = node["split_feature"]
            gain = node["split_gain"]
            tok_id_feat = tok._feat(f_idx)
            if tok_id_feat < tok.v.size():
                bias[tok_id_feat] += gain
            try:
                bin_idx = min(int(node["threshold"]), bins - 1)
                tok_id_th = tok._th(bin_idx)
                if tok_id_th < tok.v.size():
                    bias[tok_id_th] += gain
            except ValueError:
                try:
                    categories = [int(c) for c in node["threshold"].split('||')]
                    if len(categories) > 0:
                        distributed_gain = gain / len(categories)
                        for cat_idx in categories:
                            bin_idx = min(cat_idx, bins - 1)
                            tok_id_th = tok._th(bin_idx)
                            if tok_id_th < tok.v.size():
                                bias[tok_id_th] += distributed_gain
                except (ValueError, AttributeError):
                    pass
        if "left_child" in node: parse_node(node["left_child"])
        if "right_child" in node: parse_node(node["right_child"])
    for tree in tree_info:
        parse_node(tree['tree_structure'])
    bias_std = bias.std()
    if bias_std > 1e-6: bias /= bias_std
    bias *= (prior_scale / 2.0)
    return torch.tensor(bias, dtype=torch.float32)