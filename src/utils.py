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
def fl_loss(logF, log_pf, log_pb, dR):
    """Flow-Matching loss with positive rewards."""
    return (
        (logF[:, :-1] + log_pf
         - (logF[:, 1:] + log_pb + dR)          # was “- dE”
        )**2
    ).mean()

# --- utils.py --------------------------------------------------------------

# -------------------------------------------------------------------------
#  ΔR with first- and second-order stats  (XGBoost-style split gain)
# -------------------------------------------------------------------------
def deltaE_split_gain_GH(
        tokens:  torch.Tensor,     # [1, L]  token ids incl. BOS/EOS
        tok,                       # Tokenizer
        env,                       # TabularEnv  (provides X_full, idxs, ...)
        g: torch.Tensor,           # gradient   vector[ N_total ]
        h: torch.Tensor,           # Hessian    vector[ N_total ]
        lam: float = 1.0,          # L2 reg λ
        gamma: float = 0.0         # split cost γ
) -> torch.Tensor:
    """
    Per-step positive reward ΔR using gradient/Hessian formulation:

        gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_P²/(H_P+λ) ] − γ

    Equivalent to XGBoost/LightGBM when λ,γ match.
    Returns: 1×(L-1) tensor (no reward for BOS/EOS).
    """
    device = tokens.device
    seq = tokens[0, 1:-1].tolist()                      # strip BOS/EOS
    dR  = torch.zeros(len(seq), device=device)

    # Stack keeps (row_idx_tensor, G_sum, H_sum) for the *current* leaf
    full_rows = env.idxs.clone()                       # rows of root leaf
    G_root = g[full_rows].sum()
    H_root = h[full_rows].sum()
    stack_rows  = [full_rows]
    stack_G     = [G_root]
    stack_H     = [H_root]

    tok_iter = iter(tok.decode(seq))
    t = 0
    for kind, idx in tok_iter:
        if kind == "feat":                             # split node
            t += 1
            # paired threshold token
            try:
                _, th = next(tok_iter)
            except StopIteration:
                break

            rows_P = stack_rows.pop()
            G_P    = stack_G.pop()
            H_P    = stack_H.pop()

            # split parent rows
            fv   = env.X_full[rows_P, idx]
            mask = fv <= th
            rows_L, rows_R = rows_P[mask], rows_P[~mask]

            G_L, H_L = g[rows_L].sum(), h[rows_L].sum()
            G_R, H_R = G_P - G_L, H_P - H_L      # faster than re-indexing

            # classic second-order gain
            gain = 0.5 * (
                (G_L*G_L)/(H_L + lam) +
                (G_R*G_R)/(H_R + lam) -
                (G_P*G_P)/(H_P + lam)
            ) - gamma
            dR[t] = max(gain, 0.0)               # keep reward ≥ 0

            # push children for DFS order   (R first so L is processed next)
            stack_rows.extend([rows_R, rows_L])
            stack_G.extend([G_R, G_L])
            stack_H.extend([H_R, H_L])
            t += 1
        else:                                     # "th" or "leaf"
            # simply pop when we finish a leaf path
            if stack_rows:
                stack_rows.pop(); stack_G.pop(); stack_H.pop()
            t += 1

    return dR.unsqueeze(0)



def get_tree_predictor(traj, X_binned, y_target, tok):
    """
    Decode a token-ID trajectory into a tree predictor function (vectorized version).
    """
    path_iter = iter(tok.decode(traj[1:-1]))
    def build_recursive():
        try: k, i = next(path_iter)
        except StopIteration: return None
        if k == 'feat': return {'type': 'split', 'f': i, 't': next(path_iter)[1], 'L': build_recursive(), 'R': build_recursive()}
        return {'type': 'leaf', 'value': 0}

    tree_structure = build_recursive()
    if tree_structure is None: return lambda X: torch.zeros(X.size(0), device=X.device)

    # Get leaf values from the training data
    leaf_values = {}
    q = deque([(tree_structure, torch.arange(X_binned.size(0), device=X_binned.device))])
    leaf_idx = 0
    while q:
        node, indices = q.popleft()
        if node['type'] == 'split':
            if not indices.numel() or node.get('L') is None: continue
            mask = X_binned[indices, node['f']] <= node['t']
            q.append((node['L'], indices[mask])); q.append((node['R'], indices[~mask]))
        else:
            node['leaf_idx'] = leaf_idx
            leaf_values[leaf_idx] = y_target[indices].mean().item() if indices.numel() > 0 else y_target.mean().item()
            leaf_idx += 1

    def predict(X_test):
        # Vectorized prediction
        out = torch.zeros(X_test.size(0), device=X_test.device)
        
        # Get the leaf index for each sample in a vectorized way
        leaf_indices = torch.zeros(X_test.size(0), dtype=torch.long, device=X_test.device)
        
        q = deque([(tree_structure, torch.arange(X_test.size(0), device=X_test.device))])
        while q:
            node, indices = q.popleft()
            if not indices.numel() or not node: continue
            if node['type'] == 'leaf':
                leaf_indices[indices] = node.get('leaf_idx', -1)
            else:
                mask = X_test[indices, node['f']] <= node['t']
                if node.get('L'): q.append((node['L'], indices[mask]))
                if node.get('R'): q.append((node['R'], indices[~mask]))
        
        # Map leaf indices to leaf values
        for l_idx, l_val in leaf_values.items():
            out[leaf_indices == l_idx] = l_val
            
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

import torch
import torch.nn.functional as F

END_TOKEN = 2   # whatever your tokenizer uses
EPS       = 1e-9

def _safe_sample(logits: torch.Tensor,
                 mask:   torch.Tensor,
                 temperature: float = 1.0,
                 fallback: str = "end") -> int:
    """
    Sample an action index safely.
    • logits:      [A] tensor of raw scores.
    • mask:        [A] bool tensor – True  ⟹ keep    • False ⟹ illegal action
    • temperature: >0
    • fallback:    "end" → return END_TOKEN
                   "greedy" → argmax over *all* logits
                   "error"  → raise ValueError
    """
    # 1. apply temperature
    logits = logits / max(temperature, EPS)

    # 2. mask out illegal actions
    logits = logits.masked_fill(~mask, float("-inf"))

    # 3. if *every* action is masked, decide what to do
    if (~mask).all():
        if fallback == "end":
            return END_TOKEN
        elif fallback == "greedy":
            return torch.argmax(logits).item()  # identical to original logits
        else:
            raise ValueError("No valid actions at this state.")

    # 4. numerically stable softmax
    probs = F.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or probs.sum() < EPS:
        # re-normalise just in case (shouldn’t happen, but prevents rare NaNs)
        probs = torch.where(torch.isfinite(logits), torch.exp(logits), torch.zeros_like(logits))
        probs = probs / probs.sum()

    # 5. sample
    return torch.multinomial(probs, 1).item()

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