from __future__ import annotations
import math
import random
from collections import deque
from typing import Callable, List, Optional, Tuple, Deque, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightgbm as lgb

# --------------------
# Loss Functions
# --------------------

@torch.jit.script
def tb_loss(log_pf: torch.Tensor,
            log_pb: torch.Tensor,
            log_z: torch.Tensor,
            R: torch.Tensor,
            prior: torch.Tensor) -> torch.Tensor:
    """
    Trajectory Balance (batched).
    Shapes:
      log_pf, log_pb: [B, T-1] or [T-1]
      log_z, R, prior: [B] or scalar
    """
    if log_pf.dim() == 1: log_pf = log_pf.unsqueeze(0)
    if log_pb.dim() == 1: log_pb = log_pb.unsqueeze(0)

    lp = log_pf.sum(dim=-1)  # [B]
    lb = log_pb.sum(dim=-1)  # [B]
    logR = torch.log(R + 1e-9)

    if log_z.dim() == 0: log_z = log_z.expand_as(lp)
    if logR.dim()  == 0: logR  = logR.expand_as(lp)
    #if prior.dim() == 0: prior = prior.expand_as(lp)

    diff = log_z + lp - (logR + lb)
    return (diff * diff).mean()


@torch.jit.script
def fl_loss(logF: torch.Tensor,
            log_pf: torch.Tensor,
            log_pb: torch.Tensor,
            dR: torch.Tensor) -> torch.Tensor:
    """
    Flow Matching / Detailed Balance (batched).
    dR is the per-edge delta reward (can be shaped gains); shapes:
      logF:  [B, T]
      log_pf, log_pb, dR: [B, T-1]
    """
    return ((logF[:, :-1] + log_pf) - (logF[:, 1:] + log_pb + dR)).pow(2).mean()

# --------------------
# Tree Build & Traverse
# --------------------

def _build_tree_recursively(path_iter: Iterator, token_cursor: List[int]) -> Optional[dict]:
    """
    Build a binary tree from decoded tokens. Assumes sequence like:
      ('feat', f), ('th', t), <left subtree>, <right subtree>, ...
    A 'leaf' token yields a leaf node.
    """
    try:
        start_token_idx = token_cursor[0]
        kind, val = next(path_iter)          # ('feat', f) | ('th', t) | ('leaf', _)
        token_cursor[0] += 1

        if kind == 'feat':
            # Immediately expect a threshold token
            th_kind, thr = next(path_iter)
            token_cursor[0] += 1
            # If something odd came (not 'th'), treat as malformed and make leaf
            if th_kind != 'th':
                return {'type': 'leaf', 'token_idx': start_token_idx}
            return {
                'type': 'split',
                'f': val,
                't': thr,
                'token_idx': start_token_idx,  # index of the 'feat' token in the decoded stream
                'L': _build_tree_recursively(path_iter, token_cursor),
                'R': _build_tree_recursively(path_iter, token_cursor),
            }
        else:
            # 'th' or 'leaf' appearing here acts like a leaf sentinel
            return {'type': 'leaf', 'token_idx': start_token_idx}
    except StopIteration:
        return None


def build_tree(tokens: torch.Tensor, tok: "Tokenizer") -> Optional[dict]:
    """
    tokens: [T] or [1,T], includes BOS ... EOS
    Decodes tokens[1:-1] to (kind,idx) pairs and builds a tree.
    """
    if tokens.dim() > 1:
        if tokens.size(0) == 1:
            tokens = tokens.squeeze(0)
        else:
            raise ValueError("build_tree expects a 1D token tensor or batch size 1.")
    decoded = tok.decode(tokens[1:-1].tolist())
    return _build_tree_recursively(iter(decoded), [0])


def _traverse_and_get_leaves(tokens: torch.Tensor,
                             tok: "Tokenizer",
                             env: "TabularEnv") -> Tuple[List[torch.Tensor], int]:
    """
    BFS traverse to map each leaf to row indices; also count decision nodes.
    Returns (list_of_leaf_indices, n_decision_nodes).
    """
    root = build_tree(tokens, tok)
    if root is None:
        return [env.idxs], 0

    leaves: List[torch.Tensor] = []
    n_dec = 0
    q: Deque = deque([(root, env.idxs)])

    while q:
        node, idxs = q.popleft()
        if node is None:
            continue
        if not idxs.numel():
            if node.get('type') == 'leaf':
                leaves.append(torch.empty(0, dtype=torch.long, device=env.device))
            continue

        if node.get('type') == 'split' and node.get('L') is not None and node.get('R') is not None:
            fv = env.X_full[idxs, node['f']]
            m = fv <= node['t']
            L, R = idxs[m], idxs[~m]
            if L.numel() == 0 or R.numel() == 0:
                # invalid split — treat as leaf
                leaves.append(idxs)
                continue
            n_dec += 1
            q.append((node['L'], L))
            q.append((node['R'], R))
        else:
            leaves.append(idxs)

    if not leaves:
        return [env.idxs], 0
    return leaves, n_dec

# --------------------
# Bayesian Rewards (DT-GFN style)
# --------------------

@torch.no_grad()
def calculate_bayesian_reward(tokens: torch.Tensor,
                              tok: "Tokenizer",
                              env: "TabularEnv",
                              beta: float) -> torch.Tensor:
    """
    Classification reward: Dirichlet–Multinomial evidence with structure prior.
      log R = [Σ_leaves log P(y_leaf | α) - log P(y_root | α)] / N  -  β * n_splits / N
    Returns: [1] positive reward.
    """
    leaves, n_dec = _traverse_and_get_leaves(tokens, tok, env)
    y = env.y_full[env.idxs]
    if y.numel() == 0:
        return torch.tensor([1e-9], device=env.device)

    K = int(y.max().item() + 1)
    alpha = torch.full((K,), 0.1, device=env.device)  # modest symmetric prior

    def dm_log_ev(counts: torch.Tensor) -> torch.Tensor:
        n0 = counts.sum()
        a0 = alpha.sum()
        return (torch.lgamma(a0) - torch.lgamma(a0 + n0)
                + torch.lgamma(alpha + counts).sum()
                - torch.lgamma(alpha).sum())

    root_counts = torch.bincount(y, minlength=K).float()
    L0 = dm_log_ev(root_counts)
    L = torch.tensor(0.0, device=env.device)
    for idx in leaves:
        if idx.numel() == 0: continue
        L = L + dm_log_ev(torch.bincount(y[idx], minlength=K).float())

    N = max(1, int(y.numel()))
    logR = (L - L0) / N - beta * (n_dec / N)
    return logR.exp().clamp_min(1e-9).unsqueeze(0)


@torch.no_grad()
def calculate_bayesian_reward_regression(tokens: torch.Tensor,
                                         tok: "Tokenizer",
                                         env: "TabularEnv",
                                         beta: float) -> torch.Tensor:
    """
    Regression reward: Normal–Inverse–Gamma evidence with structure prior.
      log R = [Σ_leaves log P(y_leaf | μ0,κ0,a0,b0) - log P(y_root | ...)] / N  -  β * n_splits / N
    Returns: [1]
    """
    leaves, n_dec = _traverse_and_get_leaves(tokens, tok, env)
    y = env.y_full[env.idxs].float()
    if y.numel() == 0:
        return torch.tensor([1e-9], device=env.device)

    def nig_log_ev(y_leaf: torch.Tensor,
                   mu0: float, kappa0: float, a0: float, b0: float) -> torch.Tensor:
        n = y_leaf.numel()
        if n == 0: return torch.tensor(0.0, device=y_leaf.device)
        ybar = y_leaf.mean()
        sse = ((y_leaf - ybar)**2).sum()
        kappa_n = kappa0 + n
        a_n = a0 + 0.5 * n
        b_n = b0 + 0.5 * sse + (kappa0 * n * (ybar - mu0)**2) / (2.0 * kappa_n)
        return (torch.lgamma(torch.tensor(a_n, device=y_leaf.device)) - torch.lgamma(torch.tensor(a0, device=y_leaf.device))
                + torch.log(torch.tensor(b0, device=y_leaf.device)) * a0
                - torch.log(b_n) * a_n
                + 0.5 * (math.log(kappa0) - math.log(kappa_n))
                - 0.5 * n * math.log(2.0 * math.pi))

    mu0, kappa0, a0, b0 = 0.0, 1.0, 1.0, 1.0
    L0 = nig_log_ev(y, mu0, kappa0, a0, b0)
    L = torch.tensor(0.0, device=y.device)
    for idx in leaves:
        if idx.numel() == 0: continue
        L = L + nig_log_ev(y[idx], mu0, kappa0, a0, b0)

    N = max(1, int(y.numel()))
    logR = (L - L0) / N - beta * (n_dec / N)
    return logR.exp().clamp_min(1e-9).unsqueeze(0)

# --------------------
# Per-step gains (optional, for FL shaping)
# --------------------

def deltaE_split_gain_regression(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """
    Per-token variance-reduction gains as [1, T-1], placed at 'feat' token positions.
    """
    y = env.y_full[env.idxs].float()
    dR = torch.zeros(tokens.size(-1) - 1, device=y.device)
    root = build_tree(tokens, tok)
    if root is None: return dR.unsqueeze(0)

    def var(rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() < 2: return torch.tensor(0.0, device=y.device)
        return y[rows].var(unbiased=False)

    N = y.numel()
    q = deque([(root, torch.arange(N, device=y.device))])
    while q:
        node, idxs = q.popleft()
        if not (node and node.get('type') == 'split' and node.get('L') and node.get('R') and idxs.numel() > 1):
            continue
        parent_var = var(idxs)
        fv = env.X_full[env.idxs[idxs], node['f']]
        m = fv <= node['t']
        L, R = idxs[m], idxs[~m]
        if L.numel() == 0 or R.numel() == 0: continue
        gain = parent_var - (L.numel()/idxs.numel()) * var(L) - (R.numel()/idxs.numel()) * var(R)
        dR[node['token_idx']] = gain
        q.append((node['L'], L)); q.append((node['R'], R))
    return dR.unsqueeze(0)


def deltaE_split_gain_classification(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """
    Per-token Gini-impurity-reduction gains as [1, T-1], placed at 'feat' token positions.
    """
    y = env.y_full[env.idxs].long()
    dR = torch.zeros(tokens.size(-1) - 1, device=y.device)
    root = build_tree(tokens, tok)
    if root is None: return dR.unsqueeze(0)

    def gini(rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() < 2: return torch.tensor(0.0, device=y.device)
        counts = torch.bincount(y[rows].clamp(0, env.n_classes - 1), minlength=env.n_classes)
        p = counts.float() / counts.sum().clamp_min(1)
        return 1.0 - (p * p).sum()

    N = y.numel()
    q = deque([(root, torch.arange(N, device=y.device))])
    while q:
        node, idxs = q.popleft()
        if not (node and node.get('type') == 'split' and node.get('L') and node.get('R') and idxs.numel() > 1):
            continue
        parent_g = gini(idxs)
        fv = env.X_full[env.idxs[idxs], node['f']]
        m = fv <= node['t']
        L, R = idxs[m], idxs[~m]
        if L.numel() == 0 or R.numel() == 0: continue
        gain = parent_g - (L.numel()/idxs.numel()) * gini(L) - (R.numel()/idxs.numel()) * gini(R)
        dR[node['token_idx']] = gain
        q.append((node['L'], L)); q.append((node['R'], R))
    return dR.unsqueeze(0)

# --------------------
# Predictors (posterior-mean for probs; mean for residuals/regression)
# --------------------

def get_tree_predictor(traj: List[int],
                       X_binned: torch.Tensor,
                       y_target: torch.Tensor,
                       tok: "Tokenizer") -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build the tree by *replaying tokens on training data* (LIFO expansion like rollout),
    accept a split only if both children are non-empty on train, then:
      - if y_target looks like class-probabilities (>=0, <=1, rows sum~1): **Dirichlet sample**
        leaf probs (paper behavior)
      - else (residuals/regression with negatives): **mean** vector per leaf.

    Returns: predictor(X_binned_test) -> Tensor
    """
    device = X_binned.device
    y_target = y_target.detach().to(dtype=torch.float32, device=device)
    v = tok.v

    # --- decode tokens (strip BOS/EOS if present) ---
    if len(traj) > 0 and traj[0] == v.BOS: start = 1
    else: start = 0
    end = len(traj) - 1 if len(traj) and traj[-1] == v.EOS else len(traj)
    decoded = tok.decode(traj[start:end])

    N = X_binned.size(0)
    all_idx = torch.arange(N, device=device, dtype=torch.long)

    # --- classify target mode: probs vs residuals ---
    is_matrix = (y_target.dim() == 2)
    if is_matrix:
        min_ok = float(y_target.min()) >= -1e-6
        max_ok = float(y_target.max()) <= 1.0 + 1e-6
        row_sums = y_target.sum(dim=1)
        sums_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=0)
        is_clf_probs = (min_ok and max_ok and bool(sums_ok))
    else:
        is_clf_probs = False

    # --- build by data with LIFO expansion (matches rollout .pop()) ---
    class Node(dict): pass
    root = Node(type='leaf', idxs=all_idx)
    # stack holds *leaf* nodes to expand (LIFO)
    stack: List[Node] = [root]
    pending: Optional[Tuple[Node, int]] = None  # (node_to_split, feature_id)

    for kind, val in decoded:
        if not stack and pending is None:
            break

        if pending is None:
            if kind == 'feat':
                # take the last (deepest) open leaf, like rollout
                node = stack.pop() if stack else None
                if node is None or node.get('type') != 'leaf':
                    continue
                pending = (node, int(val))
            elif kind == 'leaf':
                # finalize one leaf (consume one open leaf)
                if stack:
                    stack.pop()
            else:
                # 'th' without a preceding 'feat' → ignore
                continue
        else:
            # expecting a threshold for the pending feature
            if kind != 'th':
                # broken grammar -> drop pending and continue
                pending = None
                continue

            node, f = pending
            t = int(val)
            idxs = node.get('idxs', all_idx)

            # split on train data
            fv = X_binned[idxs, f]
            m = fv <= t
            L_idx = idxs[m]
            R_idx = idxs[~m]

            if L_idx.numel() == 0 or R_idx.numel() == 0:
                # reject split -> keep leaf
                pending = None
                continue

            # accept split: mutate node into split, push children (R then L for LIFO → go left next)
            node.clear()
            node.update(type='split', f=f, t=t)
            L = Node(type='leaf', idxs=L_idx)
            R = Node(type='leaf', idxs=R_idx)
            node['L'] = L; node['R'] = R
            stack.append(R)
            stack.append(L)
            pending = None

    # --- collect leaves (with training indices) ---
    leaves: List[Tuple[Node, torch.Tensor]] = []
    q: Deque[Node] = deque([root])
    while q:
        n = q.popleft()
        if n.get('type') == 'split':
            q.append(n['L']); q.append(n['R'])
        else:
            leaves.append((n, n.get('idxs', torch.empty(0, dtype=torch.long, device=device))))

    # --- compute leaf values ---
    if is_matrix and is_clf_probs:
        K = y_target.size(1)
        alpha = torch.full((K,), 0.1 / max(1, K), device=device)
        for node, idxs in leaves:
            if idxs.numel() == 0:
                node['val'] = torch.full((K,), 1.0 / K, device=device)
                continue
            counts = y_target[idxs].sum(0)  # [K], one-hot sums
            conc = counts + alpha
            # Dirichlet *sampling* as requested
            node['val'] = torch.distributions.Dirichlet(conc).sample()
    else:
        # regression / classification residuals: mean vector (handles 1D or 2D)
        for node, idxs in leaves:
            if idxs.numel() == 0:
                node['val'] = torch.zeros_like(y_target[0]) if y_target.dim() > 1 else torch.tensor(0.0, device=device)
            else:
                node['val'] = y_target[idxs].mean(dim=0, keepdim=False)

    # strip training indices to avoid leaks
    for node, _ in leaves:
        if 'idxs' in node: del node['idxs']

    # --- prediction using the *accepted* training tree structure ---
    def predict(X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if is_matrix:
                out = torch.zeros(X.size(0), y_target.size(1), device=X.device, dtype=y_target.dtype)
            else:
                out = torch.zeros(X.size(0), device=X.device, dtype=y_target.dtype)

            q: Deque[Tuple[Node, torch.Tensor]] = deque([(root, torch.arange(X.size(0), device=X.device))])
            while q:
                n, idxs = q.popleft()
                if idxs.numel() == 0:
                    continue
                if n.get('type') == 'split':
                    f = int(n['f']); t = int(n['t'])
                    m = X[idxs, f] <= t
                    L_rows, R_rows = idxs[m], idxs[~m]
                    q.append((n['L'], L_rows))
                    q.append((n['R'], R_rows))
                else:
                    out[idxs] = n['val']
            return out
    return predict


# --------------------
# Replay Buffer
# --------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        # entries: (reward_scalar, seq, prior_scalar, idxs_tensor, pb_weight, last_refresh_step)
        self.data: List[Tuple[float, List[int], float, torch.Tensor, Optional[float], int]] = []
        self.step = 0

    def add(self, r: float, t: List[int], p: float, idxs: torch.Tensor):
        if any(t == traj for _, traj, _, _, _, _ in self.data):
            return
        self.data.append((r, t, p, idxs, None, -1))
        self.data.sort(key=lambda x: x[0], reverse=True)
        if len(self.data) > self.capacity:
            self.data.pop()

    def sample(self, k: int) -> list:
        return random.sample(self.data, min(k, len(self.data)))

    def mark_policy_update(self):
        self.step += 1

# --------------------
# Safe sample (token masking + temperature)
# --------------------

END_TOKEN = 2
EPS = 1e-9

def _safe_sample(logits: torch.Tensor,
                 mask: torch.Tensor,
                 temperature: float = 1.0) -> torch.Tensor:
    """
    Apply mask and temperature, then sample multinomially.
    If all tokens are masked on a row, force END_TOKEN to be selectable.
    """
    logits = logits.masked_fill(~mask, -float("inf"))
    all_masked = (~mask).all(dim=-1)
    if all_masked.any():
        logits[all_masked, END_TOKEN] = 0.0

    if temperature is not None and temperature > EPS:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(1)

# --------------------
# Optional token prior (gain bias)
# --------------------

def create_gain_bias(df_train: pd.DataFrame,
                     feats: List[str],
                     target: str,
                     tok: "Tokenizer",
                     bins: int,
                     prior_scale: float = 0.5) -> torch.Tensor:
    """
    LightGBM mining to produce a token prior over feature/threshold tokens.
    """
    df_sample = df_train.sample(n=min(len(df_train), 200_000), random_state=42)
    X_binned_list = []
    for f in feats:
        s = df_sample[f].replace([np.inf, -np.inf], np.nan).fillna(df_sample[f].median()).values
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(s, qs)) if len(np.unique(s)) > 1 else np.array([s[0] - 1, s[0] + 1])
        edges[0] -= 1e-9; edges[-1] += 1e-9
        X_binned_list.append(np.searchsorted(edges, s, side="right") - 1)
    X_binned = np.stack(X_binned_list, 1)
    y_train = df_sample[target].values

    lgb_train = lgb.Dataset(X_binned, y_train, feature_name=feats, free_raw_data=False)
    params = dict(objective='regression_l1', metric='l1', n_estimators=100, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  num_leaves=1024, max_depth=10, verbose=-1, n_jobs=-1)
    gbm = lgb.train(params, lgb_train)
    tree_info = gbm.dump_model()['tree_info']

    bias = torch.zeros(tok.v.size(), dtype=torch.float32)
    def parse_node(node: dict):
        nonlocal bias
        if 'split_gain' in node and node['split_gain'] > 0:
            gain = node['split_gain']
            f_idx = node['split_feature']
            tok_feat = tok._feat(f_idx)
            if tok_feat < tok.v.size(): bias[tok_feat] += gain
            try:
                bin_idx = int(node['threshold'])
                tok_th = tok._th(min(bin_idx, bins - 1))
                if tok_th < tok.v.size(): bias[tok_th] += gain
            except Exception:
                pass
        if 'left_child' in node: parse_node(node['left_child'])
        if 'right_child' in node: parse_node(node['right_child'])

    for tree in tree_info:
        parse_node(tree['tree_structure'])

    std = bias.std()
    if std > 1e-6: bias /= std
    bias *= (prior_scale / 2.0)
    return bias
