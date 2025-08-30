from __future__ import annotations
import math
import random
from collections import deque
from typing import Callable, List, Optional, Tuple, Deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import lightgbm as lgb

# ============================================================
# Loss Functions
# ============================================================

@torch.jit.script
def tb_loss(log_pf: torch.Tensor,
            log_pb: torch.Tensor,
            log_z: torch.Tensor,
            R: torch.Tensor,
            prior: torch.Tensor) -> torch.Tensor:
    """
    Trajectory Balance (batched).
      logZ + Σ log p_f  ≈  logR + Σ log p_b
    NOTE: 'prior' is accepted for API compatibility but not used here;
          include structure priors inside R to avoid double-counting.
    Shapes:
      log_pf, log_pb : [B, T-1] or [T-1]
      log_z, R       : [B] or scalar
    """
    if log_pf.dim() == 1:
        log_pf = log_pf.unsqueeze(0)
    if log_pb.dim() == 1:
        log_pb = log_pb.unsqueeze(0)

    lp = log_pf.sum(dim=-1)                          # [B]
    lb = log_pb.sum(dim=-1)                          # [B]
    logR = torch.log(R.clamp_min(1e-9))              # [B] or scalar

    if log_z.dim() == 0:
        log_z = log_z.expand_as(lp)
    if logR.dim() == 0:
        logR = logR.expand_as(lp)

    diff = log_z + lp - (logR + lb)
    return (diff * diff).mean()


@torch.jit.script
def fl_loss(logF: torch.Tensor,
            log_pf: torch.Tensor,
            log_pb: torch.Tensor,
            dR: torch.Tensor) -> torch.Tensor:
    """
    Flow Matching / Detailed Balance (batched).
      (logF_t + log p_f)  ≈  (logF_{t+1} + log p_b + ΔR_t)
    Shapes:
      logF:  [B, T]
      log_pf, log_pb, dR: [B, T-1]
    """
    t1 = logF.size(1) - 1
    t2 = log_pf.size(1)
    t3 = log_pb.size(1)
    t4 = dR.size(1)
    T = min(t1, t2, t3, t4)
    if T <= 0:
        return torch.zeros((), device=logF.device)
    return ((logF[:, :T] + log_pf[:, :T]) - (logF[:, 1:T+1] + log_pb[:, :T] + dR[:, :T])).pow(2).mean()

# ============================================================
# Token utilities (replay & building)
# ============================================================

def _decode_no_bos_eos(tokens: torch.Tensor, tok: "Tokenizer") -> List[Tuple[str, int]]:
    """Strip BOS/EOS and decode."""
    if tokens.dim() > 1:
        if tokens.size(0) == 1:
            tokens = tokens.squeeze(0)
        else:
            raise ValueError("Expect 1D tokens or batch size 1.")
    ids = tokens.tolist()
    if len(ids) and ids[0] == tok.v.BOS:
        ids = ids[1:]
    if len(ids) and ids[-1] == tok.v.EOS:
        ids = ids[:-1]
    return tok.decode(ids)

class _Node(dict):
    pass

def _build_tree_by_data(tokens: torch.Tensor,
                        tok: "Tokenizer",
                        X_binned: torch.Tensor,
                        idxs: torch.Tensor) -> Tuple[_Node, int]:
    """
    Rebuild the tree by replaying tokens on TRAIN DATA:
      • LIFO expansion of leaves (like rollout)
      • accept split only if both children non-empty
      • store token_idx of 'feat' position for per-step gains
    Returns: (root_node, n_decision_nodes)
    """
    decoded = _decode_no_bos_eos(tokens, tok)
    N = idxs.numel()
    device = X_binned.device

    root = _Node(type='leaf', idxs=torch.arange(N, device=device, dtype=torch.long))
    stack: List[_Node] = [root]
    pending: Optional[Tuple[_Node, int, int]] = None   # (node, feature, feat_pos)
    n_dec = 0

    for pos, (kind, val) in enumerate(decoded):
        if not stack and pending is None:
            break

        if pending is None:
            if kind == 'feat':
                node = stack.pop() if stack else None
                if node is None or node.get('type') != 'leaf':
                    continue
                pending = (node, int(val), pos)
            elif kind == 'leaf':
                if stack:
                    stack.pop()
            else:
                continue
        else:
            if kind != 'th':
                pending = None
                continue
            node, f, feat_pos = pending
            t = int(val)
            # node.idxs are indices into the local subset (0..N_batch-1)
            local = node.get('idxs')
            global_rows = idxs[local]
            fv = X_binned[global_rows, f]
            m = fv <= t
            L_idx = local[m]
            R_idx = local[~m]
            if L_idx.numel() == 0 or R_idx.numel() == 0:
                pending = None
                continue

            node.clear()
            node.update(type='split', f=f, t=t, token_idx=feat_pos)
            L = _Node(type='leaf', idxs=L_idx)
            R = _Node(type='leaf', idxs=R_idx)
            node['L'] = L; node['R'] = R
            stack.append(R)  # R then L to replicate LIFO (expand left next)
            stack.append(L)
            n_dec += 1
            pending = None

    return root, n_dec


# OLD signature:
# def _collect_leaf_indices(root: _Node) -> List[torch.Tensor]:

def _collect_leaf_indices(root: _Node, device: torch.device) -> List[torch.Tensor]:
    """Return list of train row-index tensors from a built tree."""
    leaves: List[torch.Tensor] = []
    q: Deque[_Node] = deque([root])
    while q:
        n = q.popleft()
        if n.get('type') == 'split':
            q.append(n['L']); q.append(n['R'])
        else:
            idxs = n.get('idxs', None)
            if idxs is None:
                idxs = torch.empty(0, dtype=torch.long, device=device)
            leaves.append(idxs)
    return leaves


# ============================================================
# Bayesian Rewards (DT-GFN style) — numerically stable
# ============================================================

@torch.no_grad()
def _dm_log_evidence_from_counts(counts64: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a0 = alpha.sum()
    n0 = counts64.sum()
    return (torch.lgamma(a0)
            - torch.lgamma(a0 + n0)
            + torch.lgamma(alpha + counts64).sum()
            - torch.lgamma(alpha).sum())

@torch.no_grad()
def _nig_log_evidence(y64: torch.Tensor,
                      mu0: float = 0.0, kappa0: float = 1.0,
                      a0: float = 1.0, b0: float = 1.0) -> torch.Tensor:
    if y64.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=y64.device)
    n = torch.tensor(float(y64.numel()), dtype=torch.float64, device=y64.device)
    ybar = y64.mean()
    sse = ((y64 - ybar) ** 2).sum()
    kappa0 = torch.tensor(kappa0, dtype=torch.float64, device=y64.device)
    a0 = torch.tensor(a0, dtype=torch.float64, device=y64.device)
    b0 = torch.tensor(b0, dtype=torch.float64, device=y64.device)
    mu0 = torch.tensor(mu0, dtype=torch.float64, device=y64.device)
    kappa_n = kappa0 + n
    a_n = a0 + 0.5 * n
    b_n = b0 + 0.5 * sse + (kappa0 * n * (ybar - mu0) ** 2) / (2.0 * kappa_n) + 1e-12
    return (torch.lgamma(a_n) - torch.lgamma(a0)
            + a0 * torch.log(b0 + 1e-12)
            - a_n * torch.log(b_n)
            + 0.5 * (torch.log(kappa0) - torch.log(kappa_n))
            - 0.5 * n * math.log(2.0 * math.pi))

@torch.no_grad()
def calculate_bayesian_reward(tokens: torch.Tensor,
                              tok: "Tokenizer",
                              env: "TabularEnv",
                              beta: float) -> torch.Tensor:
    """
    Classification reward (Dirichlet–Multinomial evidence + structure prior).
      log R = [Σ_leaves log P(y_leaf | α) - log P(y_root | α)] / N  -  β * n_splits / N
    """
    root, n_dec = _build_tree_by_data(tokens, tok, env.X_full, env.idxs)
    leaves = _collect_leaf_indices(root, env.X_full.device)
    y = env.y_full[env.idxs].to(torch.long)
    if y.numel() == 0:
        return torch.tensor([1e-9], device=env.device)

    K = int(getattr(env, "n_classes", int(y.max().item()) + 1))
    alpha = torch.full((K,), 0.1, dtype=torch.float64, device=env.device)

    root_counts = torch.bincount(y, minlength=K).to(torch.float64)
    L0 = _dm_log_evidence_from_counts(root_counts, alpha)

    L = torch.zeros((), dtype=torch.float64, device=env.device)
    any_leaf = False
    for idx in leaves:
        if idx.numel() == 0:
            continue
        any_leaf = True
        leaf_counts = torch.bincount(y[idx], minlength=K).to(torch.float64)
        L = L + _dm_log_evidence_from_counts(leaf_counts, alpha)
    if not any_leaf:
        L, n_dec = L0, 0

    N = max(1, int(y.numel()))
    logR = (L - L0) / N - float(beta) * (n_dec / N)
    logR = torch.clamp(logR, min=-50.0, max=50.0).to(torch.float32)
    return torch.exp(logR).clamp_min(1e-9).unsqueeze(0)

@torch.no_grad()
def calculate_bayesian_reward_regression(tokens: torch.Tensor,
                                         tok: "Tokenizer",
                                         env: "TabularEnv",
                                         beta: float) -> torch.Tensor:
    """
    Regression reward (Normal–Inverse–Gamma evidence + structure prior).
      log R = [Σ_leaves log P(y_leaf | μ0,κ0,a0,b0) - log P(y_root | ...)] / N  -  β * n_splits / N
    """
    root, n_dec = _build_tree_by_data(tokens, tok, env.X_full, env.idxs)
    leaves = _collect_leaf_indices(root, env.X_full.device)

    y = env.y_full[env.idxs].to(torch.float64)
    if y.numel() == 0:
        return torch.tensor([1e-9], device=env.device)

    L0 = _nig_log_evidence(y)
    L = torch.zeros((), dtype=torch.float64, device=y.device)
    any_leaf = False
    for idx in leaves:
        if idx.numel() == 0:
            continue
        any_leaf = True
        L = L + _nig_log_evidence(y[idx])
    if not any_leaf:
        L, n_dec = L0, 0

    N = max(1, int(y.numel()))
    logR = (L - L0) / N - float(beta) * (n_dec / N)
    logR = torch.clamp(logR, min=-50.0, max=50.0).to(torch.float32)
    return torch.exp(logR).clamp_min(1e-9).unsqueeze(0)

# ============================================================
# Per-step gains
# ============================================================

def deltaE_split_gain_regression(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """Per-token variance-reduction gains [1, T-1], at the 'feat' token positions."""
    root, _ = _build_tree_by_data(tokens, tok, env.X_full, env.idxs)
    y = env.y_full[env.idxs].float()
    dR = torch.zeros(tokens.size(-1) - 1, device=y.device)

    def var(rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() < 2:
            return torch.tensor(0.0, device=y.device)
        return y[rows].var(unbiased=False)

    q: Deque[Tuple[_Node, torch.Tensor]] = deque([(root, torch.arange(y.numel(), device=y.device))])
    while q:
        node, idxs = q.popleft()
        if node.get('type') != 'split' or idxs.numel() <= 1:
            continue
        parent_var = var(idxs)
        fv = env.X_full[env.idxs[idxs], node['f']]
        m = fv <= node['t']
        L, R = idxs[m], idxs[~m]
        if L.numel() == 0 or R.numel() == 0:
            continue
        gain = parent_var - (L.numel()/idxs.numel()) * var(L) - (R.numel()/idxs.numel()) * var(R)
        dR[int(node['token_idx'])] = gain
        q.append((node['L'], L)); q.append((node['R'], R))
    return dR.unsqueeze(0)

def deltaE_split_gain_classification(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """Per-token Gini-impurity-reduction gains [1, T-1], at the 'feat' token positions."""
    root, _ = _build_tree_by_data(tokens, tok, env.X_full, env.idxs)
    y = env.y_full[env.idxs].long()
    dR = torch.zeros(tokens.size(-1) - 1, device=y.device)

    def gini(rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() < 2:
            return torch.tensor(0.0, device=y.device)
        counts = torch.bincount(y[rows].clamp(0, env.n_classes - 1), minlength=env.n_classes)
        p = counts.float() / counts.sum().clamp_min(1)
        return 1.0 - (p * p).sum()

    q: Deque[Tuple[_Node, torch.Tensor]] = deque([(root, torch.arange(y.numel(), device=y.device))])
    while q:
        node, idxs = q.popleft()
        if node.get('type') != 'split' or idxs.numel() <= 1:
            continue
        parent_g = gini(idxs)
        fv = env.X_full[env.idxs[idxs], node['f']]
        m = fv <= node['t']
        L, R = idxs[m], idxs[~m]
        if L.numel() == 0 or R.numel() == 0:
            continue
        gain = parent_g - (L.numel()/idxs.numel()) * gini(L) - (R.numel()/idxs.numel()) * gini(R)
        dR[int(node['token_idx'])] = gain
        q.append((node['L'], L)); q.append((node['R'], R))
    return dR.unsqueeze(0)

def deltaE_split_gain_sse(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """
    Per-token SSE reduction [1, T-1] using env.y (vector or matrix).
    Works for regression AND multi-class residuals.
    """
    root, _ = _build_tree_by_data(tokens, tok, env.X_full, env.idxs)

    # prefer env.y (residuals) if present; otherwise y_full
    Y_all = (env.y if getattr(env, "y", None) is not None else env.y_full).float()
    Y = Y_all[env.idxs]
    dR = torch.zeros(tokens.size(-1) - 1, device=Y.device)

    def sse(rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() <= 1:
            return torch.tensor(0.0, device=Y.device)
        Z = Y[rows]
        if Z.ndim == 1:
            mu = Z.mean()
            return ((Z - mu) ** 2).sum()
        mu = Z.mean(dim=0, keepdim=True)
        return ((Z - mu) ** 2).sum()

    q: Deque[Tuple[_Node, torch.Tensor]] = deque([(root, torch.arange(Y.size(0), device=Y.device))])
    while q:
        node, idxs = q.popleft()
        if node.get('type') != 'split' or idxs.numel() <= 1:
            continue
        parent = sse(idxs)
        fv = env.X_full[env.idxs[idxs], node['f']]
        m = fv <= node['t']
        L, R = idxs[m], idxs[~m]
        if L.numel() == 0 or R.numel() == 0:
            continue
        gain = parent - (sse(L) + sse(R))
        dR[int(node['token_idx'])] = gain
        q.append((node['L'], L)); q.append((node['R'], R))
    return dR.unsqueeze(0)

# ============================================================
# Predictor (Dirichlet sampling / posterior mean for probs)
# ============================================================

def get_tree_predictor(
    traj: List[int],
    X_binned: torch.Tensor,
    y_target: torch.Tensor,
    tok: "Tokenizer",
    *,
    min_child_size: int = 20,
    min_gain: float = 0.0,
    predictor_mode: str = "dirichlet",  # "dirichlet" | "mean" (classification only)
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Rebuild the tree by replaying tokens on TRAIN data with stronger split checks:
      • accept split only if both children have >= min_child_size
      • and (optionally) gain >= min_gain

    Leaf values:
      • classification (y_target one-hot probs):
          - predictor_mode == "dirichlet": sample from Dirichlet posterior
          - predictor_mode == "mean":      use Dirichlet posterior mean
      • regression / residuals: per-leaf mean

    Returns: predictor(X_binned_test) -> Tensor
    """
    device = X_binned.device
    y_target = y_target.detach().to(dtype=torch.float32, device=device)
    v = tok.v

    # decode tokens (strip BOS/EOS if present)
    start = 1 if len(traj) and traj[0] == v.BOS else 0
    end = len(traj) - 1 if len(traj) and traj[-1] == v.EOS else len(traj)
    decoded = tok.decode(traj[start:end])

    N = X_binned.size(0)
    all_idx = torch.arange(N, device=device, dtype=torch.long)

    # detect mode
    is_matrix = (y_target.dim() == 2)
    if is_matrix:
        min_ok = float(y_target.min()) >= -1e-6
        max_ok = float(y_target.max()) <= 1.0 + 1e-6
        row_sums = y_target.sum(dim=1)
        sums_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=0.0)
        is_clf_probs = (min_ok and max_ok and bool(sums_ok))
        K = y_target.size(1)
    else:
        is_clf_probs = False
        K = None

    # impurity helpers
    def node_sse(idxs: torch.Tensor) -> torch.Tensor:
        if idxs.numel() <= 1:
            return torch.tensor(0.0, device=device)
        Y = y_target[idxs]
        if Y.ndim == 1:
            mu = Y.mean()
            return ((Y - mu) ** 2).sum()
        else:
            mu = Y.mean(dim=0, keepdim=True)
            return ((Y - mu) ** 2).sum()

    def node_gini(idxs: torch.Tensor) -> torch.Tensor:
        if not is_clf_probs or idxs.numel() == 0:
            return torch.tensor(0.0, device=device)
        counts = y_target[idxs].sum(0)  # [K]
        n = counts.sum().clamp_min(1.0)
        p = counts / n
        return 1.0 - (p * p).sum()

    def split_gain(parent_idx: torch.Tensor, L_idx: torch.Tensor, R_idx: torch.Tensor) -> float:
        nP = float(parent_idx.numel())
        if nP <= 1:
            return 0.0
        if is_clf_probs:
            gP = node_gini(parent_idx)
            gL = node_gini(L_idx)
            gR = node_gini(R_idx)
            nL = float(L_idx.numel()); nR = float(R_idx.numel())
            gain = gP * nP - (gL * nL + gR * nR)
        else:
            sP = node_sse(parent_idx)
            sL = node_sse(L_idx)
            sR = node_sse(R_idx)
            gain = float((sP - (sL + sR)).item())
        return float(gain)

    # build by data with LIFO expansion
    class Node(dict): pass
    root = Node(type='leaf', idxs=all_idx)
    stack: List[Node] = [root]
    pending: Optional[Tuple[Node, int]] = None  # (node_to_split, feature_id)

    for kind, val in decoded:
        if not stack and pending is None:
            break

        if pending is None:
            if kind == 'feat':
                node = stack.pop() if stack else None
                if node is None or node.get('type') != 'leaf':
                    continue
                pending = (node, int(val))
            elif kind == 'leaf':
                if stack:
                    stack.pop()
            else:
                continue
        else:
            if kind != 'th':
                pending = None
                continue

            node, f = pending
            t = int(val)
            idxs = node.get('idxs', all_idx)

            fv = X_binned[idxs, f]
            m = fv <= t
            L_idx = idxs[m]
            R_idx = idxs[~m]

            # guards
            if (L_idx.numel() < min_child_size) or (R_idx.numel() < min_child_size):
                pending = None
                continue
            if min_gain > 0.0:
                g = split_gain(idxs, L_idx, R_idx)
                if g < min_gain:
                    pending = None
                    continue

            node.clear()
            node.update(type='split', f=f, t=t)
            L = Node(type='leaf', idxs=L_idx)
            R = Node(type='leaf', idxs=R_idx)
            node['L'] = L; node['R'] = R
            stack.append(R); stack.append(L)
            pending = None

    # collect leaves
    leaves: List[Tuple[Node, torch.Tensor]] = []
    q: Deque[Node] = deque([root])
    while q:
        n = q.popleft()
        if n.get('type') == 'split':
            q.append(n['L']); q.append(n['R'])
        else:
            leaves.append((n, n.get('idxs', torch.empty(0, dtype=torch.long, device=device))))

    # compute leaf values
    if is_clf_probs:
        alpha = torch.full((K,), 0.1 / max(1, K), device=device)
        for node, idxs in leaves:
            if idxs.numel() == 0:
                node['val'] = torch.full((K,), 1.0 / K, device=device)
                continue
            counts = y_target[idxs].sum(0)  # [K]
            conc = counts + alpha
            if predictor_mode == "dirichlet":
                node['val'] = torch.distributions.Dirichlet(conc).sample()
            else:
                node['val'] = conc / conc.sum()
    else:
        for node, idxs in leaves:
            if idxs.numel() == 0:
                node['val'] = (torch.zeros_like(y_target[0])
                               if y_target.dim() > 1
                               else torch.tensor(0.0, device=device))
            else:
                node['val'] = y_target[idxs].mean(dim=0, keepdim=False)

    # strip training indices from nodes
    for node, _ in leaves:
        node.pop('idxs', None)

    # prediction fn
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
                    q.append((n['L'], idxs[m]))
                    q.append((n['R'], idxs[~m]))
                else:
                    out[idxs] = n['val']
            return out
    return predict
# ============================================================
# Replay Buffer
# ============================================================
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

    def sample_tempered(self, k: int, tau: float = 1.0) -> list:
        if not self.data:
            return []
        k = min(k, len(self.data))
        if tau <= 0:
            # nearly argmax → fall back to top-k by reward
            entries = sorted(self.data, key=lambda e: e[0], reverse=True)
            return entries[:k]
        import numpy as np
        rewards = np.array([max(e[0], 1e-9) for e in self.data], dtype=np.float64)
        w = np.exp(rewards / float(tau))
        w = w / w.sum()
        idx = np.random.choice(len(self.data), size=k, replace=False, p=w)
        return [self.data[i] for i in idx]

    def mark_policy_update(self):
        self.step += 1

# ============================================================
# Safe sample (token masking + temperature)
# ============================================================

EPS = 1e-9

def _safe_sample(logits: torch.Tensor,
                 mask: torch.Tensor,
                 temperature: float = 1.0,
                 eos_id: int = 2) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -float("inf"))
    all_masked = (~mask).all(dim=-1)
    if all_masked.any():
        logits[all_masked, eos_id] = 0.0
    if temperature is not None and temperature > EPS:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(1)

# ============================================================
# Optional: uniform backward log-prob surrogate
# ============================================================

@torch.no_grad()
def uniform_backward_log_prob(padded_seq: torch.Tensor, tok, max_depth: int) -> torch.Tensor:
    """
    Approximate teacher-forced backward log-probabilities assuming a
    UNIFORM policy over the *forward* grammar’s valid actions.

    This ignores data-dependent feasibility and only enforces:
      • max depth,
      • feature window feasibility via (lo, hi) thresholds,
      • LEAF is forbidden at root unless no split is possible.

    Returns: (B, T-1) float32 with zeros at EOS and PAD.
    """
    seq = padded_seq  # (B, T) long
    B, T = seq.shape
    device = seq.device
    v = tok.v
    PAD, EOS, LEAF = int(v.PAD), int(v.EOS), int(tok._leaf(0))

    out = torch.zeros((B, T - 1), device=device, dtype=torch.float32)

    for b in range(B):
        s = seq[b]

        depth_stack = [0]
        lo_stack = [torch.zeros(v.num_feat, dtype=torch.long, device=device)]
        hi_stack = [torch.full((v.num_feat,), v.num_th - 1, dtype=torch.long, device=device)]
        pending = None  # (depth, lo_top, hi_top, f_idx)

        for t in range(T - 1):
            nxt = int(s[t + 1].item())

            if nxt == PAD:
                break
            if nxt == EOS:
                break
            if len(depth_stack) == 0:
                out[b, t] = 0.0
                continue

            if pending is None:
                # Feature/Leaf decision
                d = depth_stack[-1]
                root = (d == 0)
                lo_top = lo_stack[-1]
                hi_top = hi_stack[-1]
                can_split = (d < max_depth)
                valid_feats = (lo_top <= hi_top).nonzero(as_tuple=False).flatten()
                can_close = (not root) or (not can_split) or (valid_feats.numel() == 0)
                count = int(valid_feats.numel()) + (1 if can_close else 0)
                if count <= 0:
                    count = 1
                out[b, t] = -math.log(count)

                if nxt == LEAF:
                    depth_stack.pop(); lo_stack.pop(); hi_stack.pop()
                else:
                    _, f_idx = tok.decode_one(nxt)
                    depth = depth_stack.pop()
                    lo_t = lo_stack.pop()
                    hi_t = hi_stack.pop()
                    pending = (depth, lo_t.clone(), hi_t.clone(), int(f_idx))
            else:
                # Threshold decision
                depth, lo_top, hi_top, f_idx = pending
                lo_f = int(lo_top[f_idx].item())
                hi_f = int(hi_top[f_idx].item())
                count = max(0, hi_f - lo_f + 1)
                if count <= 0:
                    count = 1
                out[b, t] = -math.log(count)

                _, t_idx = tok.decode_one(nxt)
                t_idx = int(t_idx)

                lo_L, hi_L = lo_top.clone(), hi_top.clone()
                hi_L[f_idx] = min(hi_L[f_idx].item(), t_idx)
                lo_R, hi_R = lo_top.clone(), hi_top.clone()
                lo_R[f_idx] = max(lo_R[f_idx].item(), t_idx + 1)

                depth_stack.append(depth + 1); lo_stack.append(lo_R); hi_stack.append(hi_R)
                depth_stack.append(depth + 1); lo_stack.append(lo_L); hi_stack.append(hi_L)

                pending = None

    return out

# ============================================================
# Optional token prior (gain bias)
# ============================================================

def create_gain_bias(df_train: pd.DataFrame,
                     feats: List[str],
                     target: str,
                     tok: "Tokenizer",
                     bins: int,
                     prior_scale: float = 0.5) -> torch.Tensor:
    # (unchanged; keep your original implementation)
    df_sample = df_train.sample(n=min(len(df_train), 200_000), random_state=42)
    X_binned_list = []
    for f in feats:
        s = df_sample[f].replace([np.inf, -np.inf], np.nan).fillna(df_sample[f].median()).values
        qs = np.linspace(0, 1, bins + 1)
        if len(np.unique(s)) > 1:
            edges = np.unique(np.quantile(s, qs))
        else:
            edges = np.array([s[0] - 1, s[0] + 1])
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
