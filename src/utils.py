# src/utils.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-9

# =========================
# Replay Buffer (Top-K)
# =========================
class ReplayBuffer:
    """
    Stores tuples (reward, seq, prior, idxs).
    Kept sorted by reward (desc) up to capacity.
    """
    def __init__(self, capacity: int = 200):
        self.capacity = int(capacity)
        self.data: List[Tuple[float, List[int], float, torch.Tensor]] = []
        self.step: int = 0

    def add(self, reward: float, seq: List[int], prior: float, idxs: torch.Tensor):
        self.data.append((float(reward), list(seq), float(prior), idxs.clone()))
        # keep best at front
        self.data.sort(key=lambda x: x[0], reverse=True)
        if len(self.data) > self.capacity:
            self.data = self.data[: self.capacity]

    def mark_policy_update(self):
        self.step += 1


# =========================
# Loss Functions (TorchScript-safe)
# =========================
@torch.jit.script
def tb_loss(log_pf: torch.Tensor, log_pb: torch.Tensor, log_z: torch.Tensor,
            R: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """
    Trajectory Balance loss:
      (log Z + Σ_t log P_f(a_t|s_t) - (log R + prior + Σ_t log P_b(a_t|s_{t+1})))^2
    Shapes:
      log_pf: [B, T]        (masked/padded sums OK)
      log_pb: [B, T]
      log_z:  scalar param
      R:      [B]
      prior:  [B]
    """
    eps = 1e-9
    sum_pf = log_pf.sum(1)
    sum_pb = log_pb.sum(1)
    rhs = torch.log(R.clamp_min(eps)) + sum_pb
    diff = log_z + sum_pf - rhs
    return (diff * diff).mean()


@torch.jit.script
def fl_loss(logF: torch.Tensor, log_pf: torch.Tensor, log_pb: torch.Tensor,
            dR: torch.Tensor) -> torch.Tensor:
    """
    Flow-matching / detailed balance surrogate for shaped per-step rewards.
    We regress: logF[:, :-1] + log_pf - log_pb  ≈  log dR
    """
    eps = 1e-9
    Tm1 = min(logF.size(1) - 1, log_pf.size(1), log_pb.size(1), dR.size(1))
    lhs = logF[:, :Tm1] + log_pf[:, :Tm1] - log_pb[:, :Tm1]
    rhs = torch.log(dR[:, :Tm1].clamp_min(eps))
    return ((lhs - rhs) ** 2).mean()


# =========================
# Safe sampler (ε-greedy)
# =========================
def _safe_sample(logits: torch.Tensor,
                 mask: torch.Tensor,
                 temperature: float = 1.0,
                 epsilon: float = 0.0) -> torch.Tensor:
    """
    Apply mask and temperature, then sample.
    With prob ε, sample uniformly over valid actions (ε-greedy).
    If a row has all tokens masked, fall back to uniform over all (rare).
    """
    # mask invalid
    masked_logits = logits.masked_fill(~mask, -float("inf"))

    # fix rows with all -inf
    all_masked = (~mask).all(dim=-1)
    if all_masked.any():
        # replace with zeros and allow all
        masked_logits[all_masked] = 0.0
        mask = mask.clone()
        mask[all_masked] = True

    if temperature is not None and temperature > EPS:
        masked_logits = masked_logits / float(temperature)

    probs = torch.softmax(masked_logits, dim=-1)

    if epsilon is not None and epsilon > 0.0:
        valid_counts = mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform = mask.float() / valid_counts
        probs = (1.0 - float(epsilon)) * probs + float(epsilon) * uniform

    # sample
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# =========================
# Tree predictor (robust)
# =========================
def get_tree_predictor(
    traj: List[int],
    X_binned: torch.Tensor,
    y_target: torch.Tensor,
    tok: "Tokenizer",
    *,
    min_child_size: int = 1,
    min_gain: float = 0.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a predictor function from a trajectory.
    Strategy:
      • Decode tokens (feat, th, leaf) into a binary tree structure.
      • For each leaf, compute the mean of y_target on training rows routed to that leaf.
      • Returned fn routes new X and outputs the stored leaf mean.

    Works for both regression (y: [N]) and classification/residuals (y: [N, C]).
    """
    device = X_binned.device
    v = tok.v

    # -------- build a generic tree from tokens ----------
    actions = tok.decode(traj[1:-1])  # skip BOS/EOS
    it = iter(actions)

    def build():
        try:
            kind, idx = next(it)
        except StopIteration:
            return {"type": "leaf"}
        if kind == "feat":
            # next token must be a threshold
            try:
                kind2, t = next(it)
            except StopIteration:
                return {"type": "leaf"}
            if kind2 not in ("th", "thr", "threshold"):
                # unexpected token -> stop
                return {"type": "leaf"}
            return {"type": "split", "f": int(idx), "t": int(t), "L": build(), "R": build()}
        else:
            # explicit leaf token or anything else -> leaf
            return {"type": "leaf"}

    tree = build()

    # -------- index routing helpers ----------
    def route_indices(node, idxs: torch.Tensor):
        if node["type"] == "leaf":
            return [idxs]
        f, t = node["f"], node["t"]
        col = X_binned[idxs, f]
        L_mask = (col <= t)
        R_mask = ~L_mask
        idxs_L = idxs[L_mask]
        idxs_R = idxs[R_mask]

        # if children too small, stop splitting
        if idxs_L.numel() < min_child_size or idxs_R.numel() < min_child_size:
            return [idxs]
        leaves = []
        for child, ci in (("L", idxs_L), ("R", idxs_R)):
            leaves.extend(route_indices(node[child], ci))
        return leaves

    all_idx = torch.arange(X_binned.size(0), device=device)
    leaf_indices = route_indices(tree, all_idx)

    # -------- compute leaf outputs ----------
    y = y_target.to(device).float()
    if y.dim() == 1:
        out_dim = 1
    else:
        out_dim = y.size(1)

    leaf_values: List[torch.Tensor] = []
    for idxs in leaf_indices:
        if idxs.numel() == 0:
            mu = y.mean(dim=0, keepdim=True)
        else:
            mu = y[idxs].mean(dim=0, keepdim=True)
        leaf_values.append(mu)
    if not leaf_values:
        leaf_values = [y.mean(dim=0, keepdim=True)]
    leaf_values = torch.cat(leaf_values, dim=0)  # [L, out_dim or 1]

    # -------- route a new batch ----------
    def predict(X_new: torch.Tensor) -> torch.Tensor:
        device2 = X_new.device
        N = X_new.size(0)
        if out_dim == 1:
            out = torch.empty((N,), device=device2, dtype=torch.float32)
        else:
            out = torch.empty((N, out_dim), device=device2, dtype=torch.float32)

        # assign rows to leaves in one pass
        def assign(node, idxs: torch.Tensor, acc: List[Tuple[torch.Tensor, int]]):
            if node["type"] == "leaf":
                acc.append((idxs, len(acc)))
                return
            f, t = node["f"], node["t"]
            col = X_new[idxs, f]
            L_mask = (col <= t)
            R_mask = ~L_mask
            idxs_L = idxs[L_mask]
            idxs_R = idxs[R_mask]
            # if split invalid at inference, send all to this leaf
            if idxs_L.numel() < min_child_size or idxs_R.numel() < min_child_size:
                acc.append((idxs, len(acc)))
                return
            assign(node["L"], idxs_L, acc)
            assign(node["R"], idxs_R, acc)

        acc: List[Tuple[torch.Tensor, int]] = []
        assign(tree, torch.arange(N, device=device2), acc)
        # write outputs
        for idxs, leaf_id in acc:
            val = leaf_values[min(leaf_id, leaf_values.size(0) - 1)]
            if out_dim == 1:
                out[idxs] = val.squeeze()
            else:
                out[idxs] = val
        return out

    return predict


# =========================
# Split-gain placeholders (for non-bayesian rewards)
# =========================
def _variance(y: torch.Tensor) -> torch.Tensor:
    y = y.float()
    if y.numel() == 0:
        return torch.tensor(0.0, device=y.device)
    return y.var(unbiased=False)


@torch.no_grad()
def deltaE_split_gain_regression(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """
    Returns a vector of per-decision variance reductions (approx).
    Used only if reward_function == 'variance'.
    """
    X, y = env.X_full, env.y_full.float()
    seq = tokens[0].tolist()
    actions = tok.decode(seq[1:-1])
    it = iter(actions)
    gains: List[float] = []
    idxs = env.idxs
    try:
        while True:
            kind, f = next(it)
            if kind != "feat":
                gains.append(0.0)
                continue
            _, t = next(it)
            col = X[idxs, f]
            L = idxs[col <= t]
            R = idxs[col > t]
            before = _variance(y[idxs])
            after = (_variance(y[L]) * (L.numel() / max(1, idxs.numel()))
                     + _variance(y[R]) * (R.numel() / max(1, idxs.numel())))
            gains.append(float((before - after).clamp_min(0.0).item()))
            # continue down the larger side (approximate per-step)
            idxs = L if L.numel() >= R.numel() else R
    except StopIteration:
        pass
    if not gains:
        gains = [0.0]
    return torch.tensor(gains, device=X.device, dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def deltaE_split_gain_classification(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """
    Returns a vector of per-decision Gini reductions (approx).
    Used only if reward_function == 'gini'.
    """
    X, y = env.X_full, env.y_full.long()
    K = int(getattr(env, "n_classes", int(y.max().item()) + 1))
    seq = tokens[0].tolist()
    actions = tok.decode(seq[1:-1])
    it = iter(actions)
    gains: List[float] = []
    idxs = env.idxs
    try:
        while True:
            kind, f = next(it)
            if kind != "feat":
                gains.append(0.0)
                continue
            _, t = next(it)
            col = X[idxs, f]
            L = idxs[col <= t]
            R = idxs[col > t]
            def gini(idxs_):
                if idxs_.numel() == 0:
                    return 0.0
                counts = torch.bincount(y[idxs_], minlength=K).float()
                p = counts / counts.sum().clamp_min(1.0)
                return float((1.0 - (p * p).sum()).item())
            before = gini(idxs)
            after = (gini(L) * (L.numel() / max(1, idxs.numel()))
                     + gini(R) * (R.numel() / max(1, idxs.numel())))
            gains.append(max(0.0, before - after))
            idxs = L if L.numel() >= R.numel() else R
    except StopIteration:
        pass
    if not gains:
        gains = [0.0]
    return torch.tensor(gains, device=X.device, dtype=torch.float32).unsqueeze(0)


# =========================
# Bayesian rewards (no /N scaling)
# =========================
@torch.no_grad()
def calculate_bayesian_reward(tokens: torch.Tensor,
                              tok: "Tokenizer",
                              env: "TabularEnv",
                              beta: float) -> torch.Tensor:
    """
    Classification reward = Dirichlet–Multinomial evidence difference + structure prior:
      log R = [Σ_leaves log P(y_leaf | α) - log P(y_root | α)]  -  β * (#splits)
    """
    # Build leaves over GLOBAL indices (root starts at env.idxs)
    root, n_dec = _build_tree_by_data(tokens, tok, env.X_full, env.idxs)
    leaves = _collect_leaf_indices(root)

    # Work with the full label tensor, not pre-subsetted
    y_full = env.y_full.to(torch.long)
    if env.n_classes is not None:
        K = int(env.n_classes)
    else:
        # safe fallback
        K = int(y_full.max().item()) + 1

    if env.idxs.numel() == 0:
        return torch.tensor([1e-9], device=env.device)

    # Root evidence on the ROOT subset (env.idxs)
    root_counts = torch.bincount(y_full[env.idxs], minlength=K).to(torch.float64)
    alpha = torch.full((K,), 0.1, dtype=torch.float64, device=env.device)
    L0 = _dm_log_evidence_from_counts(root_counts, alpha)

    # Sum leaf evidences (each leaf idx is a subset of GLOBAL indices)
    L = torch.zeros((), dtype=torch.float64, device=env.device)
    any_leaf = False
    for idx in leaves:
        if idx.numel() == 0:
            continue
        any_leaf = True
        leaf_counts = torch.bincount(y_full[idx], minlength=K).to(torch.float64)
        L = L + _dm_log_evidence_from_counts(leaf_counts, alpha)
    if not any_leaf:
        L, n_dec = L0, 0

    logR = (L - L0) - float(beta) * n_dec
    logR = torch.clamp(logR, min=-50.0, max=50.0).to(torch.float32)
    return torch.exp(logR).clamp_min(1e-9).unsqueeze(0)


@torch.no_grad()
def calculate_bayesian_reward_regression(tokens: torch.Tensor,
                                         tok: "Tokenizer",
                                         env: "TabularEnv",
                                         beta: float) -> torch.Tensor:
    """
    Regression reward = Normal–Inverse–Gamma evidence difference + structure prior:
      log R = [Σ_leaves log P(y_leaf | μ0,κ0,a0,b0) - log P(y_root | ...)]  -  β * (#splits)
    """
    root, n_dec = _build_tree_by_data(tokens, tok, env.X_full, env.idxs)
    leaves = _collect_leaf_indices(root)

    y_full = env.y_full.to(torch.float64)
    if env.idxs.numel() == 0:
        return torch.tensor([1e-9], device=env.device)

    # Root evidence on ROOT subset
    L0 = _nig_log_evidence(y_full[env.idxs])

    # Sum leaf evidences (GLOBAL indices)
    L = torch.zeros((), dtype=torch.float64, device=y_full.device)
    any_leaf = False
    for idx in leaves:
        if idx.numel() == 0:
            continue
        any_leaf = True
        L = L + _nig_log_evidence(y_full[idx])
    if not any_leaf:
        L, n_dec = L0, 0

    logR = (L - L0) - float(beta) * n_dec
    logR = torch.clamp(logR, min=-50.0, max=50.0).to(torch.float32)
    return torch.exp(logR).clamp_min(1e-9).unsqueeze(0)



# =========================
# Helpers for rewards
# =========================
def _dm_log_evidence_from_counts(counts: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Dirichlet-Multinomial log evidence for counts given concentration alpha.
    """
    from torch.special import gammaln
    N = counts.sum()
    A = alpha.sum()
    return (gammaln(A) - gammaln(N + A)
            + gammaln(counts + alpha).sum() - gammaln(alpha).sum())


def _nig_log_evidence(y: torch.Tensor) -> torch.Tensor:
    """
    Normal-Inverse-Gamma marginal likelihood log P(y | μ0, κ0, a0, b0)
    using conjugate prior with mild regularization.
    """
    from torch.special import gammaln
    n = y.numel()
    if n == 0:
        return torch.tensor(0.0, dtype=y.dtype, device=y.device)
    y_mean = y.mean()
    sse = ((y - y_mean) ** 2).sum()

    # Prior (mild)
    mu0 = torch.tensor(0.0, dtype=y.dtype, device=y.device)
    k0  = torch.tensor(1e-2, dtype=y.dtype, device=y.device)
    a0  = torch.tensor(1.0, dtype=y.dtype, device=y.device)
    b0  = torch.tensor(1.0, dtype=y.dtype, device=y.device)

    kn = k0 + n
    an = a0 + n / 2.0
    bn = b0 + 0.5 * sse + (k0 * n) * (y_mean - mu0) ** 2 / (2.0 * kn)

    out = (gammaln(an) - gammaln(a0)
           + 0.5 * (torch.log(k0) - torch.log(kn))
           + a0 * torch.log(b0) - an * torch.log(bn)
           - (n / 2.0) * math.log(math.pi))
    return out


def _build_tree_by_data(tokens: torch.Tensor,
                        tok: "Tokenizer",
                        X_full: torch.Tensor,
                        idxs: torch.Tensor) -> Tuple[Dict, int]:
    """
    Build a binary tree structure from tokens and attach the subset indices to each node.
    Returns (root_node, n_decision_nodes actually split into non-empty children).
    """
    seq = tokens[0].tolist()
    actions = tok.decode(seq[1:-1])
    it = iter(actions)

    def build(local_idxs):
        try:
            kind, idx = next(it)
        except StopIteration:
            return {"type": "leaf", "idxs": local_idxs}, 0

        if kind != "feat":
            return {"type": "leaf", "idxs": local_idxs}, 0

        # threshold must follow
        try:
            kind2, t = next(it)
        except StopIteration:
            return {"type": "leaf", "idxs": local_idxs}, 0

        col = X_full[local_idxs, int(idx)]
        L = local_idxs[col <= int(t)]
        R = local_idxs[col >  int(t)]
        if L.numel() == 0 or R.numel() == 0:
            return {"type": "leaf", "idxs": local_idxs}, 0

        L_node, L_cnt = build(L)
        R_node, R_cnt = build(R)
        return {"type": "split", "f": int(idx), "t": int(t), "L": L_node, "R": R_node, "idxs": local_idxs}, (1 + L_cnt + R_cnt)

    root, n_dec = build(idxs.clone())
    return root, int(n_dec)


def _collect_leaf_indices(root: Dict) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    def dfs(node):
        if node["type"] == "leaf":
            out.append(node["idxs"])
            return
        dfs(node["L"]); dfs(node["R"])
    dfs(root)
    return out
