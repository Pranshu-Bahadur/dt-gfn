from __future__ import annotations

import random
import math
import torch

from typing import Callable, List, Optional, Tuple
import torch
from collections import deque


# --------------------------------------------------------------------- #
# Helper: build a flat “heap” representation of the tree
# --------------------------------------------------------------------- #
def _build_flat_tree(
    actions: List[Tuple[str, int]],
) -> Tuple[torch.LongTensor, torch.LongTensor,
           torch.LongTensor, torch.LongTensor]:
    """
    Convert (kind, idx) action list to 4 parallel tensors describing a
    binary tree in breadth-first order. This version correctly consumes
    the token stream sequentially.

    Returns
    -------
    feat   : (S,) int64   feature index  (−1 for leaf)
    thresh : (S,) int64   threshold bin  (arbitrary for leaf)
    left   : (S,) int64   left-child index (−1 if none)
    right  : (S,) int64   right-child index (−1 if none)
    """
    if not actions:
        return (torch.tensor([-1], dtype=torch.long),) * 4

    feat, thr, L, R = [], [], [], []
    it = iter(actions)
    
    # Queue for nodes to visit in breadth-first order
    nodes_to_process = deque([0]) 
    next_node_idx = 1

    # Helper to pre-allocate space in our lists
    def grow(n: int):
        while n >= len(feat):
            feat.append(-1); thr.append(-1)
            L.append(-1);   R.append(-1)

    while nodes_to_process:
        current_node = nodes_to_process.popleft()
        grow(current_node)
        
        try:
            kind, idx = next(it)
        except StopIteration:
            # No more tokens, this branch terminates
            continue

        if kind == "feature":
            _, t = next(it)
            feat[current_node] = idx
            thr[current_node] = t
            
            L[current_node] = next_node_idx
            R[current_node] = next_node_idx + 1
            
            nodes_to_process.append(next_node_idx)
            nodes_to_process.append(next_node_idx + 1)
            next_node_idx += 2
        # No 'else' needed: leaves are handled by default values

    return (
        torch.tensor(feat, dtype=torch.long),
        torch.tensor(thr,  dtype=torch.long),
        torch.tensor(L,    dtype=torch.long),
        torch.tensor(R,    dtype=torch.long),
    )


# --------------------------------------------------------------------- #
# Main factory (This function was already correct)
# --------------------------------------------------------------------- #
def sequence_to_predictor(
    traj: List[int],
    X_binned: torch.Tensor,
    y_target: Optional[torch.Tensor],
    tokenizer,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a fast predictor from a token trajectory.
    """

    # 1. Decode tokens  → flat tensors
    actions = tokenizer.decode(traj)
    feat_arr, thr_arr, left_arr, right_arr = _build_flat_tree(actions)

    device = X_binned.device
    feat_arr  = feat_arr.to(device)
    thr_arr   = thr_arr.to(device)
    left_arr  = left_arr.to(device)
    right_arr = right_arr.to(device)

    # 2. Compute leaf values
    default_val = 0.0 if y_target is None else float(y_target.mean().item())
    leaf_val = torch.full(feat_arr.shape, default_val,
                          dtype=torch.float32, device=device)

    if y_target is not None:
        N = X_binned.size(0)
        stack = [(0, torch.arange(N, device=device))]      # (node_idx, row_indices)

        while stack:
            node, rows = stack.pop()
            if rows.numel() == 0:
                continue

            if feat_arr[node] == -1:                       # is_leaf
                if y_target is not None and rows.numel() > 0:
                    leaf_val[node] = float(y_target[rows].mean().item())
            else:                                          # is_split
                f, t = feat_arr[node].item(), thr_arr[node].item()
                mask = X_binned[rows, f] <= t
                
                left_child, right_child = left_arr[node].item(), right_arr[node].item()
                if left_child != -1:
                    stack.append((left_child,  rows[mask]))
                if right_child != -1:
                    stack.append((right_child, rows[~mask]))

    # 3. Predictor closure (fully vectorized)
    def predict(X_new: torch.Tensor) -> torch.Tensor:
        B = X_new.size(0)
        
        # node_idx tracks which node in the flat tree each sample is at.
        node_idx = torch.zeros(B, dtype=torch.long, device=X_new.device)

        # Loop until all samples have reached a leaf node.
        active_rows = torch.arange(B, device=X_new.device)
        while active_rows.numel() > 0:
            # Get current features and thresholds for all active samples
            feats_now = feat_arr[node_idx[active_rows]]
            
            # Find which samples have just arrived at a leaf
            is_leaf_mask = (feats_now == -1)
            if is_leaf_mask.any():
                # Mark these rows as done by removing them from `active_rows`
                newly_finished_rows = active_rows[is_leaf_mask]
                
                # Keep only the rows that are still traversing the tree
                active_rows = active_rows[~is_leaf_mask]
                if active_rows.numel() == 0:
                    break # All finished

            # Continue routing for non-leaf rows
            active_nodes = node_idx[active_rows]
            
            f = feat_arr[active_nodes]
            t = thr_arr[active_nodes]
            
            # This is the key: gather values from X_new using the feature
            # indices specific to each sample's current node.
            values = X_new[active_rows, f]
            go_left = values <= t
            
            # Update node_idx for each active sample based on the split
            node_idx[active_rows] = torch.where(
                go_left,
                left_arr[active_nodes],
                right_arr[active_nodes]
            )

        # All samples have reached a leaf, return the corresponding leaf values
        return leaf_val[node_idx]

    return predict








class ReplayBuffer:
    """
    Simple replay buffer storing top trajectories by reward.
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.data: List[Tuple[float, List[int], float, torch.Tensor]] = []

    def add(self, reward: float, traj: List[int], prior: float, idxs: torch.Tensor) -> None:
        """
        Add a new rollout to the buffer.
        Keeps only top-`capacity` by reward.
        """
        self.data.append((reward, traj, prior, idxs))
        # sort descending by reward
        self.data.sort(key=lambda x: x[0], reverse=True)
        if len(self.data) > self.capacity:
            self.data.pop()

    def sample(self, k: int):
        """
        Uniformly sample up to k trajectories.
        """
        return random.sample(self.data, min(k, len(self.data)))


@torch.jit.script
def tb_loss(
    log_pf: torch.Tensor,
    log_pb: torch.Tensor,
    log_z: torch.Tensor,
    R: torch.Tensor,
    prior: torch.Tensor,
) -> torch.Tensor:
    """
    Trajectory balance loss:
      (log_Z + sum log_pf - (log R + prior + sum log_pb))^2
    """
    return ((log_z + log_pf.sum(1) - (torch.log(R) + prior + log_pb.sum(1)))**2).mean()


@torch.jit.script
def fl_loss(
    logF: torch.Tensor,
    log_pf: torch.Tensor,
    log_pb: torch.Tensor,
    dE: torch.Tensor,
) -> torch.Tensor:
    """
    Flow matching loss:
      (logF_t + log_pf_t - (logF_{t+1} + log_pb_t - dE_t))^2
    """
    return ((logF[:, :-1] + log_pf - (logF[:, 1:] + log_pb - dE))**2).mean()


def _safe_sample(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> int:
    """
    Masked sampling at given temperature.
    """
    # scale logits
    scaled = logits / temperature
    # apply mask
    neg_inf = torch.tensor(-1e9, device=logits.device)
    masked = torch.where(mask, scaled, neg_inf)
    # softmax & sample
    probs = torch.softmax(masked, dim=-1)
    return torch.multinomial(probs, 1).item()


import torch
from typing import Tuple


@torch.no_grad()
def dE_split_gain(
    token_ids: torch.Tensor,          # shape (1, L) int64
    tokenizer,                        # our Simple Tokenizer
    env,                              # TabularEnv (must have .X_full, .y, .idxs)
) -> torch.Tensor:
    """
    Compute ΔE (negative split gain) for every TRANSITION in the trajectory.
    Returned tensor has shape (1, L-1) so you can align it with
    log_pf[:, :-1] when computing FL loss.

    • Only positions *immediately after* a THRESHOLD token get non-zero ΔE.
      All other positions are zero.
    • Gain =  MSE(parent) – [ w_L * MSE(L) + w_R * MSE(R) ]
    • We negate the gain so *reducing* MSE → negative energy difference.
    """
    assert token_ids.dim() == 2 and token_ids.size(0) == 1, "Expect (1, L)"
    seq = token_ids[0].tolist()[1:-1]  # strip BOS/EOS
    decoded = tokenizer.decode(seq)

    y = env.y[env.idxs]                # (B,)
    X = env.X_full[env.idxs]           # (B, F)
    n = y.numel()

    # parent statistics
    y2 = y * y
    full_mse = (y2.mean() - y.mean().pow(2)).item()

    # stacks emulate recursive traversal
    row_stack   = [torch.arange(n, device=y.device)]
    mse_stack   = [full_mse]
    dE          = torch.zeros(len(seq), device=y.device)   # per token

    it = iter(decoded)
    t_idx = -1  # running position in seq
    while True:
        try:
            kind, idx = next(it)
            t_idx += 1
        except StopIteration:
            break

        if kind == "feature":
            # consume threshold
            th_kind, th_idx = next(it); t_idx += 1
            assert th_kind == "threshold", "Feature must be followed by TH"

            parent_rows = row_stack.pop()
            parent_mse  = mse_stack.pop()

            # partition rows
            mask = X[parent_rows, idx] <= th_idx
            L_rows = parent_rows[mask]
            R_rows = parent_rows[~mask]

            def mse(rows):
                if rows.numel() < 2:
                    return 0.0
                yy = y[rows]
                return ( (yy * yy).mean() - yy.mean().pow(2) ).item()

            mse_L, mse_R = mse(L_rows), mse(R_rows)
            row_stack.extend([R_rows, L_rows])
            mse_stack.extend([mse_R, mse_L])

            wL, wR = L_rows.numel(), R_rows.numel()
            parent_N = wL + wR
            if parent_N > 0:
                gain = parent_mse - (wL / parent_N * mse_L + wR / parent_N * mse_R)
                # dE aligns with the THRESHOLD position (t_idx)
                dE[t_idx] = -gain   # negative gain = energy drop

        elif kind == "leaf":
            if row_stack:
                row_stack.pop(); mse_stack.pop()
    dE = torch.cat([torch.zeros(1, device=dE.device), dE])
    # shape (1, L-1) to match log_pf[:, :-1]
    return dE.unsqueeze(0)

