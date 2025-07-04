from __future__ import annotations

import random
import math
import torch
import heapq
from typing import Callable, List, Optional, Tuple
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
    binary tree in breadth-first order.

    Returns
    -------
    feat   : (S,) int64   feature index  (−1 for leaf)
    thresh : (S,) int64   threshold bin  (arbitrary for leaf)
    left   : (S,) int64   left-child index (−1 if none)
    right  : (S,) int64   right-child index (−1 if none)
    """
    if not actions:
        # Handle empty trajectory gracefully
        return (torch.tensor([-1], dtype=torch.long),) * 4

    feat, thr, L, R = [-1], [-1], [-1], [-1]
    nodes_to_process = deque([0])
    next_node_idx = 1
    it = iter(actions)

    def grow(n: int):
        # Pre-allocate space in our lists if needed
        while n >= len(feat):
            feat.extend([-1, -1]); thr.extend([-1, -1])
            L.extend([-1, -1]);   R.extend([-1, -1])

    while nodes_to_process:
        current_node_idx = nodes_to_process.popleft()
        
        try:
            kind, idx = next(it)
        except StopIteration:
            continue # No more actions, this is a leaf

        if kind == "feature":
            try:
                # A feature must be followed by a threshold
                th_kind, th_idx = next(it)
                assert th_kind == "threshold"
            except (StopIteration, AssertionError):
                # Incomplete action, treat as a leaf
                continue

            grow(next_node_idx + 1)
            feat[current_node_idx] = idx
            thr[current_node_idx] = th_idx
            L[current_node_idx] = next_node_idx
            R[current_node_idx] = next_node_idx + 1

            nodes_to_process.append(next_node_idx)
            nodes_to_process.append(next_node_idx + 1)
            next_node_idx += 2

    return (
        torch.tensor(feat, dtype=torch.long),
        torch.tensor(thr,  dtype=torch.long),
        torch.tensor(L,    dtype=torch.long),
        torch.tensor(R,    dtype=torch.long),
    )


# --------------------------------------------------------------------- #
# Main factory for creating a tree predictor
# --------------------------------------------------------------------- #
def sequence_to_predictor(
    traj: List[int],
    X_binned: torch.Tensor,
    y_target: Optional[torch.Tensor],
    tokenizer,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a fast, vectorized predictor from a token trajectory.
    """
    actions = tokenizer.decode(traj)
    feat, thr, left, right = _build_flat_tree(actions)

    device = X_binned.device
    feat, thr, left, right = feat.to(device), thr.to(device), left.to(device), right.to(device)

    # Compute leaf values by propagating data down the tree
    default_val = 0.0 if y_target is None else y_target.mean().item()
    leaf_values = torch.full(feat.shape, default_val, dtype=torch.float32, device=device)

    if y_target is not None and feat[0] != -1:
        N = X_binned.size(0)
        node_indices = torch.arange(N, device=device)
        stack = [(0, node_indices)] # (tree_node_idx, data_row_indices)

        while stack:
            node_idx, row_indices = stack.pop()
            if row_indices.numel() == 0:
                continue

            if feat[node_idx] == -1: # Is a leaf node
                leaf_values[node_idx] = y_target[row_indices].mean().item()
            else: # Is a split node
                f, t = feat[node_idx].item(), thr[node_idx].item()
                mask = X_binned[row_indices, f] <= t
                
                left_child, right_child = left[node_idx].item(), right[node_idx].item()
                if left_child != -1:
                    stack.append((left_child, row_indices[mask]))
                if right_child != -1:
                    stack.append((right_child, row_indices[~mask]))

    @torch.no_grad() # Ensure no gradients are computed during prediction
    def predict(X_new: torch.Tensor) -> torch.Tensor:
        B = X_new.size(0)
        node_idx = torch.zeros(B, dtype=torch.long, device=device)
        active_mask = torch.ones(B, dtype=torch.bool, device=device)

        # Loop until all samples have reached a leaf
        while active_mask.any():
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            current_nodes = node_idx[active_indices]
            
            # Check which are leaves
            is_leaf_mask = (feat[current_nodes] == -1)
            # Update the main active mask to deactivate leaf rows
            active_mask[active_indices[is_leaf_mask]] = False

            # For non-leaf nodes, decide which child to visit next
            split_mask = ~is_leaf_mask
            if not split_mask.any():
                break # All have reached leaves
            
            active_split_indices = active_indices[split_mask]
            active_split_nodes = node_idx[active_split_indices]
            
            f = feat[active_split_nodes]
            t = thr[active_split_nodes]

            values = X_new.gather(1, f.unsqueeze(1)).squeeze(1)
            go_left = values <= t

            node_idx[active_split_indices] = torch.where(
                go_left,
                left[active_split_nodes],
                right[active_split_nodes]
            )
        
        return leaf_values[node_idx]

    return predict

# --------------------------------------------------------------------- #
# Exploration Strategy
# --------------------------------------------------------------------- #
class PrioritizedReplayBuffer:
    """
    A memory-efficient prioritized replay buffer that stores trajectories
    based on reward, using a min-heap. Only lightweight Python types are stored.
    """
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer = []
        self._counter = 0

    def add(self, reward: float, traj: List[int], prior: float, idxs: Optional[torch.Tensor] = None) -> None:
        """Saves a trajectory to the buffer."""
        priority = -reward # Use negative reward for max-heap behavior with a min-heap
        
        # Store only lightweight data, not tensors
        data = (reward, traj, prior, None) 
        
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, (priority, self._counter, data))
        else:
            # Replace the smallest-reward element
            heapq.heappushpop(self.buffer, (priority, self._counter, data))
        self._counter += 1

    def sample(self, k: int) -> List[Tuple[float, List[int], float, None]]:
        """Samples k trajectories with the highest rewards."""
        k = min(k, len(self.buffer))
        return [item[2] for item in heapq.nlargest(k, self.buffer)]

    def __len__(self) -> int:
        return len(self.buffer)


class ExplorationScheduler:
    """
    Calculates an exponentially decaying exploration rate (ε).
    """
    def __init__(self, initial_epsilon: float, min_epsilon: float, decay_rate: float):
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def get_epsilon(self, current_step: int) -> float:
        """Calculates the current exploration rate."""
        return self.min_epsilon + \
            (self.initial_epsilon - self.min_epsilon) * \
            math.exp(-1. * current_step * self.decay_rate)

# --------------------------------------------------------------------- #
# Loss Functions
# --------------------------------------------------------------------- #
@torch.jit.script
def tb_loss(
    log_pf: torch.Tensor, log_pb: torch.Tensor, log_z: torch.Tensor,
    R: torch.Tensor, prior: torch.Tensor,
) -> torch.Tensor:
    """Trajectory Balance Loss, batched."""
    # Ensure tensors have the correct shape for broadcasting
    if R.dim() == 1: R = R.unsqueeze(1)
    if prior.dim() == 1: prior = prior.unsqueeze(1)

    traj_p_fwd = log_pf.sum(1, keepdim=True)
    traj_p_bwd = log_pb.sum(1, keepdim=True)
    
    return ((log_z + traj_p_fwd - (torch.log(R) + prior + traj_p_bwd))**2).mean()

@torch.jit.script
def fl_loss(
    logF: torch.Tensor, log_pf: torch.Tensor,
    log_pb: torch.Tensor, dE: torch.Tensor,
) -> torch.Tensor:
    """Flow Matching Loss, batched."""
    return ((logF[:, :-1] + log_pf - (logF[:, 1:] + log_pb - dE))**2).mean()

def _safe_sample(logits: torch.Tensor, mask: torch.Tensor, temperature: float) -> int:
    """Masked, temperature-controlled sampling from logits."""
    scaled_logits = logits / temperature
    masked_logits = torch.where(mask, scaled_logits, torch.tensor(-1e9, device=logits.device))
    probs = torch.softmax(masked_logits, dim=-1)
    return torch.multinomial(probs, 1).item()

# --------------------------------------------------------------------- #
# Energy Change Calculation
# --------------------------------------------------------------------- #
@torch.no_grad()
def dE_split_gain(
    token_ids: torch.Tensor, tokenizer, env
) -> torch.Tensor:
    """
    Compute ΔE (negative split gain) for every transition in a batch of trajectories.
    This version is designed to be more robust and clear.
    """
    batch_size, max_len = token_ids.shape
    dE = torch.zeros(batch_size, max_len - 1, device=token_ids.device)

    for i in range(batch_size):
        seq = token_ids[i].tolist()
        # Find the actual end of the sequence (before padding)
        try:
            end_idx = seq.index(tokenizer.EOS)
        except ValueError:
            end_idx = max_len
        
        decoded_actions = tokenizer.decode(seq[1:end_idx])

        if not decoded_actions: continue

        y = env.y[env.idxs]
        X = env.X_full[env.idxs]
        n_rows = y.numel()

        def get_mse(rows):
            if rows.numel() < 2: return 0.0
            subset_y = y[rows]
            return (subset_y.var(unbiased=False)).item()

        parent_mse = get_mse(torch.arange(n_rows, device=y.device))
        
        # Stacks for traversing the tree being built
        row_indices_stack = [torch.arange(n_rows, device=y.device)]
        mse_stack = [parent_mse]
        action_idx = 0

        it = iter(decoded_actions)
        while action_idx < len(decoded_actions):
            kind, idx = decoded_actions[action_idx]
            
            if kind == "feature":
                if not row_indices_stack: break
                
                parent_rows = row_indices_stack.pop()
                parent_mse = mse_stack.pop()
                
                # Consume the subsequent threshold token
                action_idx += 1
                if action_idx >= len(decoded_actions): break
                th_kind, th_val = decoded_actions[action_idx]
                if th_kind != "threshold": continue

                # Partition data
                mask = X[parent_rows, idx] <= th_val
                L_rows = parent_rows[mask]
                R_rows = parent_rows[~mask]

                mse_L = get_mse(L_rows)
                mse_R = get_mse(R_rows)

                # Push children's data and MSE onto stacks for future splits
                # Note: Order is R, L because pop() will get L first
                row_indices_stack.extend([R_rows, L_rows])
                mse_stack.extend([mse_R, mse_L])

                wL, wR = L_rows.numel(), R_rows.numel()
                parent_N = wL + wR
                if parent_N > 0:
                    gain = parent_mse - (wL / parent_N * mse_L + wR / parent_N * mse_R)
                    # The energy drop corresponds to the transition *ending* with the threshold token
                    dE[i, action_idx] = -gain # Negative gain = energy drop
            
            elif kind == "leaf":
                if row_indices_stack:
                    row_indices_stack.pop()
                    mse_stack.pop()
            
            action_idx += 1
    
    return dE