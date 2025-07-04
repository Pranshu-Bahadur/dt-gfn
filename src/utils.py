from __future__ import annotations

import random
import math
import torch
import heapq
from typing import Callable, List, Optional, Tuple
from collections import deque

def sequence_to_predictor(
    traj: List[int],
    X_binned: torch.Tensor,
    y_target: Optional[torch.Tensor],
    tokenizer,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Decode a token-ID trajectory into a tree predictor function.

    Args:
        traj: Sequence of token IDs including BOS/EOS.
        X_binned: Tensor of shape (N, F) with binned features.
        y_target: Tensor of shape (N,) with true targets, or None for inference.
        tokenizer: SimpleTokenizer with decode() method.

    Returns:
        A function f(X_new) -> Tensor of shape (X_new.size(0),)
    """
    # Decode trajectory into (kind, idx) pairs, skipping special tokens.
    actions = tokenizer.decode(traj)
    it = iter(actions)

    # Recursive tree builder
    def build_node():
        try:
            kind, idx = next(it)
        except StopIteration:
            return None
        if kind == "feature":
            # Next action must be threshold
            _, th = next(it)
            left = build_node()
            right = build_node()
            return {"type": "split", "f": idx, "t": th, "L": left, "R": right}
        # Leaf node
        return {"type": "leaf", "value": None}

    tree = build_node()
    if tree is None:
        # Return zero predictor
        return lambda X: torch.zeros(X.size(0), dtype=torch.float32, device=X.device)

    # Assign leaf values based on y_target
    N = X_binned.size(0)
    default_val = 0.0 if y_target is None else float(y_target.mean().item())

    stack = [(tree, torch.arange(N, device=X_binned.device))]
    while stack:
        node, idxs = stack.pop()
        if not idxs.numel() or node is None:
            continue
        if node["type"] == "split":
            f, t = node["f"], node["t"]
            mask = X_binned[idxs, f] <= t
            left_idxs = idxs[mask]
            right_idxs = idxs[~mask]
            if node.get("L"):
                stack.append((node["L"], left_idxs))
            if node.get("R"):
                stack.append((node["R"], right_idxs))
        else:
            # Leaf: set value
            if y_target is not None and idxs.numel() > 0:
                node["value"] = float(y_target[idxs].mean().item())

    # --- FIX STARTS HERE ---

    # 1. Fill any unassigned leaf values with the default to prevent errors
    def fill_missing_leaves(node):
        if node is None:
            return
        if node["type"] == "leaf":
            if node["value"] is None:
                node["value"] = default_val
        elif node["type"] == "split":
            fill_missing_leaves(node.get("L"))
            fill_missing_leaves(node.get("R"))

    fill_missing_leaves(tree)

    # Predictor function for new X
    def predict(X_new: torch.Tensor) -> torch.Tensor:
        M = X_new.size(0)
        # 2. Initialize output tensor with a default value instead of garbage
        out = torch.full((M,), default_val, dtype=torch.float32, device=X_new.device)
        
        stack2 = [(tree, torch.arange(M, device=X_new.device))]
        while stack2:
            node, idxs = stack2.pop()
            if not idxs.numel() or node is None:
                continue
            if node["type"] == "leaf":
                out[idxs] = node["value"]
            else:
                f, t = node["f"], node["t"]
                mask = X_new[idxs, f] <= t
                left_idxs = idxs[mask]
                right_idxs = idxs[~mask]
                if node.get("L"):
                    stack2.append((node["L"], left_idxs))
                if node.get("R"):
                    stack2.append((node["R"], right_idxs))
        return out

    # --- FIX ENDS HERE ---

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