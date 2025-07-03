from typing import List, Callable, Optional, Tuple
import random

import torch


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


