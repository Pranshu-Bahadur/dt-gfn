from typing import List, Callable, Optional, Tuple
import random

import torch

from typing import Callable, List, Optional
import torch


def sequence_to_predictor(
    traj: List[int],
    X_binned: torch.Tensor,
    y_target: Optional[torch.Tensor],
    tokenizer,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Decode a token-ID trajectory into a callable predictor.

    • If `y_target` is given (training-time) each leaf = mean(y_target[rows]).
    • At inference (`y_target is None`) unused / empty leaves get `default_val`
      (the global mean seen at training time, passed in earlier).
    """

    # ------------------------------------------------------------------ #
    # 1.  Parse trajectory -> tree dict
    # ------------------------------------------------------------------ #
    actions = tokenizer.decode(traj)           # skip PAD/BOS/EOS
    it = iter(actions)

    def build():
        try:
            kind, idx = next(it)
        except StopIteration:
            return None
        if kind == "feature":
            _, th = next(it)                   # consume TH_*
            return dict(type="split", f=idx, t=th, L=build(), R=build())
        return dict(type="leaf", value=None)

    tree = build()
    if tree is None:      # degenerate → constant zero predictor
        return lambda X: torch.zeros(X.size(0), dtype=torch.float32, device=X.device)

    # ------------------------------------------------------------------ #
    # 2.  Assign leaf values  (single DFS)
    # ------------------------------------------------------------------ #
    N = X_binned.size(0)
    default_val = 0.0 if y_target is None else float(y_target.mean().item())

    stack = [(tree, torch.arange(N, device=X_binned.device))]
    while stack:
        node, rows = stack.pop()
        if node is None or rows.numel() == 0:
            # Nothing reached this branch → set default later
            if node and node["type"] == "leaf":
                node["value"] = default_val
            continue

        if node["type"] == "split":
            f, t = node["f"], node["t"]
            mask = X_binned[rows, f] <= t
            stack.append((node["L"], rows[mask]))
            stack.append((node["R"], rows[~mask]))
        else:                              # leaf
            if y_target is not None:
                node["value"] = float(y_target[rows].mean().item())
            else:
                node["value"] = default_val

    # ------------------------------------------------------------------ #
    # 3.  Predictor closure
    # ------------------------------------------------------------------ #
    def predict(X_new: torch.Tensor) -> torch.Tensor:
        M = X_new.size(0)
        out = torch.full((M,), default_val, dtype=torch.float32, device=X_new.device)

        stack2 = [(tree, torch.arange(M, device=X_new.device))]
        while stack2:
            node, rows = stack2.pop()
            if node is None or rows.numel() == 0:
                continue
            if node["type"] == "leaf":
                out[rows] = node["value"]
            else:
                f, t = node["f"], node["t"]
                mask = X_new[rows, f] <= t
                stack2.append((node["L"], rows[mask]))
                stack2.append((node["R"], rows[~mask]))
        return out

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


