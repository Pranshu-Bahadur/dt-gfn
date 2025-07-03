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
    Turn a token-ID trajectory into a callable predictor.

    • If `y_target` is provided (training-time) each leaf = mean(y_target[rows]).
    • If `y_target` is None (inference) or a branch receives zero rows,
      the leaf takes `default_val` (global mean if available, else 0.0).
    """

    # ------------------------------------------------------------------ #
    # 1. Decode token sequence → tree dictionary
    # ------------------------------------------------------------------ #
    actions = tokenizer.decode(traj)          # skip PAD/BOS/EOS
    it = iter(actions)

    def build():
        try:
            kind, idx = next(it)
        except StopIteration:
            return None
        if kind == "feature":
            # next token must be TH_*
            _, th = next(it)
            return dict(type="split", f=idx, t=th, L=build(), R=build())
        return dict(type="leaf", value=None)

    tree = build()
    if tree is None:
        return lambda X: torch.zeros(X.size(0), dtype=torch.float32, device=X.device)

    # ------------------------------------------------------------------ #
    # 2. Assign numeric values to every leaf
    # ------------------------------------------------------------------ #
    N = X_binned.size(0)
    default_val: float = (
        0.0 if y_target is None else float(y_target.mean().item())
    )

    stack = [(tree, torch.arange(N, device=X_binned.device))]
    while stack:
        node, rows = stack.pop()
        if node is None:
            continue

        if node["type"] == "split":
            f, t = node["f"], node["t"]
            mask = X_binned[rows, f] <= t
            stack.append((node["L"], rows[mask]))
            stack.append((node["R"], rows[~mask]))
        else:  # leaf
            if y_target is not None and rows.numel() > 0:
                node["value"] = float(y_target[rows].mean().item())
            else:
                node["value"] = default_val

    # ------------------------------------------------------------------ #
    # 3. Predictor closure (works on CPU or CUDA tensors)
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
                # final guard: if value somehow still None, use default_val
                value = default_val if node["value"] is None else node["value"]
                out[rows] = value
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

