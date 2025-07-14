from __future__ import annotations

import random
import math
from collections import deque
from typing import Callable, List, Optional, Tuple, Deque

import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F

# --- Loss Functions ---

@torch.jit.script
def tb_loss(log_pf: torch.Tensor, log_pb: torch.Tensor, log_z: torch.Tensor, R: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Trajectory Balance (TB) loss, batched.
    The objective is to match the forward and backward probabilities of generating a trajectory,
    scaled by the reward and prior.
    """
    # R is the total reward for the trajectory (a scalar tensor)
    # log_pf and log_pb are sums of log probabilities over the trajectory's actions
    loss = (log_z + log_pf.sum(1) - (torch.log(R) + prior + log_pb.sum(1)))**2
    return loss.mean()

@torch.jit.script
def fl_loss(logF: torch.Tensor, log_pf: torch.Tensor, log_pb: torch.Tensor, dR: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Flow-Matching (FM) or Detailed Balance loss.
    This loss aligns the single-step forward and backward transition probabilities,
    using the intermediate rewards (dR) for each step.
    """
    # logF represents the log flow (log F(s)) at each state in the trajectory
    # log_pf and log_pb are per-step log probabilities
    # dR is the tensor of intermediate rewards for each transition
    loss = (logF[:, :-1] + log_pf - (logF[:, 1:] + log_pb + dR))**2
    return loss.mean()

# --- Tree & Reward Utilities ---

def deltaE_split_gain(tokens: torch.Tensor, tok: "Tokenizer", env: "TabularEnv") -> torch.Tensor:
    """
    Computes the reward (ΔR) as the MSE reduction (split gain) for each split
    action in a trajectory. Higher gain is a better reward.
    """
    y: torch.Tensor = env.y[env.idxs]
    N: int = y.numel()
    dR: torch.Tensor = torch.zeros(tokens.shape[1] - 1, device=y.device)

    # Baseline MSE for the entire current data subset (parent node)
    full_mse = (y.mul(y).mean() - y.mean()**2).item() if N > 1 else 0.0

    # Stacks to track data indices and MSE during tree traversal
    stack_rows: Deque[torch.Tensor] = deque([torch.arange(N, device=y.device)])
    stack_mse: Deque[float] = deque([full_mse])

    action_sequence: List[int] = tokens[0, 1:-1].tolist()
    it = iter(tok.decode(action_sequence))
    token_idx = 0

    for kind, idx in it:
        if kind == "feat":
            try:
                # Expect a threshold token to follow a feature token
                _, th = next(it)
            except StopIteration:
                break # Trajectory ended unexpectedly

            if not stack_rows: continue
            parent_rows = stack_rows.pop()
            parent_mse = stack_mse.pop()

            # Partition the data based on the feature split
            fv = env.X_full[env.idxs[parent_rows], idx]
            mask = fv <= th
            L_rows, R_rows = parent_rows[mask], parent_rows[~mask]

            def mse(rows: torch.Tensor) -> float:
                if rows.numel() < 2: return 0.0
                yy = y[rows]
                return ((yy * yy).mean() - yy.mean()**2).item()

            mseL, mseR = mse(L_rows), mse(R_rows)
            stack_rows.extend([R_rows, L_rows]) # Order corresponds to L/R children
            stack_mse.extend([mseR,  mseL])

            wL, wR = L_rows.numel(), R_rows.numel()
            parent_N = wL + wR
            if parent_N > 0:
                # Standard split gain calculation (reduction in variance)
                gain = parent_mse - (wL / parent_N * mseL + wR / parent_N * mseR)
                dR[token_idx] = gain # Assign positive reward for the split action
            token_idx += 2 # Advance past both feature and threshold tokens
        else:  # 'leaf' token
            if stack_rows:
                stack_rows.pop()
                stack_mse.pop()
            token_idx += 1
    return dR.unsqueeze(0)


def get_tree_predictor(traj: List[int], X_binned: torch.Tensor, y_target: torch.Tensor, tok: "Tokenizer") -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Decodes a token trajectory into a callable tree predictor function.
    The predictor is trained on `y_target` (e.g., residuals) and can predict on new data.
    """
    path_iter = iter(tok.decode(traj[1:-1]))
    def build_recursive() -> Optional[dict]:
        try:
            kind, idx = next(path_iter)
        except StopIteration:
            return None
        if kind == 'feat':
            # This is a split node
            return {'type': 'split', 'f': idx, 't': next(path_iter)[1], 'L': build_recursive(), 'R': build_recursive()}
        # This is a leaf node
        return {'type': 'leaf', 'value': 0}

    tree_structure = build_recursive()
    if tree_structure is None:
        # Return a dummy predictor for an empty tree
        return lambda X: torch.zeros(X.size(0), device=X.device)

    # Determine leaf values by propagating training data through the tree
    leaf_values = {}
    q = deque([(tree_structure, torch.arange(X_binned.size(0), device=X_binned.device))])
    leaf_idx_counter = 0
    while q:
        node, indices = q.popleft()
        if not node or not indices.numel(): continue

        if node['type'] == 'split':
            mask = X_binned[indices, node['f']] <= node['t']
            if node.get('L'): q.append((node['L'], indices[mask]))
            if node.get('R'): q.append((node['R'], indices[~mask]))
        else: # 'leaf'
            node['leaf_idx'] = leaf_idx_counter
            # Leaf value is the mean of the targets that fall into it
            value = y_target[indices].mean().item() if indices.numel() > 0 else y_target.mean().item()
            leaf_values[leaf_idx_counter] = value
            leaf_idx_counter += 1

    def predict(X_test: torch.Tensor) -> torch.Tensor:
        """A vectorized function to predict outcomes for a batch of test data."""
        # This implementation avoids looping over samples in Python by processing
        # all samples down the tree simultaneously.
        preds = torch.zeros(X_test.size(0), device=X_test.device)
        q = deque([(tree_structure, torch.arange(X_test.size(0), device=X_test.device))])

        while q:
            node, indices = q.popleft()
            if not node or not indices.numel(): continue

            if node['type'] == 'leaf':
                # Assign the pre-calculated value to all samples that reached this leaf
                preds[indices] = leaf_values.get(node.get('leaf_idx'), 0.0)
            else: # 'split'
                mask = X_test[indices, node['f']] <= node['t']
                if node.get('L'): q.append((node['L'], indices[mask]))
                if node.get('R'): q.append((node['R'], indices[~mask]))
        return preds

    return predict

# --- Sampling & Buffer ---

class ReplayBuffer:
    """A simple, list-based replay buffer that sorts trajectories by reward."""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        # Data stores tuples of (reward, trajectory, prior, indices)
        self.data: List[Tuple[float, List[int], float, torch.Tensor]] = []

    def add(self, r: float, t: List[int], p: float, idxs: torch.Tensor):
        """Adds a trajectory to the buffer and maintains sorted order by reward."""
        self.data.append((r, t, p, idxs))
        # Note: Sorting is based on reward `r`. If the training loop provides a
        # constant placeholder reward, this sorting will not be meaningful.
        self.data.sort(key=lambda x: x[0], reverse=True)
        if len(self.data) > self.capacity:
            self.data.pop()

    def sample(self, k: int) -> list:
        """Samples k trajectories randomly from the buffer."""
        return random.sample(self.data, min(k, len(self.data)))


END_TOKEN = 2   # A default EOS token ID, adjust if necessary
EPS = 1e-9

def _safe_sample(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Samples action indices safely from a batch of logits.
    - logits:      [B, A] tensor of raw scores.
    - mask:        [B, A] bool tensor – True for legal actions.
    - temperature: > 0, for controlling randomness.
    """
    if temperature > EPS:
        logits = logits / temperature

    # Mask out illegal actions by setting their logit to negative infinity
    logits = logits.masked_fill(~mask, -float("inf"))

    # If all actions in a row are masked, sample END_TOKEN
    all_masked = (~mask).all(dim=-1)
    if all_masked.any():
        # Fallback to a valid token to prevent errors with multinomial
        logits[all_masked, END_TOKEN] = 0

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(1)


def create_gain_bias(df_train: pd.DataFrame, feats: List[str], target: str, tok: "Tokenizer", bins: int, prior_scale: float = 0.5) -> torch.Tensor:
    """Uses a LightGBM model to create a gain-based prior bias for the policy network."""
    # Subsample for efficiency if the dataset is large
    df_sample = df_train.sample(n=min(len(df_train), 200_000), random_state=42)

    # Bin the features, similar to the main environment
    X_binned_list = []
    for f in feats:
        s = df_sample[f].replace([np.inf,-np.inf],np.nan).fillna(df_sample[f].median()).values
        qs = np.linspace(0,1,bins+1)
        edges = np.unique(np.quantile(s,qs))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        X_binned_list.append(np.searchsorted(edges,s,side="right")-1)
    X_binned = np.stack(X_binned_list, 1)
    y_train = df_sample[target].values

    # Train a simple LightGBM model
    lgb_train = lgb.Dataset(X_binned, y_train, feature_name=feats, free_raw_data=False)
    params = { 'objective': 'regression_l1', 'metric': 'l1', 'n_estimators': 100, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'num_leaves': 1024, 'max_depth': 10, 'verbose': -1, 'n_jobs': -1 }
    gbm = lgb.train(params, lgb_train)

    tree_info = gbm.dump_model()["tree_info"]
    bias = torch.zeros(tok.v.size(), dtype=torch.float32)

    def parse_node(node: dict):
        nonlocal bias
        if "split_gain" in node and node["split_gain"] > 0:
            gain = node["split_gain"]
            # Add gain as bias to the feature token
            f_idx = node["split_feature"]
            tok_id_feat = tok._feat(f_idx)
            if tok_id_feat < tok.v.size():
                bias[tok_id_feat] += gain

            # Add gain as bias to the threshold token
            try:
                # Handle numerical thresholds
                bin_idx = min(int(node["threshold"]), bins - 1)
                tok_id_th = tok._th(bin_idx)
                if tok_id_th < tok.v.size():
                    bias[tok_id_th] += gain
            except (ValueError, AttributeError):
                pass # Ignore non-numeric or complex thresholds for simplicity

        if "left_child" in node: parse_node(node["left_child"])
        if "right_child" in node: parse_node(node["right_child"])

    for tree in tree_info:
        parse_node(tree['tree_structure'])

    # Normalize and scale the bias tensor
    bias_std = bias.std()
    if bias_std > 1e-6:
        bias /= bias_std
    bias *= (prior_scale / 2.0)
    return bias.to(dtype=torch.float32)