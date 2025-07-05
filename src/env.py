# src/env.py

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch

class TabularEnv:
    """
    A tabular decision-tree environment for GFN rollouts, updated to match
    the production pipeline for Numerai data.

    - Features are binned using quantile-based method.
    - Reward is 1 / (1 + MSE).
    - Step logic correctly handles open leaves.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_bins: int,
        device: str = "cpu",
    ):
        self.device = device
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_bins = n_bins

        # Featurization is now handled internally
        self.X_full = self._featurise(df_train, df_train, feature_cols, n_bins)
        self.y_full = torch.tensor(
            df_train[target_col].values, dtype=torch.float32, device=device
        )
        self.y = self.y_full.clone()

        # Will be set in reset()
        self.paths: List[Tuple[str, int]]
        self.open_leaves: int
        self.done: bool
        self.idxs: torch.Tensor

    def _featurise(
        self,
        df_target: pd.DataFrame,
        df_source: pd.DataFrame,
        feats: List[str],
        bins: int
    ) -> torch.Tensor:
        """
        Bins features using quantiles from the source dataframe.
        Skips binning if the data is already of integer type.
        """
        # --- Check if data is already binned ---
        if all(pd.api.types.is_integer_dtype(df_source[f]) for f in feats):
            return torch.tensor(
                df_target[feats].values.astype(np.int8), device=self.device
            )

        # --- If not binned, proceed with quantile binning ---
        X_binned = []
        for f in feats:
            # Get series from source and target dataframes
            s_source = df_source[f].replace([np.inf, -np.inf], np.nan).fillna(df_source[f].median()).values
            s_eval = df_target[f].replace([np.inf, -np.inf], np.nan).fillna(df_source[f].median()).values

            # Create quantile-based bin edges from the source data
            quantiles = np.linspace(0, 1, bins + 1)
            edges = np.quantile(s_source, quantiles)
            edges = np.unique(edges)
            edges[0] -= 1e-9  # Ensure the lowest values are included
            edges[-1] += 1e-9 # Ensure the highest values are included

            # Bin the target data using the calculated edges
            binned_eval = np.searchsorted(edges, s_eval, side="right") - 1
            X_binned.append(binned_eval)

        return torch.tensor(
            np.stack(X_binned, 1).astype(np.int32), device=self.device
        )

    def reset(self, batch_size: int):
        """
        Resets the environment for a new rollout, using a non-shuffling,
        circular pointer to select a contiguous block of data.
        """
        if not hasattr(self, "_ptr"):
            self._ptr = 0  # Initialise the pointer on the first call

        n = len(self.y_full)
        end = self._ptr + batch_size

        if end <= n:
            # Standard case: the batch fits within the remaining data
            idxs = torch.arange(self._ptr, end, device=self.device)
        else:
            # Wrap-around case: the batch goes beyond the end of the data
            first_part = torch.arange(self._ptr, n, device=self.device)
            second_part = torch.arange(0, end - n, device=self.device)
            idxs = torch.cat([first_part, second_part])

        self._ptr = end % n  # Move the pointer for the next batch

        self.idxs = idxs
        self.paths, self.open_leaves, self.done = [], 1, False

    def step(self, action: Tuple[str, int]):
        """
        Advance the environment by one token action.
        Updates open_leaves based on the action taken.
        """
        self.paths.append(action)
        kind, _ = action
        if kind == "feat":
            self.open_leaves += 1
        elif kind == "leaf":
            self.open_leaves -= 1

        # A trajectory is done if all leaves are closed or if it's too long
        self.done = (self.open_leaves == 0 or len(self.paths) > 8192)

    def evaluate(self, current_beta: float):
        """
        Computes the reward for a completed trajectory using 1 / (1 + MSE).
        This version is optimized for speed using vectorized operations.
        """
        prior = -current_beta * sum(1 for k, _ in self.paths if k == "feat")
        y_t = self.y[self.idxs]

        if y_t.numel() == 0:
            return 0.0, torch.tensor([prior], device=self.device), None, None

        X_batch = self.X_full[self.idxs]
        base_mse = ((y_t - y_t.mean()) ** 2).mean()

        # For incomplete trees, reward is based on the baseline MSE
        if not self.done:
            return 1 / (1 + base_mse), torch.tensor([prior], device=self.device), None, y_t

        # Vectorized prediction
        leaf_indices = torch.zeros(y_t.numel(), dtype=torch.long, device=self.device)
        active_leaves = {0}
        leaf_map = {0: 0}
        leaf_values = {0: y_t.mean()}
        
        path_iter = iter(self.paths)
        for action in path_iter:
            kind, val = action
            if kind == 'feat':
                # Get the threshold for the current feature
                _, threshold = next(path_iter)
                
                # Identify the samples in the current leaf
                leaf_id = active_leaves.pop()
                samples_in_leaf = (leaf_indices == leaf_map[leaf_id])
                
                # Split the samples
                left_mask = (X_batch[samples_in_leaf, val] <= threshold)
                
                # Create new leaf IDs
                left_leaf_id = max(leaf_map.keys()) + 1
                right_leaf_id = max(leaf_map.keys()) + 2
                
                # Update leaf map
                leaf_map[left_leaf_id] = left_leaf_id
                leaf_map[right_leaf_id] = right_leaf_id
                
                # Update leaf indices
                leaf_indices[samples_in_leaf][left_mask] = left_leaf_id
                leaf_indices[samples_in_leaf][~left_mask] = right_leaf_id
                
                # Update active leaves
                active_leaves.add(left_leaf_id)
                active_leaves.add(right_leaf_id)
                
                # Update leaf values
                y_left = y_t[leaf_indices == left_leaf_id]
                y_right = y_t[leaf_indices == right_leaf_id]
                
                leaf_values[left_leaf_id] = y_left.mean() if y_left.numel() > 0 else y_t.mean()
                leaf_values[right_leaf_id] = y_right.mean() if y_right.numel() > 0 else y_t.mean()

        # Create prediction tensor
        pred = torch.zeros_like(y_t)
        for leaf_id, leaf_val in leaf_values.items():
            pred[leaf_indices == leaf_id] = leaf_val
            
        # MSE-based Reward Calculation
        mse = ((pred - y_t)**2).mean()
        reward = 1 / (1 + mse)

        return reward.item(), torch.tensor([prior], device=self.device), pred, y_t