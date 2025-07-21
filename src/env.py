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
    - Reward calculation is removed as it's handled externally.
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

    def get_prior(self, current_beta: float) -> torch.Tensor:
        """
        Computes the prior for a completed trajectory.
        Reward is now calculated externally.
        """
        prior = -current_beta * sum(1 for k, _ in self.paths if k == "feat")
        return torch.tensor([prior], device=self.device)