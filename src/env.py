from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


class TabularEnv:
    """
    A tabular decision-tree environment for GFN rollouts.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_bins: int,
        task: str = "regression",
        device: str = "cpu",
        shuffle_on_reset: bool = False
    ):
        self.device = device
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_bins = n_bins
        self.task = task
        self.shuffle_on_reset = shuffle_on_reset
        self.le = None
        self.n_classes: Optional[int] = None

        # Featurization is handled internally
        self.X_full = self._featurise(df_train, df_train, feature_cols, n_bins)
        
        if self.task == "classification":
            self.le = LabelEncoder()
            y_encoded = self.le.fit_transform(df_train[target_col].values)
            self.y_full = torch.tensor(y_encoded, dtype=torch.long, device=device)
            self.n_classes = len(self.le.classes_)
        else:
            self.y_full = torch.tensor(
                df_train[target_col].values, dtype=torch.float32, device=device
            )

        self.y = self.y_full.clone()
        
        # Keep a master set of indices to shuffle from
        self._master_indices = torch.arange(len(self.y_full), device=device)

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
        """
        if all(pd.api.types.is_integer_dtype(df_source[f]) for f in feats):
            return torch.tensor(
                df_target[feats].values.astype(np.int8), device=self.device
            )

        X_binned = []
        for f in feats:
            s_source = df_source[f].replace([np.inf, -np.inf], np.nan).fillna(df_source[f].median()).values
            s_eval = df_target[f].replace([np.inf, -np.inf], np.nan).fillna(df_source[f].median()).values

            quantiles = np.linspace(0, 1, bins + 1)
            edges = np.quantile(s_source, quantiles)
            edges = np.unique(edges)
            edges[0] -= 1e-9
            edges[-1] += 1e-9

            binned_eval = np.searchsorted(edges, s_eval, side="right") - 1
            X_binned.append(binned_eval)

        return torch.tensor(
            np.stack(X_binned, 1).astype(np.int32), device=self.device
        )

    def reset(self, batch_size: int):
        """
        Resets the environment for a new rollout.
        If shuffle_on_reset is True, it shuffles the data indices before selecting a batch.
        Otherwise, it uses a non-shuffling, circular pointer.
        """
        if not hasattr(self, "_ptr"):
            self._ptr = 0  # Initialise the pointer
            # Shuffle indices at the beginning of the first epoch if required
            if self.shuffle_on_reset:
                self._master_indices = self._master_indices[torch.randperm(len(self._master_indices))]

        # If we've completed an epoch, reset pointer and maybe shuffle
        if self._ptr + batch_size > len(self._master_indices):
            self._ptr = 0
            if self.shuffle_on_reset:
                # Reshuffle for the new epoch
                self._master_indices = self._master_indices[torch.randperm(len(self._master_indices))]
        
        # Select the next batch of indices
        self.idxs = self._master_indices[self._ptr : self._ptr + batch_size]
        
        # Move the pointer for the next batch
        self._ptr += batch_size

        self.paths, self.open_leaves, self.done = [], 1, False

    def step(self, action: Tuple[str, int]):
        """
        Advance the environment by one token action.
        """
        self.paths.append(action)
        kind, _ = action
        if kind == "feat":
            self.open_leaves += 1
        elif kind == "leaf":
            self.open_leaves -= 1

        self.done = (self.open_leaves == 0 or len(self.paths) > 8192)

    def get_prior(self, current_beta: float) -> torch.Tensor:
        """
        Computes the prior for a completed trajectory.
        """
        prior = -current_beta * sum(1 for k, _ in self.paths if k == "feat")
        return torch.tensor([prior], device=self.device)