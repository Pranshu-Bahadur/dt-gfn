from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


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
        shuffle_on_reset: bool = False,
        binning_strategy: str = "global_uniform" # "quantile" or "global_uniform"
    ):
        self.device = device
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_bins = n_bins
        self.task = task
        self.shuffle_on_reset = True#shuffle_on_reset
        self.binning_strategy = binning_strategy
        self.le = None
        self.n_classes: Optional[int] = None
        
        self.feature_scaler = None

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
            self.n_classes=1

        self.y = self.y_full.clone()
        
        self._master_indices = torch.arange(len(self.y_full), device=device)

        self.paths: List[Tuple[str, int]] = []
        self.open_leaves: int = 1
        self.done: bool = False
        self.idxs: torch.Tensor = self._master_indices

    def _featurise(
        self,
        df_target: pd.DataFrame,
        df_source: pd.DataFrame,
        feats: List[str],
        bins: int
    ) -> torch.Tensor:
        """
        Normalizes (for classification) and bins features. For regression, if features
        are already integers, they are returned directly as an int8 tensor.
        """
        # --- NEW: Handle pre-binned integer features for regression ---
        if self.task == "regression" and all(pd.api.types.is_integer_dtype(df_source[f]) for f in feats):
            return torch.tensor(
                df_target[feats].values.astype(np.int8), device=self.device
            )

        df_target_processed = df_target[feats].copy()
        df_source_processed = df_source[feats].copy()
        
        if self.task == "classification" and self.binning_strategy != "quantile":
            if self.feature_scaler is None:
                self.feature_scaler = MinMaxScaler()
                self.feature_scaler.fit(df_source_processed)
            
            df_target_processed[:] = self.feature_scaler.transform(df_target_processed)
            if not df_source.equals(df_target):
                df_source_processed[:] = self.feature_scaler.transform(df_source_processed)

        X_binned = []
        
        if self.binning_strategy == "quantile":
            for f in feats:
                # Compute median from source (train) for consistent imputation
                source_series = df_source_processed[f].replace([np.inf, -np.inf], np.nan)
                source_median = source_series.median()
                
                # Impute source with source median
                s_source = source_series.fillna(source_median).values
                
                # Impute target with source median (to avoid leakage)
                target_series = df_target_processed[f].replace([np.inf, -np.inf], np.nan)
                s_eval = target_series.fillna(source_median).values

                quantiles = np.linspace(0, 1, bins + 1)
                edges = np.quantile(s_source, quantiles)
                edges = np.unique(edges)
                edges[0] -= 1e-9
                edges[-1] += 1e-9

                binned_eval = np.searchsorted(edges, s_eval, side="right") - 1
                X_binned.append(binned_eval)

        elif self.binning_strategy == "global_uniform":
            edges = np.linspace(0, 1, bins + 1)
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            
            for f in feats:
                binned_eval = np.searchsorted(edges, df_target_processed[f].values.ravel(), side="right") - 1
                X_binned.append(binned_eval)
        else:
            raise ValueError(f"Unknown binning_strategy: {self.binning_strategy}")

        return torch.tensor(
            np.stack(X_binned, 1).astype(np.int32), device=self.device
        )

    def reset(self, batch_size: int):
        """
        Resets the environment for a new rollout.
        """
        if not hasattr(self, "_ptr") or self._ptr + batch_size > len(self._master_indices):
            self._ptr = 0
            if self.shuffle_on_reset:
                self._master_indices = self._master_indices[torch.randperm(len(self._master_indices))]
        
        self.idxs = self._master_indices[self._ptr : self._ptr + batch_size]
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
