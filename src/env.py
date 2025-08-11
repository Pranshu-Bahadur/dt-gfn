from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class TabularEnv:
    """
    A tabular decision-tree environment for GFN rollouts.

    Grammar accounting:
      - 'feat'   : choose a feature for the CURRENT open leaf (no change)
      - 'th'     : choose a threshold; split 1 open leaf into 2  -> open_leaves += 1
      - 'leaf'   : close the CURRENT open leaf                   -> open_leaves -= 1
      - done     : when open_leaves == 0 (or a hard safety cap)
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_bins: int,
        task: str = "regression",
        device: str = "cpu",
        shuffle_on_reset: bool = True,
        binning_strategy: str = "global_uniform"  # "quantile" | "global_uniform"
    ):
        self.device = device
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_bins = n_bins
        self.task = task
        self.shuffle_on_reset = shuffle_on_reset
        self.binning_strategy = binning_strategy
        self.le: Optional[LabelEncoder] = None
        self.n_classes: Optional[int] = None

        self.feature_scaler: Optional[MinMaxScaler] = None

        # Featurize on init
        self.X_full = self._featurise(df_train, df_train, feature_cols, n_bins)

        if self.task == "classification":
            self.le = LabelEncoder()
            y_encoded = self.le.fit_transform(df_train[target_col].values)
            self.y_full = torch.tensor(y_encoded, dtype=torch.long, device=device)
            self.n_classes = int(self.y_full.max().item()) + 1
        else:
            self.y_full = torch.tensor(df_train[target_col].values, dtype=torch.float32, device=device)
            self.n_classes = 1

        # working target (overridable by trainer: residuals, etc.)
        self.y = self.y_full.clone()

        # master index pool (shuffled by draw_indices if requested)
        self._master_indices = torch.arange(len(self.y_full), device=device)
        self._ptr: int = 0

        # rollout state
        self.paths: List[Tuple[str, int]] = []
        self.open_leaves: int = 1
        self.done: bool = False
        self.idxs: torch.Tensor = self._master_indices

        # safety cap on emitted tokens to avoid runaway sequences
        self._max_tokens: int = 8192

    # -----------------------------
    # Shared index sampler (FIX)
    # -----------------------------
    def draw_indices(self, batch_size: int) -> torch.Tensor:
        """
        Draw a slice of row indices from a shared pointer,
        shuffling when we wrap around. This replaces calling reset()
        on shallow copies (which each had their own pointer at 0).
        """
        if self._ptr + batch_size > len(self._master_indices):
            self._ptr = 0
            if self.shuffle_on_reset:
                perm = torch.randperm(len(self._master_indices), device=self.device)
                self._master_indices = self._master_indices[perm]
        idxs = self._master_indices[self._ptr : self._ptr + batch_size]
        self._ptr += batch_size
        return idxs

    def _featurise(
        self,
        df_target: pd.DataFrame,
        df_source: pd.DataFrame,
        feats: List[str],
        bins: int
    ) -> torch.Tensor:
        """
        Bin features. For classification with 'global_uniform',
        apply MinMax scaling first to keep bins consistent.
        """
        # If regression with already integer-binned features, pass through
        if self.task == "regression" and all(pd.api.types.is_integer_dtype(df_source[f]) for f in feats):
            return torch.tensor(df_target[feats].values.astype(np.int32), device=self.device)

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
                source_series = df_source_processed[f].replace([np.inf, -np.inf], np.nan)
                source_median = source_series.median()

                s_source = source_series.fillna(source_median).values
                target_series = df_target_processed[f].replace([np.inf, -np.inf], np.nan)
                s_eval = target_series.fillna(source_median).values

                quantiles = np.linspace(0, 1, bins + 1)
                edges = np.unique(np.quantile(s_source, quantiles))
                if edges.size == 1:
                    # constant feature guard
                    edges = np.array([edges[0] - 1, edges[0] + 1], dtype=float)
                edges[0] -= 1e-9
                edges[-1] += 1e-9

                binned_eval = np.searchsorted(edges, s_eval, side="right") - 1
                X_binned.append(binned_eval)

        elif self.binning_strategy == "global_uniform":
            edges = np.linspace(0, 1, bins + 1)
            edges[0] -= 1e-9
            edges[-1] += 1e-9

            for f in feats:
                vals = df_target_processed[f].values.ravel()
                binned_eval = np.searchsorted(edges, vals, side="right") - 1
                X_binned.append(binned_eval)
        else:
            raise ValueError(f"Unknown binning_strategy: {self.binning_strategy}")

        Xb = np.stack(X_binned, 1).astype(np.int32)
        return torch.tensor(Xb, device=self.device)

    # legacy reset (still used by training env itself, not by shallow copies)
    def reset(self, batch_size: int):
        """
        Resets this environment (NOT copies) for a new rollout batch.
        """
        self.idxs = self.draw_indices(batch_size)
        self.paths = []
        self.open_leaves = 1
        self.done = False

    def step(self, action: Tuple[str, int]):
        """
        Advance the environment by one token action.
        """
        self.paths.append(action)
        kind, _ = action

        if kind == "feat":
            pass
        elif kind == "th":
            self.open_leaves += 1
        elif kind == "leaf":
            self.open_leaves -= 1

        if self.open_leaves < 0:
            self.open_leaves = 0

        # done when no open leaves or token budget exceeded
        self.done = (self.open_leaves == 0) or (len(self.paths) > self._max_tokens)

    def get_prior(self, current_beta: float) -> torch.Tensor:
        """
        Structure prior for a completed trajectory:
        penalize by number of 'feat' tokens (splits).
        """
        n_feats = sum(1 for k, _ in self.paths if k == "feat")
        prior = -current_beta * float(n_feats)
        return torch.tensor([prior], device=self.device)
