# src/env.py
from __future__ import annotations
from typing import List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


class TabularEnv:
    """
    Environment wrapper that:
      • fits per-feature binning on train,
      • treats 0/1 (or ≤2-unique) columns as binary,
      • exposes per-feature effective bins (for threshold constraints),
      • provides X_full (binned) and y_full tensors,
      • supplies small helpers used by rollouts/rewards (draw_indices/reset/step/prior).
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_bins: int,
        task: str,
        binning_strategy: str = "quantile",   # "quantile" | "global_uniform"
        device: str = "cpu",
    ):
        assert task in ("classification", "regression")
        self.device = torch.device(device)
        self.task = task
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.n_bins = int(n_bins)
        self.binning_strategy = binning_strategy

        # label-encode y if classification
        if task == "classification":
            self.le = LabelEncoder()
            y_np = self.le.fit_transform(df_train[target_col].to_numpy())
            self.n_classes = int(np.max(y_np)) + 1
            self.y_full = torch.as_tensor(y_np, device=self.device, dtype=torch.long)
        else:
            self.le = None
            self.n_classes = None
            y_np = df_train[target_col].to_numpy(dtype=np.float32)
            self.y_full = torch.as_tensor(y_np, device=self.device, dtype=torch.float32)

        # fit bin edges on train (per feature)
        self.bin_edges, self.per_feat_bins, self.binary_mask = self._fit_bin_edges(df_train[self.feature_cols])

        # bin X_train
        Xb = self._bin_dataframe(df_train[self.feature_cols])
        self.X_full = torch.as_tensor(Xb, device=self.device, dtype=torch.long)

        # rolling state for rollout/prior
        self.paths: List = []
        self.open_leaves: int = 1
        self.done: bool = False
        self.idxs: torch.Tensor = torch.arange(self.X_full.size(0), device=self.device)
        self._n_splits: int = 0

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------
    def _fit_bin_edges(self, Xdf: pd.DataFrame) -> Tuple[List[torch.Tensor], torch.LongTensor, torch.BoolTensor]:
        """
        Build per-feature bin edges. For binary/2-unique columns, fix to 2 bins with edge 0.5.
        For continuous, use quantiles or global uniform. Return:
          - bin_edges: list of 1D tensors of cutpoints (length = bins-1 for that feature)
          - per_feat_bins: LongTensor of effective bins per feature
          - binary_mask: BoolTensor marking features treated as binary
        """
        edges: List[torch.Tensor] = []
        eff_bins: List[int] = []
        binary_mask: List[bool] = []

        Xnp = Xdf.to_numpy(copy=False)
        N, D = Xnp.shape

        for j, col in enumerate(Xdf.columns):
            x = Xnp[:, j]
            x = x[~np.isnan(x)]
            uniq = np.unique(x)

            # binary / two-unique detection
            is_binary = False
            if uniq.size <= 2:
                # common 0/1, but also handles any two distinct values cleanly
                is_binary = True
            # also treat strict 0/1 as binary even if typed oddly
            if set(np.unique(Xdf[col].dropna().astype(float))) <= {0.0, 1.0}:
                is_binary = True

            if is_binary:
                # two bins, single cut at 0.5 (works for {0,1} or {a,b} after normalization)
                cut = torch.tensor([0.5], dtype=torch.float32)
                edges.append(cut)
                eff_bins.append(2)
                binary_mask.append(True)
                continue

            # non-binary continuous
            if self.binning_strategy == "global_uniform":
                lo = float(np.nanmin(x)) if x.size > 0 else 0.0
                hi = float(np.nanmax(x)) if x.size > 0 else lo + 1.0
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = 0.0, 1.0
                if hi <= lo:
                    # nearly constant
                    edges.append(torch.tensor([lo], dtype=torch.float32))
                    eff_bins.append(1)
                    binary_mask.append(False)
                else:
                    # build n_bins uniform bins
                    cuts = np.linspace(lo, hi, num=self.n_bins + 1, endpoint=True)[1:-1]
                    cuts = np.unique(cuts)
                    if cuts.size == 0:
                        cuts = np.array([lo + 1e-6], dtype=np.float32)
                    edges.append(torch.from_numpy(cuts.astype(np.float32)))
                    eff_bins.append(int(cuts.size + 1))
                    binary_mask.append(False)
        else:
                # quantile binning
                qs = np.linspace(0.0, 1.0, num=self.n_bins + 1, endpoint=True)
                qv = np.quantile(x, qs, method="linear") if x.size > 0 else np.linspace(0.0, 1.0, self.n_bins + 1)
                cuts = np.unique(qv[1:-1])  # remove endpoints
                if cuts.size == 0:
                    # fallback: treat as constant with one bin
                    cuts = np.array([qv[0]], dtype=np.float32)
                edges.append(torch.from_numpy(cuts.astype(np.float32)))
                eff_bins.append(int(cuts.size + 1))
                binary_mask.append(False)

        per_feat_bins = torch.as_tensor(eff_bins, dtype=torch.long)
        binary_mask_t = torch.as_tensor(binary_mask, dtype=torch.bool)
        return edges, per_feat_bins, binary_mask_t

    def _bucketize_col(self, x: np.ndarray, cutpoints: torch.Tensor, is_binary: bool) -> np.ndarray:
        """
        x: 1D numpy array
        cutpoints: 1D torch tensor of cut thresholds (length B-1)
        return: int64 bin ids in [0..B-1]
        """
        if is_binary:
            # Fast path: anything > 0.5 goes to bin 1
            out = (x > 0.5).astype(np.int64)
            return out

        cp = cutpoints.cpu().numpy()
        # np.digitize assigns 0..len(cp) by comparing against cp (strict > when right=False)
        # We want bins 0..B-1
        out = np.digitize(x, cp, right=False).astype(np.int64)
        return out

    def _bin_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        Xnp = df.to_numpy(copy=False)
        D = Xnp.shape[1]
        out = np.empty_like(Xnp, dtype=np.int64)
        for j in range(D):
            out[:, j] = self._bucketize_col(Xnp[:, j], self.bin_edges[j], bool(self.binary_mask[j]))
        return out

    # ------------------------------------------------------------------
    # Public helpers used by Trainer
    # ------------------------------------------------------------------
    def _featurise(self, df_new: pd.DataFrame, df_fit: pd.DataFrame, feature_cols: List[str], n_bins: int) -> torch.Tensor:
        """
        Featurise a new dataframe with the *fitted* bin edges.
        """
        X_new = df_new[self.feature_cols]
        Xb = self._bin_dataframe(X_new)
        return torch.as_tensor(Xb, device=self.device, dtype=torch.long)

    # rollout sampling helper
    def draw_indices(self, batch_size: int) -> torch.Tensor:
        N = self.X_full.size(0)
        if batch_size >= N:
            return torch.randperm(N, device=self.device)
        # sample without replacement
        idx = torch.randperm(N, device=self.device)[:batch_size]
        return idx

    def reset(self, length: int):
        """
        For reward computation on a provided y (same order as training rows).
        """
        self.idxs = torch.arange(length, device=self.device)
        self.paths = []
        self.open_leaves = 1
        self.done = False
        self._n_splits = 0

    def step(self, action: Tuple[str, int]):
        """
        No-op for routing (rollout maintains its own stacks),
        but we count thresholds to form a simple structure prior if desired.
        """
        kind, _ = action
        if kind in ("th", "thr", "threshold"):
            self._n_splits += 1

    def get_prior(self, beta: float) -> torch.Tensor:
        """
        Simple structure prior: -beta * (#decision nodes).
        (Kept for compatibility; many TB variants ignore this term.)
        """
        return torch.tensor(-beta * float(self._n_splits), device=self.device, dtype=torch.float32)
