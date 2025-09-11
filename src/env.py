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

    Binning strategies:
      - "quantile"       : exact quantiles via np.nanquantile, deduped cutpoints
      - "global_uniform" : uniform grid between min-max
      - "lgbm_like"      : quantile-like on unique values with frequency weights (LightGBM-ish),
                           collapses duplicate/tied cutpoints, never forces bins beyond support.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_bins: int,
        task: str,
        binning_strategy: str = "quantile",   # "quantile" | "global_uniform" | "lgbm_like"
        device: str = "cpu",
    ):
        assert task in ("classification", "regression")
        assert binning_strategy in ("quantile", "global_uniform", "lgbm_like")
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

    @staticmethod
    def _two_unique_midpoint(u: np.ndarray) -> float:
        """Midpoint between two sorted unique values."""
        a, b = float(u[0]), float(u[1])
        return (a + b) * 0.5

    def _fit_bin_edges(
        self, Xdf: pd.DataFrame
    ) -> Tuple[List[torch.Tensor], torch.LongTensor, torch.BoolTensor]:
        """
        Build per-feature bin edges.

        Rules:
          • Constant → 0 cutpoints, eff_bins=1
          • Two-unique (binary) → 1 cutpoint at the midpoint of the two observed values
            (if values are exactly {0,1}, midpoint is 0.5)
          • Continuous → strategy-dependent cutpoints (may be fewer than n_bins-1 after de-dup)

        Return:
          - bin_edges: list of 1D tensors of cutpoints (length = eff_bins-1 for that feature)
          - per_feat_bins: LongTensor of effective bins per feature
          - binary_mask: BoolTensor marking features treated as binary
        """
        edges: List[torch.Tensor] = []
        eff_bins: List[int] = []
        binary_mask: List[bool] = []

        Xnp = Xdf.to_numpy(copy=False)
        _, D = Xnp.shape

        max_bins = max(1, int(self.n_bins))

        for j, col in enumerate(Xdf.columns):
            # Prepare floating copy (handles ints as well); drop NaN for edge fitting
            x = pd.to_numeric(Xdf[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            x_nn = x[~np.isnan(x)]
            # Unique support
            uniq = np.unique(x_nn)

            # --- constant ---
            if uniq.size <= 1:
                edges.append(torch.tensor([], dtype=torch.float32))
                eff_bins.append(1)
                binary_mask.append(False)
                continue

            # --- binary / two unique ---
            if uniq.size == 2:
                cut = self._two_unique_midpoint(uniq)
                edges.append(torch.tensor([cut], dtype=torch.float32))
                eff_bins.append(2)
                binary_mask.append(True)
                continue

            # --- continuous: build cutpoints per strategy ---
            if self.binning_strategy == "global_uniform":
                lo, hi = float(np.nanmin(x_nn)), float(np.nanmax(x_nn))
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = 0.0, 1.0
                if hi <= lo:
                    # should have been caught by constant branch, but guard anyway
                    cuts = np.array([], dtype=np.float32)
                else:
                    # n_bins → n_bins-1 interior cuts
                    cuts = np.linspace(lo, hi, num=max_bins + 1, endpoint=True)[1:-1].astype(np.float64)
                    cuts = np.unique(cuts)  # dedup in case max_bins==1
                edges.append(torch.from_numpy(cuts.astype(np.float32)))
                eff_bins.append(int(cuts.size + 1))
                binary_mask.append(False)

            elif self.binning_strategy == "quantile":
                # quantile positions (exclude 0 and 1)
                qs = np.linspace(0.0, 1.0, num=max_bins + 1, endpoint=True)[1:-1]
                if qs.size == 0:
                    cuts = np.array([], dtype=np.float64)
                else:
                    # use nan-aware quantiles; dedup to avoid repeated cutpoints on ties
                    qv = np.nanquantile(x_nn, qs, method="linear").astype(np.float64)
                    cuts = np.unique(qv)
                    # drop extremes equal to min/max to avoid empty end-bins
                    cuts = cuts[(cuts > uniq[0]) & (cuts < uniq[-1])]
                edges.append(torch.from_numpy(cuts.astype(np.float32)))
                eff_bins.append(int(cuts.size + 1))
                binary_mask.append(False)

            else:  # "lgbm_like"
                # LightGBM-ish: choose cut values so that each bin has ~equal counts,
                # but operate on unique values with frequency weights, then dedup.
                vals, cnts = np.unique(x_nn, return_counts=True)
                N = float(cnts.sum())
                if vals.size <= 1:
                    cuts = np.array([], dtype=np.float64)
                else:
                    cdf = np.cumsum(cnts) / N
                    targets = np.linspace(0.0, 1.0, num=max_bins + 1, endpoint=True)[1:-1]
                    # map each target quantile to the first unique where cdf >= target
                    idx = np.searchsorted(cdf, targets, side="left")
                    idx = np.clip(idx, 0, vals.size - 1)
                    cuts = np.unique(vals[idx].astype(np.float64))
                    # keep interior only (avoid min/max as cuts)
                    cuts = cuts[(cuts > vals[0]) & (cuts < vals[-1])]
                edges.append(torch.from_numpy(cuts.astype(np.float32)))
                eff_bins.append(int(cuts.size + 1))
                binary_mask.append(False)

        per_feat_bins = torch.as_tensor(eff_bins, dtype=torch.long)
        binary_mask_t = torch.as_tensor(binary_mask, dtype=torch.bool)
        return edges, per_feat_bins, binary_mask_t

    def _bucketize_col(self, x: np.ndarray, cutpoints: torch.Tensor, is_binary: bool) -> np.ndarray:
        """
        x: 1D numpy array (may contain NaN; NaNs will be treated as their own extreme bin via digitize behavior)
        cutpoints: 1D torch tensor of cut thresholds (length B-1)
        return: int64 bin ids in [0..B-1]
        """
        cp = cutpoints.detach().cpu().numpy()
        x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64, copy=False)

        if is_binary:
            # Use the learned binary threshold (midpoint of the two observed values)
            thr = cp[0] if cp.size == 1 else 0.5  # fallback, though cp.size should be 1 here
            out = (x > thr).astype(np.int64)
            # Put NaNs in bin 0 to be safe (could also choose own bin; keep it simple)
            out[np.isnan(x)] = 0
            return out

        if cp.size == 0:
            out = np.zeros_like(x, dtype=np.int64)
            out[np.isnan(x)] = 0
            return out

        # np.digitize: returns 0..len(cp) using <= when right=False
        out = np.digitize(x, cp, right=False).astype(np.int64)
        out[np.isnan(x)] = 0
        return out

    def _bin_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        Xnp = df.to_numpy(copy=False)
        D = Xnp.shape[1]
        out = np.empty((Xnp.shape[0], D), dtype=np.int64)
        for j in range(D):
            out[:, j] = self._bucketize_col(Xnp[:, j], self.bin_edges[j], bool(self.binary_mask[j]))
        return out

    # ------------------------------------------------------------------
    # Public helpers used by Trainer
    # ------------------------------------------------------------------
    def _featurise(
        self, df_new: pd.DataFrame, df_fit: pd.DataFrame, feature_cols: List[str], n_bins: int
    ) -> torch.Tensor:
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