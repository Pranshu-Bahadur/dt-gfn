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
      • supplies helpers used by rollouts/rewards.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_bins: int,
        task: str,
        binning_strategy: str = "lgbm_quantile",   # "quantile" | "global_uniform" | "lgbm_quantile"
        device: str = "cpu",
        *,
        # LGBM-style knobs
        min_data_in_bin: int = 500,
        subsample_for_bin: int = 200_000,
        # only used with global_uniform; quantile modes ignore scaling
        pre_normalize: Optional[str] = None,  # None | "rank" | "zscore" | "minmax"
    ):
        assert task in ("classification", "regression")
        self.device = torch.device(device)
        self.task = task
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.n_bins = int(n_bins)
        self.binning_strategy = binning_strategy
        self.min_data_in_bin = int(min_data_in_bin)
        self.subsample_for_bin = int(subsample_for_bin)
        self.pre_normalize = pre_normalize

        # y
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
        Xfit = df_train[self.feature_cols].copy()
        if self.binning_strategy == "global_uniform" and self.pre_normalize == "rank":
            # rank-normalize into [0,1] to make uniform bins behave like quantile bins
            for col in self.feature_cols:
                s = Xfit[col].to_numpy()
                mask = ~np.isnan(s)
                ranks = np.zeros_like(s, dtype=np.float32)
                if mask.any():
                    order = np.argsort(s[mask], kind="mergesort")
                    inv = np.empty_like(order)
                    inv[order] = np.arange(order.size)
                    r = inv.astype(np.float32) / max(1, order.size - 1)
                    ranks[mask] = r
                Xfit[col] = ranks

        self.bin_edges, self.per_feat_bins, self.binary_mask = self._fit_bin_edges(Xfit)

        # bin X_train
        Xb = self._bin_dataframe(df_train[self.feature_cols])
        self.X_full = torch.as_tensor(Xb, device=self.device, dtype=torch.long)

        # rollout state
        self.paths: List = []
        self.open_leaves: int = 1
        self.done: bool = False
        self.idxs: torch.Tensor = torch.arange(self.X_full.size(0), device=self.device)
        self._n_splits: int = 0

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------
    def _compute_lgbm_edges(self, x_full: np.ndarray) -> np.ndarray:
        """
        LGBM-ish quantile binning:
          • at most n_bins bins
          • guarantee ~min_data_in_bin per bin by inflating quantile step
          • subsample for speed on large arrays
        Returns cutpoints (length B-1). May be empty if effectively constant.
        """
        x = x_full
        x = x[np.isfinite(x)]
        N = x.size
        if N == 0:
            return np.asarray([0.0], dtype=np.float32)

        # (optional) subsample to speed quantiles; LightGBM uses ~200k by default
        if self.subsample_for_bin > 0 and N > self.subsample_for_bin:
            idx = np.random.default_rng(123).choice(N, size=self.subsample_for_bin, replace=False)
            xq = x[idx]
            Nq = xq.size
        else:
            xq = x
            Nq = N

        # step chosen to satisfy both max_bin and min_data_in_bin
        step_by_maxbin = 1.0 / float(self.n_bins)
        step_by_mincount = float(self.min_data_in_bin) / float(max(1, Nq))
        step = max(step_by_maxbin, step_by_mincount)
        if step >= 1.0:
            return np.asarray([np.nanmin(xq)], dtype=np.float32)

        qs = np.arange(step, 1.0, step)
        cuts = np.quantile(xq, qs, method="linear").astype(np.float32)

        # de-duplicate equal cuts (flat regions), cap to <= n_bins-1 edges
        cuts = np.unique(cuts)
        if cuts.size > (self.n_bins - 1):
            # thin evenly
            take = np.linspace(0, cuts.size - 1, num=self.n_bins - 1, dtype=int)
            cuts = cuts[take]

        # guard for effectively constant features
        if cuts.size == 0:
            cuts = np.asarray([np.nanmin(xq)], dtype=np.float32)

        return cuts

    def _fit_bin_edges(self, Xdf: pd.DataFrame) -> Tuple[List[torch.Tensor], torch.LongTensor, torch.BoolTensor]:
        edges: List[torch.Tensor] = []
        eff_bins: List[int] = []
        binary_mask: List[bool] = []

        Xnp = Xdf.to_numpy(copy=False)
        D = Xnp.shape[1]

        for j, col in enumerate(Xdf.columns):
            x = Xnp[:, j]

            # detect binary / 2-unique
            x_no_nan = x[np.isfinite(x)]
            uniq = np.unique(x_no_nan)
            is_binary = uniq.size <= 2
            # also treat strict 0/1 as binary (robust)
            try:
                if set(np.unique(Xdf[col].dropna().astype(float))) <= {0.0, 1.0}:
                    is_binary = True
            except Exception:
                pass

            if is_binary:
                edges.append(torch.tensor([0.5], dtype=torch.float32))
                eff_bins.append(2)
                binary_mask.append(True)
                continue

            # non-binary continuous
            if self.binning_strategy == "global_uniform":
                lo = float(np.nanmin(x_no_nan)) if x_no_nan.size > 0 else 0.0
                hi = float(np.nanmax(x_no_nan)) if x_no_nan.size > 0 else lo + 1.0
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    cuts = np.asarray([lo], dtype=np.float32)
                else:
                    cuts = np.linspace(lo, hi, num=self.n_bins + 1, endpoint=True)[1:-1].astype(np.float32)
                    cuts = np.unique(cuts)
                    if cuts.size == 0:
                        cuts = np.asarray([lo + 1e-6], dtype=np.float32)

            elif self.binning_strategy == "lgbm_quantile":
                cuts = self._compute_lgbm_edges(x)

            else:  # "quantile"
                qs = np.linspace(0.0, 1.0, num=self.n_bins + 1, endpoint=True)
                qv = np.quantile(x_no_nan, qs, method="linear") if x_no_nan.size > 0 else np.linspace(0.0, 1.0, self.n_bins + 1)
                cuts = np.unique(qv[1:-1]).astype(np.float32)
                if cuts.size == 0:
                    cuts = np.asarray([qv[0]], dtype=np.float32)

            edges.append(torch.from_numpy(cuts))
            eff_bins.append(int(cuts.size + 1))
            binary_mask.append(False)

        per_feat_bins = torch.as_tensor(eff_bins, dtype=torch.long)
        binary_mask_t = torch.as_tensor(binary_mask, dtype=torch.bool)
        return edges, per_feat_bins, binary_mask_t

    def _bucketize_col(self, x: np.ndarray, cutpoints: torch.Tensor, is_binary: bool) -> np.ndarray:
        if is_binary:
            return (x > 0.5).astype(np.int64)
        cp = cutpoints.cpu().numpy()
        # treat NaN as the lowest bin (consistent with many tree impls) — optionally adjust
        out = np.digitize(np.nan_to_num(x, nan=cp[0] if cp.size else 0.0), cp, right=False).astype(np.int64)
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
    def _featurise(self, df_new: pd.DataFrame, df_fit: pd.DataFrame, feature_cols: List[str], n_bins: int) -> torch.Tensor:
        X_new = df_new[self.feature_cols]
        Xb = self._bin_dataframe(X_new)
        return torch.as_tensor(Xb, device=self.device, dtype=torch.long)

    def draw_indices(self, batch_size: int) -> torch.Tensor:
        N = self.X_full.size(0)
        if batch_size >= N:
            return torch.randperm(N, device=self.device)
        return torch.randperm(N, device=self.device)[:batch_size]

    def reset(self, length: int):
        self.idxs = torch.arange(length, device=self.device)
        self.paths = []
        self.open_leaves = 1
        self.done = False
        self._n_splits = 0

    def step(self, action):
        kind, _ = action
        if kind in ("th", "thr", "threshold"):
            self._n_splits += 1

    def get_prior(self, beta: float) -> torch.Tensor:
        return torch.tensor(-beta * float(self._n_splits), device=self.device, dtype=torch.float32)
