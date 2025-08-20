# src/env.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch


class TabularEnv:
    """
    Training environment for DT-GFN on tabular data.

    Responsibilities:
      • Hold binned train matrix (X_full) and target (y_full)
      • Provide per-feature binning (LightGBM-like or classic quantile/uniform)
      • Re-bin new data with the learned edges (_featurise)
      • Offer rollout utilities used by Trainer: draw_indices/reset/step/get_prior

    Notes on structure prior hooks:
      • step(('leaf', 0))   → close one open leaf
      • step(('feat', f))   → mark intent to split (no leaf count change yet)
      • step(('th', t))     → commit split, open leaves +1, increment n_splits
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: List[str],
        target_col: str,
        n_bins: int = 255,
        task: str = "classification",          # "classification" | "regression"
        binning_strategy: str = "quantile",    # "quantile" | "global_uniform" | "lgbm_like" | "per_feature_auto"
        device: str = "cpu",

        # LightGBM-like binning controls (used when binning_strategy is lgbm_like/per_feature_auto)
        min_data_in_bin: Optional[int] = None,
        subsample_for_bin: Optional[int] = None,

        # Optional future hook: explicit per-feature overrides (unused if None)
        per_feature_binning: Optional[Dict[str, Dict]] = None,
    ):
        self.device = torch.device(device)
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.task = task
        self.n_bins = int(n_bins)
        self.binning_strategy = str(binning_strategy)
        self.min_data_in_bin = 1 if min_data_in_bin is None else int(min_data_in_bin)
        self.subsample_for_bin = 200_000 if subsample_for_bin is None else int(subsample_for_bin)
        self.per_feature_binning = per_feature_binning or {}

        # ----- Targets -----
        y = df[target_col].to_numpy()
        if task == "classification":
            from sklearn.preprocessing import LabelEncoder
            self.le = LabelEncoder()
            y_enc = self.le.fit_transform(y)
            self.n_classes = int(len(self.le.classes_))
            self.y_full = torch.from_numpy(y_enc).long().to(self.device)
        else:
            self.le = None
            self.n_classes = None
            self.y_full = torch.from_numpy(y.astype(np.float32)).to(self.device)

        # ----- Features (bin and store edges) -----
        X_df = df[self.feature_cols].copy()

        strat = self.binning_strategy.lower()
        if strat in ("lgbm_like", "per_feature_auto", "quantile_per_feature"):
            X_binned_np, meta = self._build_bins_lgbm_like(
                X_df,
                max_bins=self.n_bins,
                min_data_in_bin=self.min_data_in_bin,
                subsample_for_bin=self.subsample_for_bin,
                overrides=self.per_feature_binning,
            )
            self.bin_meta = meta  # edges per feature, bins per feature, total bins
        elif strat in ("quantile",):
            X_binned_np, meta = self._build_bins_quantile(X_df, n_bins=self.n_bins)
            self.bin_meta = meta
        elif strat in ("global_uniform", "uniform"):
            X_binned_np, meta = self._build_bins_uniform(X_df, n_bins=self.n_bins)
            self.bin_meta = meta
        else:
            raise ValueError(f"Unknown binning_strategy='{self.binning_strategy}'.")

        self.X_full = torch.from_numpy(X_binned_np.astype(np.int16, copy=False)).to(self.device)
        self.idxs = torch.arange(self.X_full.size(0), device=self.device, dtype=torch.long)

        # ----- Rollout bookkeeping (mutable; trainer will copy envs) -----
        self.paths: List[Tuple] = []   # free to use as you like
        self.open_leaves: int = 1
        self.done: bool = False
        self._batch_size: int = 0
        self._n_splits: int = 0        # decision (split) count for structure prior

    # ======================================================================
    # Binning implementations
    # ======================================================================
    def _build_bins_lgbm_like(
        self,
        X_df: pd.DataFrame,
        *,
        max_bins: int,
        min_data_in_bin: int,
        subsample_for_bin: int,
        overrides: Dict[str, Dict],
    ) -> Tuple[np.ndarray, Dict]:
        """
        Per-feature binning roughly like LightGBM:
          • binary/constant → 1–2 bins
          • numeric         → quantile cuts (on subsample), cap at max_bins
          • optional merge of tiny adjacent bins to satisfy min_data_in_bin
          • respects simple 'overrides' like {"type": "binary"|"continuous", "n_bins": K}
        Prints:
          [Binning] Total Bins {sum over feats}
          [Binning] Number of data points in the train set: N, number of used features: F
        """
        rng = np.random.default_rng(12345)
        N, F = X_df.shape

        if subsample_for_bin and N > subsample_for_bin:
            sample_idx = rng.choice(N, size=subsample_for_bin, replace=False)
            Xs = X_df.iloc[sample_idx]
        else:
            Xs = X_df

        edges_by_feat: Dict[str, np.ndarray] = {}
        bins_by_feat: Dict[str, int] = {}
        X_binned = np.zeros((N, F), dtype=np.int32)

        for j, col in enumerate(X_df.columns):
            s = X_df[col].to_numpy()
            ss = Xs[col].to_numpy()
            ov = overrides.get(col, {})

            # If explicitly forced to binary/continuous with custom bins
            forced_type = ov.get("type", None)
            forced_bins = int(ov.get("n_bins", max_bins)) if "n_bins" in ov else max_bins

            edges: np.ndarray
            # ---- Detect binary quickly unless overridden ----
            if forced_type == "binary":
                edges = np.array([-0.5, 0.5, 1.5], dtype=np.float64)
            elif forced_type == "continuous":
                edges = self._quantile_edges(ss, forced_bins)
                edges = self._merge_tiny_bins(ss, edges, min_data_in_bin, forced_bins)
            else:
                uniq_sample = np.unique(ss[~pd.isna(ss)])
                if uniq_sample.size <= 1:
                    # constant
                    edges = np.array([-np.inf, np.inf], dtype=np.float64)  # 1 bin
                elif uniq_sample.size == 2 and set(uniq_sample).issubset({0, 1}):
                    # true binary
                    edges = np.array([-0.5, 0.5, 1.5], dtype=np.float64)   # 2 bins
                else:
                    # numeric-like
                    edges = self._quantile_edges(ss, max_bins)
                    edges = self._merge_tiny_bins(ss, edges, min_data_in_bin, max_bins)

            # Apply to full column
            bj = np.searchsorted(edges, s, side="right") - 1
            bj = np.clip(bj, 0, edges.size - 2).astype(np.int32)
            X_binned[:, j] = bj

            edges_by_feat[col] = edges
            bins_by_feat[col] = int(edges.size - 1)

        total_bins = int(sum(bins_by_feat.values()))
        print(f"[Binning] Total Bins {total_bins}")
        print(f"[Binning] Number of data points in the train set: {N}, number of used features: {F}")

        meta = dict(
            strategy="lgbm_like",
            edges=edges_by_feat,
            bins_per_feature=bins_by_feat,
            total_bins=total_bins,
        )
        return X_binned, meta

    def _quantile_edges(self, sample_values: np.ndarray, max_bins: int) -> np.ndarray:
        xs = sample_values.astype(np.float64)
        xs = xs[np.isfinite(xs)]
        if xs.size == 0:
            return np.array([-np.inf, np.inf], dtype=np.float64)
        # start with max_bins bins → max_bins+1 edges
        q = np.linspace(0.0, 1.0, num=min(max_bins, max(2, xs.size)) + 1)
        raw = np.unique(np.quantile(xs, q))
        # cap edges to (max_bins + 1)
        if raw.size > (max_bins + 1):
            stride = max(1, raw.size // (max_bins + 1))
            raw = raw[::stride]
            if raw.size > (max_bins + 1):
                raw = raw[: (max_bins + 1)]
        # widen outer edges slightly
        raw[0] = np.floor(raw[0] - 1e-9)
        raw[-1] = np.ceil(raw[-1] + 1e-9)
        return raw.astype(np.float64, copy=False)

    def _merge_tiny_bins(
        self,
        sample_values: np.ndarray,
        edges: np.ndarray,
        min_data_in_bin: int,
        max_bins: int,
    ) -> np.ndarray:
        """Merge adjacent bins left-to-right until all sample hist counts ≥ min_data_in_bin."""
        if min_data_in_bin <= 1 or edges.size <= 2:
            return edges

        xs = sample_values.astype(np.float64)
        xs = xs[np.isfinite(xs)]
        if xs.size == 0:
            return edges

        hist = np.searchsorted(edges, xs, side="right") - 1
        hist = np.clip(hist, 0, edges.size - 2)
        counts = np.bincount(hist, minlength=edges.size - 1).astype(np.int64)

        E = edges.tolist()
        C = counts.tolist()
        k = 0
        while k < len(C):
            if C[k] >= min_data_in_bin:
                k += 1
                continue
            # merge with right neighbor if possible; else with left
            if k + 1 < len(C):
                C[k + 1] += C[k]
                del C[k]
                del E[k + 1]           # remove boundary between k and k+1
            else:
                C[k - 1] += C[k]
                del C[k]
                del E[k]               # remove last edge
                k -= 1

            # hard cap to max_bins (merge smallest adjacent pair)
            if len(C) > max_bins and len(C) >= 2:
                pairs = [C[i] + C[i + 1] for i in range(len(C) - 1)]
                idx = int(np.argmin(pairs))
                C[idx] += C[idx + 1]
                del C[idx + 1]
                del E[idx + 1]
        return np.asarray(E, dtype=np.float64)

    def _build_bins_quantile(self, X_df: pd.DataFrame, *, n_bins: int) -> Tuple[np.ndarray, Dict]:
        edges_by_feat: Dict[str, np.ndarray] = {}
        bins_by_feat: Dict[str, int] = {}
        X_binned = np.zeros((X_df.shape[0], X_df.shape[1]), dtype=np.int32)

        for j, col in enumerate(X_df.columns):
            s = X_df[col].to_numpy()
            uniq = np.unique(s[~pd.isna(s)])
            if uniq.size <= 1:
                edges = np.array([-np.inf, np.inf], dtype=np.float64)
            elif uniq.size == 2 and set(uniq).issubset({0, 1}):
                edges = np.array([-0.5, 0.5, 1.5], dtype=np.float64)
            else:
                edges = self._quantile_edges(s, n_bins)

            bj = np.searchsorted(edges, s, side="right") - 1
            bj = np.clip(bj, 0, edges.size - 2).astype(np.int32)
            X_binned[:, j] = bj
            edges_by_feat[col] = edges
            bins_by_feat[col] = int(edges.size - 1)

        meta = dict(
            strategy="quantile",
            edges=edges_by_feat,
            bins_per_feature=bins_by_feat,
            total_bins=int(sum(bins_by_feat.values())),
        )
        return X_binned, meta

    def _build_bins_uniform(self, X_df: pd.DataFrame, *, n_bins: int) -> Tuple[np.ndarray, Dict]:
        edges_by_feat: Dict[str, np.ndarray] = {}
        bins_by_feat: Dict[str, int] = {}
        X_binned = np.zeros((X_df.shape[0], X_df.shape[1]), dtype=np.int32)

        for j, col in enumerate(X_df.columns):
            s = X_df[col].to_numpy()
            uniq = np.unique(s[~pd.isna(s)])
            if uniq.size <= 1:
                edges = np.array([-np.inf, np.inf], dtype=np.float64)
            elif uniq.size == 2 and set(uniq).issubset({0, 1}):
                edges = np.array([-0.5, 0.5, 1.5], dtype=np.float64)
            else:
                lo = np.nanmin(s.astype(np.float64))
                hi = np.nanmax(s.astype(np.float64))
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    edges = np.array([-np.inf, np.inf], dtype=np.float64)
                else:
                    edges = np.linspace(lo, hi, num=n_bins + 1, dtype=np.float64)
                    edges[0] = np.floor(edges[0] - 1e-9)
                    edges[-1] = np.ceil(edges[-1] + 1e-9)

            bj = np.searchsorted(edges, s, side="right") - 1
            bj = np.clip(bj, 0, edges.size - 2).astype(np.int32)
            X_binned[:, j] = bj
            edges_by_feat[col] = edges
            bins_by_feat[col] = int(edges.size - 1)

        meta = dict(
            strategy="global_uniform",
            edges=edges_by_feat,
            bins_per_feature=bins_by_feat,
            total_bins=int(sum(bins_by_feat.values())),
        )
        return X_binned, meta

    # ======================================================================
    # Inference featurisation (re-bin test set with learned edges)
    # ======================================================================
    def _featurise(
        self,
        df_test: pd.DataFrame,
        df_train: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        n_bins: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Re-bin test data using the train-time edges in self.bin_meta.
        Signature kept for backward-compat with older trainer code.
        """
        cols = self.feature_cols if feature_cols is None else feature_cols
        edges_by_feat = self.bin_meta["edges"]
        X_df = df_test.loc[:, cols].copy()

        X_binned = np.zeros((X_df.shape[0], len(cols)), dtype=np.int32)
        for j, col in enumerate(cols):
            s = X_df[col].to_numpy()
            edges = edges_by_feat[col]
            bj = np.searchsorted(edges, s, side="right") - 1
            bj = np.clip(bj, 0, edges.size - 2).astype(np.int32)
            X_binned[:, j] = bj

        return torch.from_numpy(X_binned.astype(np.int16, copy=False)).to(self.device)

    # ======================================================================
    # Rollout helpers used by Trainer
    # ======================================================================
    def draw_indices(self, batch_size: int) -> torch.Tensor:
        """
        Sample a mini-batch of row indices (without replacement if possible).
        """
        N = self.X_full.size(0)
        b = int(batch_size)
        if b <= 0:
            b = N
        if b >= N:
            # use full dataset
            return torch.arange(N, device=self.device, dtype=torch.long)
        # without replacement when feasible
        idx = torch.randperm(N, device=self.device)[:b]
        return idx

    def reset(self, batch_size: int) -> None:
        """
        Reset rollout bookkeeping for a fresh trajectory on a (possibly) new subset.
        Trainer sometimes calls this with len(target) for reward eval.
        """
        self._batch_size = int(batch_size)
        self.paths = []
        self.open_leaves = 1
        self.done = False
        self._n_splits = 0

    def step(self, action: Tuple[str, int]) -> None:
        """
        Update simple structure stats for a grammar-like prior.
          action = ('leaf', 0)  → close one open leaf
                 = ('feat', f)  → intent to split (no change)
                 = ('th', t)    → commit split → open_leaves += 1, n_splits += 1
        """
        kind, _ = action
        if kind == "leaf":
            self.open_leaves = max(0, self.open_leaves - 1)
            if self.open_leaves == 0:
                self.done = True
        elif kind == "feat":
            # no-op for prior counters; threshold will decide split
            pass
        elif kind == "th":
            # splitting one leaf into two: net +1 open leaf
            self.open_leaves += 1
            self._n_splits += 1

    def get_prior(self, beta: float) -> torch.Tensor:
        """
        Simple structure prior: exp(-beta * #splits).
        You can enrich this if you want to encode depth penalties, etc.
        """
        val = float(np.exp(-float(beta) * float(self._n_splits)))
        return torch.tensor(val, device=self.device, dtype=torch.float32)
