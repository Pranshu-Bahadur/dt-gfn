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

        # Optional per-feature overrides:
        #   {"feature": {"type": "binary"/"continuous", "n_bins": K}}
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

        # Fast path: regression + already-integer pre-binned features (use int8, no re-binning)
        used_prebinned = False
        if self.task == "regression":
            prebinned_ok, X_prebinned_np, meta_prebin = self._try_prebinned_passthrough(X_df)
            if prebinned_ok:
                used_prebinned = True
                self.bin_meta = meta_prebin
                # int8 on train
                self.X_full = torch.from_numpy(X_prebinned_np.astype(np.int8, copy=False)).to(self.device)

        if not used_prebinned:
            strat = self.binning_strategy.lower()
            if strat in ("lgbm_like", "per_feature_auto", "quantile_per_feature"):
                X_binned_np, meta = self._build_bins_lgbm_like(
                    X_df,
                    max_bins=self.n_bins,
                    min_data_in_bin=self.min_data_in_bin,
                    subsample_for_bin=self.subsample_for_bin,
                    overrides=self.per_feature_binning,
                )
                self.bin_meta = meta  # edges per feature, bins per feature, totals
            elif strat in ("quantile",):
                X_binned_np, meta = self._build_bins_quantile(X_df, n_bins=self.n_bins)
                self.bin_meta = meta
            elif strat in ("global_uniform", "uniform"):
                X_binned_np, meta = self._build_bins_uniform(X_df, n_bins=self.n_bins)
                self.bin_meta = meta
            else:
                raise ValueError(f"Unknown binning_strategy='{self.binning_strategy}'.")

            # Standard tensor dtype for binned ints
            self.X_full = torch.from_numpy(X_binned_np.astype(np.int16, copy=False)).to(self.device)

        self.idxs = torch.arange(self.X_full.size(0), device=self.device, dtype=torch.long)

        # Expose per-feature bin counts as a tensor (for trainer window init, optional)
        self.feature_num_bins = torch.as_tensor(
            [self.bin_meta["bins_per_feature"][c] for c in self.feature_cols],
            device=self.device, dtype=torch.long
        )

        # ----- Rollout bookkeeping (mutable; trainer will copy envs) -----
        self.paths: List[Tuple] = []
        self.open_leaves: int = 1
        self.done: bool = False
        self._batch_size: int = 0
        self._n_splits: int = 0  # decision (split) count for structure prior

    # ======================================================================
    # Pre-binned passthrough (regression-only)
    # ======================================================================
    def _try_prebinned_passthrough(self, X_df: pd.DataFrame) -> Tuple[bool, Optional[np.ndarray], Optional[Dict]]:
        """
        Detect whether ALL features are integer-coded, non-negative, contiguous (0..K-1),
        and within int8 capacity (max ≤ 127). If yes, treat them as pre-binned and
        return (True, np.int8 array, meta). Otherwise (False, None, None).
        """
        N, F = X_df.shape
        bins_by_feat: Dict[str, int] = {}
        edges_by_feat: Dict[str, np.ndarray] = {}
        X_np = np.zeros((N, F), dtype=np.int8)

        for j, col in enumerate(X_df.columns):
            s_raw = X_df[col].to_numpy()
            s = s_raw.astype(np.float64)
            s = s[np.isfinite(s)]
            if s.size == 0:
                return False, None, None

            # integer-valued check
            if not np.all(np.mod(s, 1.0) == 0.0):
                return False, None, None

            s_full = X_df[col].to_numpy().astype(np.int64, copy=False)
            vmin = int(np.nanmin(s_full))
            vmax = int(np.nanmax(s_full))

            # non-negative, int8 capacity
            if vmin < 0 or vmax > 127:
                return False, None, None

            uniq = np.unique(s_full[~pd.isna(s_full)])
            # contiguous check: {0,1,...,K-1}
            if uniq.size == 0:
                return False, None, None
            if uniq[0] != 0 or uniq[-1] != (uniq.size - 1):
                return False, None, None
            if np.any(np.diff(uniq) != 1):
                return False, None, None

            # pass through; cast to int8 (safe due to vmax ≤ 127)
            X_np[:, j] = X_df[col].to_numpy().astype(np.int8, copy=False)

            K = int(vmax + 1)
            bins_by_feat[col] = K
            # Provide synthetic edges for completeness ([-0.5, 0.5, ..., K-0.5])
            edges_by_feat[col] = np.arange(-0.5, K + 0.5, 1.0, dtype=np.float64)

        total_bins_used = int(sum(b for b in bins_by_feat.values() if b > 1))
        n_used_feats = int(sum(1 for b in bins_by_feat.values() if b > 1))
        print(f"[Binning] (prebinned pass-through) Total Bins {total_bins_used}")
        print(f"[Binning] Number of data points in the train set: {N}, number of used features: {n_used_feats}")

        meta = dict(
            strategy="prebinned",
            edges=edges_by_feat,                 # synthetic, 1-step edges
            bins_per_feature=bins_by_feat,
            total_bins_used=total_bins_used,
            n_used_features=n_used_feats,
            total_bins_all=int(sum(bins_by_feat.values())),
        )
        return True, X_np, meta

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

        Prints (matching LightGBM semantics):
          [Binning] Total Bins <sum of bins over *used* features (bins > 1)>
          [Binning] Number of data points in the train set: N, number of used features: <count of bins > 1>
        """
        rng = np.random.default_rng(12345)
        N, F = X_df.shape

        # Subsample for cut computation (faster/robust for large N)
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

            # Optional explicit override
            forced_type = ov.get("type", None)
            forced_bins = int(ov.get("n_bins", max_bins)) if "n_bins" in ov else max_bins

            # ---- Detect constant/binary on full data (more stable than just sample) ----
            uniq_full = np.unique(s[~pd.isna(s)])
            is_constant = uniq_full.size <= 1
            is_binary01 = (uniq_full.size == 2) and set(uniq_full).issubset({0, 1})

            if forced_type == "binary":
                edges = np.array([-0.5, 0.5, 1.5], dtype=np.float64)
            elif forced_type == "continuous":
                edges = self._quantile_edges(ss, forced_bins)
                edges = self._merge_tiny_bins(ss, edges, min_data_in_bin, forced_bins)
            else:
                if is_constant:
                    edges = np.array([-np.inf, np.inf], dtype=np.float64)  # 1 bin (dropped as "unused")
                elif is_binary01:
                    edges = np.array([-0.5, 0.5, 1.5], dtype=np.float64)   # true binary (2 bins)
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

        # LightGBM logs: "Total Bins" and "used features" (exclude constant features)
        total_bins_used = int(sum(b for b in bins_by_feat.values() if b > 1))
        n_used_feats = int(sum(1 for b in bins_by_feat.values() if b > 1))

        print(f"[Binning] Total Bins {total_bins_used}")
        print(f"[Binning] Number of data points in the train set: {N}, number of used features: {n_used_feats}")

        meta = dict(
            strategy="lgbm_like",
            edges=edges_by_feat,
            bins_per_feature=bins_by_feat,
            total_bins_used=total_bins_used,
            n_used_features=n_used_feats,
            total_bins_all=int(sum(bins_by_feat.values())),
        )
        return X_binned, meta

    def _quantile_edges(self, sample_values: np.ndarray, max_bins: int) -> np.ndarray:
        xs = sample_values.astype(np.float64)
        xs = xs[np.isfinite(xs)]
        if xs.size == 0:
            return np.array([-np.inf, np.inf], dtype=np.float64)

        # Start with ≤ max_bins bins → ≤ max_bins+1 edges
        # Use unique quantiles to avoid degenerate duplicates on small data
        q = np.linspace(0.0, 1.0, num=min(max_bins, max(2, xs.size)) + 1)
        raw = np.unique(np.quantile(xs, q))

        # Cap edges to (max_bins + 1)
        if raw.size > (max_bins + 1):
            stride = max(1, raw.size // (max_bins + 1))
            raw = raw[::stride]
            if raw.size > (max_bins + 1):
                raw = raw[: (max_bins + 1)]

        # Widen outer edges a touch
        raw = raw.astype(np.float64, copy=False)
        raw[0] = np.floor(raw[0] - 1e-9)
        raw[-1] = np.ceil(raw[-1] + 1e-9)
        return raw

    def _merge_tiny_bins(
        self,
        sample_values: np.ndarray,
        edges: np.ndarray,
        min_data_in_bin: int,
        max_bins: int,
    ) -> np.ndarray:
        """Merge adjacent bins left→right until all sample hist counts ≥ min_data_in_bin.
        Also enforces a hard cap of `max_bins` bins by merging the smallest-adjacent pair."""
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

            # hard cap to max_bins → merge the smallest adjacent pair
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

        total_bins_used = int(sum(b for b in bins_by_feat.values() if b > 1))
        n_used_feats = int(sum(1 for b in bins_by_feat.values() if b > 1))
        print(f"[Binning] Total Bins {total_bins_used}")
        print(f"[Binning] Number of data points in the train set: {X_df.shape[0]}, number of used features: {n_used_feats}")

        meta = dict(
            strategy="quantile",
            edges=edges_by_feat,
            bins_per_feature=bins_by_feat,
            total_bins_used=total_bins_used,
            n_used_features=n_used_feats,
            total_bins_all=int(sum(bins_by_feat.values())),
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

        total_bins_used = int(sum(b for b in bins_by_feat.values() if b > 1))
        n_used_feats = int(sum(1 for b in bins_by_feat.values() if b > 1))
        print(f"[Binning] Total Bins {total_bins_used}")
        print(f"[Binning] Number of data points in the train set: {X_df.shape[0]}, number of used features: {n_used_feats}")

        meta = dict(
            strategy="global_uniform",
            edges=edges_by_feat,
            bins_per_feature=bins_by_feat,
            total_bins_used=total_bins_used,
            n_used_features=n_used_feats,
            total_bins_all=int(sum(bins_by_feat.values())),
        )
        return X_binned, meta

    # ======================================================================
    # Inference featurisation (re-bin test set with learned edges)
    # ======================================================================
    def _featurise(
        self,
        df_test: pd.DataFrame,
        df_train: Optional[pd.DataFrame] = None,   # kept for backward-compat
        feature_cols: Optional[List[str]] = None,  # ignored if None
        n_bins: Optional[int] = None,              # ignored; we use learned edges
    ) -> torch.Tensor:
        """
        Re-bin test data using the train-time edges in self.bin_meta.
        Signature kept for backward-compat with older trainer code.
        """
        cols = self.feature_cols if feature_cols is None else feature_cols
        strategy = self.bin_meta.get("strategy", "")

        # Pass-through path for pre-binned regression features
        if strategy == "prebinned":
            # Strict validation: integers, non-negative, within train bins
            X_df = df_test.loc[:, cols].copy()
            X_np = np.zeros((X_df.shape[0], len(cols)), dtype=np.int8)
            for j, col in enumerate(cols):
                s_full = X_df[col].to_numpy()
                if not np.all(np.isfinite(s_full)):
                    raise ValueError(f"Non-finite values in column '{col}' for prebinned inference.")
                if not np.all(np.mod(s_full, 1.0) == 0.0):
                    raise ValueError(f"Non-integer values in column '{col}' for prebinned inference.")
                s_full = s_full.astype(np.int64, copy=False)
                if np.min(s_full) < 0:
                    raise ValueError(f"Negative values in column '{col}' for prebinned inference.")
                K = int(self.bin_meta["bins_per_feature"][col])
                vmax = int(np.max(s_full))
                if vmax >= K:
                    raise ValueError(
                        f"Out-of-range bin in column '{col}' (got max {vmax}, expected < {K})."
                    )
                if vmax > 127:
                    raise ValueError(
                        f"Column '{col}' exceeds int8 capacity in inference (max {vmax})."
                    )
                X_np[:, j] = s_full.astype(np.int8, copy=False)
            return torch.from_numpy(X_np).to(self.device)

        # Normal binning path
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
        """Sample a mini-batch of row indices (without replacement if possible)."""
        N = self.X_full.size(0)
        b = int(batch_size)
        if b <= 0:
            b = N
        if b >= N:
            return torch.arange(N, device=self.device, dtype=torch.long)
        return torch.randperm(N, device=self.device)[:b]

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
            pass
        elif kind == "th":
            self.open_leaves += 1
            self._n_splits += 1

    def get_prior(self, beta: float) -> torch.Tensor:
        """Simple structure prior: exp(-beta * #splits)."""
        val = float(np.exp(-float(beta) * float(self._n_splits)))
        return torch.tensor(val, device=self.device, dtype=torch.float32)
