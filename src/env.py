# src/env.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# New: per-feature binner
# ------------------------------------------------------------
class FeatureBinner:
    """
    Per-feature binning with auto type detection.

    Types:
      - constant    -> single bin (all zeros)
      - binary 0/1  -> 2 bins via edges [-0.5, 0.5, 1.5] (threshold t=0 exists)
      - categorical (low-card) -> label-map to [0..K-1]
      - continuous  -> quantile edges (optionally LGBM-ish: subsample + min_data_in_bin)

    Produces integer bins in [0 .. n_bins_f-1], with n_bins_f <= max_bins.
    """
    def __init__(
        self,
        features: List[str],
        *,
        max_bins: int = 255,
        strategy: str = "per_feature_auto",  # "per_feature_auto" | "quantile" | "uniform"
        per_feature_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        subsample_for_bin: int = 200_000,
        min_data_in_bin: int = 1,
        max_unique_for_cat: int = 64,
    ):
        self.features = list(features)
        self.max_bins = int(max_bins)
        self.strategy = strategy
        self.per_feature_overrides = per_feature_overrides or {}
        self.subsample_for_bin = int(subsample_for_bin)
        self.min_data_in_bin = int(min_data_in_bin)
        self.max_unique_for_cat = int(max_unique_for_cat)

        self.meta: Dict[str, Dict[str, Any]] = {}   # f -> {"type":..., ...}
        self.edges: Dict[str, np.ndarray] = {}      # f -> edges (for continuous/binary)
        self.cat_map: Dict[str, Dict[Any, int]] = {}# f -> cat->idx (for categorical)

    def _infer_type(self, s: pd.Series) -> str:
        x = s.dropna()
        if x.nunique(dropna=True) <= 1:
            return "constant"
        # Strict binary 0/1?
        u = pd.unique(x)
        if set(pd.unique(x)) <= {0, 1}:
            return "binary"
        # Int-like or object with low cardinality -> categorical
        if (np.issubdtype(x.dtype, np.integer) or x.dtype == object) and x.nunique() <= self.max_unique_for_cat:
            return "categorical"
        # Else continuous
        return "continuous"

    def _quantile_edges(self, arr: np.ndarray, n_bins: int) -> np.ndarray:
        n_bins = max(2, min(n_bins, self.max_bins))
        qs = np.linspace(0, 1, num=n_bins + 1)
        # unique edges to avoid zero-width bins
        edges = np.unique(np.quantile(arr, qs, method="linear"))
        # ensure at least 2 edges
        if edges.size < 2:
            v = arr[0] if arr.size else 0.0
            edges = np.array([v - 1e-9, v + 1e-9], dtype=float)
        # widen extremes
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        return edges

    def _merge_small_bins(self, vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Merge adjacent bins until each has >= min_data_in_bin, greedy.
        """
        if self.min_data_in_bin <= 1 or vals.size == 0:
            return edges
        while True:
            idx = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, len(edges) - 2)
            counts = np.bincount(idx, minlength=len(edges) - 1)
            too_small = np.where(counts < self.min_data_in_bin)[0]
            if too_small.size == 0 or len(edges) <= 3:
                break
            i = int(too_small[0])
            # Merge with the larger neighbor to keep balance
            left_cnt = counts[i - 1] if i - 1 >= 0 else -1
            right_cnt = counts[i + 1] if i + 1 < len(counts) else -1
            if right_cnt >= left_cnt and i + 1 < len(edges) - 1:
                # merge bin i into i+1 -> drop edges[i+1]
                edges = np.delete(edges, i + 1)
            else:
                # merge bin i into i-1 -> drop edges[i]
                edges = np.delete(edges, i)
        return edges

    def fit(self, df: pd.DataFrame):
        for f in self.features:
            s = df[f]
            # clean NaN/inf
            x = pd.to_numeric(s, errors="coerce").astype(float)
            med = float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0
            x = x.replace([np.inf, -np.inf], np.nan).fillna(med)

            ov = self.per_feature_overrides.get(f, {})
            kind = ov.get("type")
            if kind is None:
                if self.strategy == "quantile":
                    kind = "continuous"
                elif self.strategy == "uniform":
                    kind = "continuous"  # we still use quantile edges for robustness
                else:
                    kind = self._infer_type(s)

            meta = {"type": kind}
            if kind == "constant":
                v = float(x.iloc[0])
                edges = np.array([v - 1e-9, v + 1e-9], dtype=float)
                self.edges[f] = edges
            elif kind == "binary":
                # 2 bins with a single threshold t=0
                self.edges[f] = np.array([-0.5, 0.5, 1.5], dtype=float)
            elif kind == "categorical":
                cats = pd.unique(s.fillna("__NA__"))
                if len(cats) > self.max_unique_for_cat:
                    # fallback to continuous if too many categories
                    arr = x.to_numpy()
                    sample = arr if arr.size <= self.subsample_for_bin else np.random.choice(arr, self.subsample_for_bin, replace=False)
                    edges = self._quantile_edges(sample, self.max_bins)
                    edges = self._merge_small_bins(arr, edges)
                    self.edges[f] = edges
                    meta["type"] = "continuous"
                else:
                    mapping = {c: i for i, c in enumerate(cats)}
                    self.cat_map[f] = mapping
            else:
                # continuous: quantile edges, with LGBM-ish options
                arr = x.to_numpy()
                sample = arr if arr.size <= self.subsample_for_bin else np.random.choice(arr, self.subsample_for_bin, replace=False)
                n_req = int(ov.get("n_bins", self.max_bins))
                edges = self._quantile_edges(sample, n_req)
                edges = self._merge_small_bins(arr, edges)
                # ensure cap by max_bins
                while (len(edges) - 1) > self.max_bins:
                    # drop every second interior edge
                    keep = np.ones(len(edges), dtype=bool)
                    keep[1:-1:2] = False
                    edges = edges[keep]
                self.edges[f] = edges

            self.meta[f] = meta

    def transform(self, df: pd.DataFrame, device: Optional[torch.device] = None) -> torch.Tensor:
        out = np.zeros((len(df), len(self.features)), dtype=np.int64)
        for j, f in enumerate(self.features):
            meta = self.meta[f]
            if meta["type"] == "categorical":
                m = self.cat_map[f]
                s = df[f].fillna("__NA__")
                out[:, j] = s.map(m).fillna(0).astype(int).to_numpy()
            else:
                x = pd.to_numeric(df[f], errors="coerce").astype(float)
                med = float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0
                x = x.replace([np.inf, -np.inf], np.nan).fillna(med)
                edges = self.edges[f]
                idx = np.searchsorted(edges, x.to_numpy(), side="right") - 1
                idx = np.clip(idx, 0, len(edges) - 2)
                out[:, j] = idx
        t = torch.from_numpy(out).long()
        if device is not None:
            t = t.to(device)
        return t

    def n_bins_per_feature(self) -> List[int]:
        nb = []
        for f in self.features:
            if self.meta[f]["type"] == "categorical":
                nb.append(max(1, len(self.cat_map[f])))
            else:
                nb.append(max(1, len(self.edges[f]) - 1))
        return nb


# ------------------------------------------------------------
# TabularEnv using the binner
# ------------------------------------------------------------
class TabularEnv:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_bins: int,
        task: str,
        *,
        binning_strategy: str = "per_feature_auto",
        device: str = "cpu",
        # new optional knobs (forwarded by trainer)
        per_feature_binning: Optional[Dict[str, Dict[str, Any]]] = None,
        subsample_for_bin: int = 200_000,
        min_data_in_bin: int = 1,
        max_unique_for_cat: int = 64,
    ):
        self.df = df
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.task = task
        self.device = torch.device(device)

        # Fit label encoder / target
        if self.task == "classification":
            le = LabelEncoder()
            y = le.fit_transform(df[target_col].to_numpy())
            self.le = le
            self.n_classes = int(len(le.classes_))
            self.y_full = torch.from_numpy(y).long().to(self.device)
        else:
            y = pd.to_numeric(df[target_col], errors="coerce").astype(float)
            y = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())
            self.le = None
            self.n_classes = None
            self.y_full = torch.from_numpy(y.to_numpy()).float().to(self.device)

        # Per-feature binner
        self.binner = FeatureBinner(
            self.feature_cols,
            max_bins=n_bins,
            strategy=binning_strategy if binning_strategy else "per_feature_auto",
            per_feature_overrides=per_feature_binning,
            subsample_for_bin=subsample_for_bin,
            min_data_in_bin=min_data_in_bin,
            max_unique_for_cat=max_unique_for_cat,
        )
        self.binner.fit(df[self.feature_cols])

        # Binned X
        self.X_full = self.binner.transform(df[self.feature_cols], device=self.device)

        # initial sampler state
        self.device = torch.device(device)
        self.paths = []
        self.open_leaves = 1
        self.done = False
        self._N = len(self.y_full)

        # Expose these so trainer can read
        self.n_bins = n_bins  # global cap (tokenizer uses this)
        self.n_th_per_feature = self.binner.n_bins_per_feature()

        # sample indices used during rollouts/rewards
        self.idxs = torch.arange(self._N, device=self.device)

    def reset(self, batch_size: int):
        self.paths = []
        self.open_leaves = 1
        self.done = False
        return self.draw_indices(batch_size)

    def draw_indices(self, batch_size: int) -> torch.Tensor:
        # robust bootstrap
        N = self._N
        if N <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        return torch.randint(0, N, (batch_size,), device=self.device)

    # (trainer expects these)
    def step(self, tok_tuple):
        self.paths.append(tok_tuple)

    def get_prior(self, beta: float) -> torch.Tensor:
        # simple length prior; trainer multiplies by exp(-beta * #splits / N)
        return torch.tensor(1.0, device=self.device)

    # called by trainer.predict() to bin test with train edges
    def _featurise(self, df_test: pd.DataFrame, df_train: pd.DataFrame, feature_cols: List[str], n_bins: int) -> torch.Tensor:
        _ = df_train  # not needed anymore; kept for API compatibility
        return self.binner.transform(df_test[feature_cols], device=self.device)
