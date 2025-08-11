from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch


def _is_binary(col: pd.Series) -> bool:
    """True if the column is strictly 0/1 (ignoring NaNs)."""
    vals = pd.unique(col.dropna().astype(float))
    if len(vals) == 0:
        return False
    if len(vals) > 2:
        return False
    return set(vals).issubset({0.0, 1.0})


@dataclass
class _Bins:
    """Stores bin edges per feature and binning meta."""
    edges: Dict[str, np.ndarray]            # feature -> 1D edges of length n_bins+1
    is_binary: Dict[str, bool]              # feature -> is 0/1
    order: List[str]                        # final feature order


class TabularEnv:
    """
    Minimal environment used during training-time rollouts and reward computation.
    - Holds binned X and encoded y over the full training set
    - Draws row indices for sub-batches
    - Tracks a light-weight per-episode state (paths/open leaves)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_bins: int,
        task: str,
        binning_strategy: str = "quantile",
        device: str = "cuda",
    ):
        assert target_col in df.columns, f"target column '{target_col}' not found"
        self.device = torch.device(device)
        self.task = str(task)
        self.n_bins = int(n_bins)
        self.binning_strategy = str(binning_strategy)
        self.target_col = target_col

        # Freeze feature order
        self.feature_cols: List[str] = [c for c in feature_cols if c != target_col]
        self.feature_cols = list(dict.fromkeys(self.feature_cols))  # dedupe, keep order

        # Fit binning on df[feature_cols] and featurise to int bins
        self._bins = self._fit_bins(df[self.feature_cols], n_bins, binning_strategy)
        Xb = self._bin_df(df[self.feature_cols], self._bins)  # (N, F) long
        self.X_full: torch.Tensor = torch.as_tensor(Xb, dtype=torch.long, device=self.device)

        # Encode target
        y = df[target_col]
        if self.task == "classification":
            # numeric or categorical → 0..C-1
            cats = pd.Categorical(y)
            self.le_categories = np.asarray(cats.categories)
            y_enc = cats.codes.astype(np.int64)
            if np.any(y_enc < 0):
                # Unseen/NaN → put into a dummy class 0 (rare on train)
                y_enc = np.where(y_enc < 0, 0, y_enc)
            self.le = None                      # kept for compatibility
            self.n_classes = int(y_enc.max() + 1)
            self.y_full: torch.Tensor = torch.as_tensor(y_enc, dtype=torch.long, device=self.device)
        else:
            self.le = None
            self.le_categories = None
            self.n_classes = None
            self.y_full: torch.Tensor = torch.as_tensor(y.values, dtype=torch.float32, device=self.device)

        # Episode state
        self.idxs: torch.Tensor = torch.arange(self.X_full.size(0), device=self.device)
        self.open_leaves: int = 1
        self.done: bool = False
        self.paths: List[Tuple[str, int]] = []  # [('feat', f), ('th', t), ('leaf', 0), ...]

        # A tiny sampler state for draw_indices
        self._sampler_pos: int = 0
        self._perm: Optional[torch.Tensor] = None

    # ---------------- Binning ----------------
    def _fit_bins(self, X: pd.DataFrame, n_bins: int, strategy: str) -> _Bins:
        edges = {}
        is_bin = {}
        order = list(X.columns)

        if strategy not in ("quantile", "global_uniform"):
            strategy = "quantile"

        for c in order:
            col = pd.to_numeric(X[c], errors="coerce")
            m = col.notna()
            if not m.any():
                # degenerate -> constant 0
                edges[c] = np.array([0.0, 1.0], dtype=float)
                is_bin[c] = False
                continue

            if _is_binary(col[m]):
                # keep 0/1 exactly at bins 0 and 1; we still embed them in [0..n_bins-1]
                edges[c] = np.array([-0.5, 0.5, 1.5], dtype=float)  # 0 -> bin 0, 1 -> bin 1
                is_bin[c] = True
                continue

            x = col[m].astype(float).values
            if np.all(x == x[0]):
                edges[c] = np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
                is_bin[c] = False
                continue

            if strategy == "quantile":
                qs = np.linspace(0, 1, num=n_bins + 1)
                qv = np.quantile(x, qs)
                # Ensure strictly increasing edges
                qv = np.unique(qv)
                if qv.size < 3:
                    qv = np.array([x.min() - 1e-6, x.mean(), x.max() + 1e-6], dtype=float)
                edges[c] = qv
            else:
                lo, hi = float(np.min(x)), float(np.max(x))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = float(x.min()), float(x.min() + 1.0)
                edges[c] = np.linspace(lo, hi, num=n_bins + 1)
            is_bin[c] = False

        return _Bins(edges=edges, is_binary=is_bin, order=order)

    def _bin_df(self, X: pd.DataFrame, bins: _Bins) -> np.ndarray:
        X = X.copy()
        X = X[bins.order]
        out = np.zeros((len(X), len(bins.order)), dtype=np.int64)

        for j, c in enumerate(bins.order):
            col = pd.to_numeric(X[c], errors="coerce").astype(float)
            col = col.fillna(col.median())
            ed = bins.edges[c]

            # Use numpy digitize to get 1..len(ed)-1 → shift to 0-based
            b = np.clip(np.digitize(col.values, ed) - 1, 0, len(ed) - 2)
            out[:, j] = b

        return out

    # ---------------- Episode lifecycle ----------------
    def reset(self, batch_size: int):
        if self._perm is None or self._sampler_pos + batch_size > self.X_full.size(0):
            self._perm = torch.randperm(self.X_full.size(0), device=self.device)
            self._sampler_pos = 0
        self.idxs = self._perm[self._sampler_pos: self._sampler_pos + batch_size]
        self._sampler_pos += batch_size

        self.paths = []
        self.open_leaves = 1
        self.done = False

    def draw_indices(self, batch_size: int) -> torch.Tensor:
        """Draw disjoint shuffled indices (wrap-around if we exhaust)."""
        if self._perm is None or self._sampler_pos + batch_size > self.X_full.size(0):
            self._perm = torch.randperm(self.X_full.size(0), device=self.device)
            self._sampler_pos = 0
        idx = self._perm[self._sampler_pos: self._sampler_pos + batch_size]
        self._sampler_pos += batch_size
        return idx

    def step(self, action: Tuple[str, int]):
        """No heavy effects; just record the path & track open leaves for convenience."""
        kind, val = action
        self.paths.append((kind, int(val)))
        if kind == "leaf":
            self.open_leaves = max(0, self.open_leaves - 1)
            if self.open_leaves == 0:
                self.done = True
        elif kind == "th":
            # each threshold creates two children
            self.open_leaves += 1

    def get_prior(self, beta: float) -> torch.Tensor:
        """Simple structure prior ~ exp(-beta * (#internal splits))."""
        n_splits = sum(1 for k, _ in self.paths if k == "th")
        return torch.exp(torch.tensor(-beta * float(n_splits), device=self.device))

    # --------------- Inference featurisation ---------------
    def _featurise(
        self,
        df_test: pd.DataFrame,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        n_bins: int,
    ) -> torch.Tensor:
        # Use fitted bins; silently drop unseen columns; ignore extra columns
        cols = [c for c in feature_cols if c in self._bins.order]
        Xt = df_test[cols].copy()
        Xb = self._bin_df(Xt, self._bins)
        return torch.as_tensor(Xb, dtype=torch.long, device=self.device)
