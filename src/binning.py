"""
Binning utilities for DT-GFN: supports both global uniform and per-feature quantile schemes.

Classes:
  - BinConfig: holds binning parameters
  - Binner: fit/transform/inverse_transform for DataFrames
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, List


@dataclass
class BinConfig:
    """
    Configuration for discretization.

    Attributes:
        n_bins: number of discrete bins (>=2)
        strategy: 'global_uniform' or 'feature_quantile'
        clip_outliers: whether to replace inf/NaN with median
    """
    n_bins: int = 128
    strategy: str = "global_uniform"
    clip_outliers: bool = True

    def validate(self) -> None:
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2.")
        if self.strategy not in {"global_uniform", "feature_quantile"}:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")


class Binner:
    """
    Lightweight binning helper (sklearn-like API) that stores cut-points.
    """

    def __init__(self, config: BinConfig | None = None):
        self.cfg = config or BinConfig()
        self.cfg.validate()
        self.edges_: Dict[str, np.ndarray] | None = None
        self._mins: pd.Series | None = None
        self._rng: pd.Series | None = None

    def fit(self, df: pd.DataFrame, columns: List[str] | None = None) -> Binner:
        """
        Fit the binner: compute edges based on df and strategy.
        """
        cols = columns or df.columns.tolist()
        x = df[cols].copy()

        # Handle outliers
        if self.cfg.clip_outliers:
            for c in cols:
                med = x[c].replace([np.inf, -np.inf], np.nan).median()
                x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(med)

        # Min-max scale
        mins = x.min()
        maxs = x.max()
        rng = (maxs - mins).replace(0, 1)
        x_scaled = (x - mins) / rng

        edges: Dict[str, np.ndarray] = {}

        if self.cfg.strategy == "global_uniform":
            global_edges = np.linspace(0.0, 1.0, self.cfg.n_bins + 1)
            for c in cols:
                edges[c] = global_edges.copy()
        else:  # feature_quantile
            for c in cols:
                qs = np.linspace(0.0, 1.0, self.cfg.n_bins + 1)
                raw = np.quantile(x_scaled[c].values, qs)
                uniq = np.unique(raw)
                uniq[0]  -= 1e-9
                uniq[-1] += 1e-9
                edges[c] = uniq

        self.edges_ = edges
        self._mins = mins
        self._rng = rng
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted edges to df, returning integer bin indices DataFrame.
        """
        if self.edges_ is None or self._mins is None or self._rng is None:
            raise RuntimeError("Must call fit before transform.")
        result = {}
        for c, e in self.edges_.items():
            col = df[c].replace([np.inf, -np.inf], np.nan).fillna(df[c].median())
            col_scaled = (col - self._mins[c]) / self._rng[c]
            result[c] = np.searchsorted(e, col_scaled.values, side='right') - 1
        return pd.DataFrame(result, index=df.index, dtype=np.int32)

    def inverse_transform(self, df_bins: pd.DataFrame) -> pd.DataFrame:
        """
        Approximate original values by mapping bin indices to bin centers.
        """
        if self.edges_ is None or self._mins is None or self._rng is None:
            raise RuntimeError("Must call fit before inverse_transform.")
        result = {}
        for c, e in self.edges_.items():
            idx = df_bins[c].clip(0, len(e) - 2).values
            centers = (e[idx] + e[idx + 1]) / 2
            result[c] = centers * self._rng[c] + self._mins[c]
        return pd.DataFrame(result, index=df_bins.index)

    def fit_transform(self, df: pd.DataFrame, columns: List[str] | None = None) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)

    def describe(self) -> str:
        if self.edges_ is None:
            return "<Binner (not fitted)>"
        return f"<Binner strategy={self.cfg.strategy!r} bins={self.cfg.n_bins} features={len(self.edges_)}>"
