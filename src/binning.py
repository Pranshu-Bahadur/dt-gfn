# dtgfn/binning.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BinConfig:
    n_bins: int = 128                  # number of discrete levels
    strategy: str = "global_uniform"   # or "feature_quantile"
    clip_outliers: bool = True         # clip ±inf & NaN to feature median

    def validate(self) -> None:
        if self.n_bins < 2:
            raise ValueError("n_bins must be ≥ 2.")
        if self.strategy not in {"global_uniform", "feature_quantile"}:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")


class Binner:
    """
    Standalone binning helper.  Fit on a DataFrame to collect edges,
    then transform other frames into integer bins [0..n_bins-1].
    """

    def __init__(self, config: BinConfig | None = None):
        self.cfg = config or BinConfig()
        self.cfg.validate()
        self.edges_: dict[str, np.ndarray] | None = None

    def fit(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> Binner:
        cols = columns or df.columns.tolist()
        x = df[cols].copy()

        # clip outliers
        if self.cfg.clip_outliers:
            for c in cols:
                med = x[c].replace([np.inf, -np.inf], np.nan).median()
                x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(med)

        # min-max scale each column to [0,1]
        mins, maxs = x.min(), x.max()
        rng = (maxs - mins).replace(0, 1)
        x = (x - mins) / rng

        edges: dict[str, np.ndarray] = {}
        if self.cfg.strategy == "global_uniform":
            # one set of edges for all features
            global_edges = np.linspace(0.0, 1.0, self.cfg.n_bins + 1)
            for c in cols:
                edges[c] = global_edges
        else:  # feature_quantile
            for c in cols:
                qs = np.linspace(0.0, 1.0, self.cfg.n_bins + 1)
                e = np.quantile(x[c].values, qs, method="linear")
                e = np.unique(e)
                e[0] -= 1e-9
                e[-1] += 1e-9
                edges[c] = e

        self.edges_ = edges
        self._mins = mins
        self._rng = rng
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.edges_ is None:
            raise RuntimeError("Call `fit` before `transform`.")
        out = {}
        max_bin = self.cfg.n_bins - 1
        for c, e in self.edges_.items():
            # apply same clipping as in fit
            col = df[c].replace([np.inf, -np.inf], np.nan).fillna(df[c].median())
            # same min-max scale
            scaled = (col - self._mins[c]) / self._rng[c]
            # assign bins, then clamp to [0, n_bins-1]
            idx = np.searchsorted(e, scaled.values, side="right") - 1
            idx = np.clip(idx, 0, max_bin)
            out[c] = idx.astype(np.int32)
        return pd.DataFrame(out, index=df.index, dtype=np.int32)

    def inverse_transform(self, df_bins: pd.DataFrame) -> pd.DataFrame:
        if self.edges_ is None:
            raise RuntimeError("Call `fit` before `inverse_transform`.")
        out = {}
        for c, e in self.edges_.items():
            idx = df_bins[c].clip(0, len(e) - 2).values
            # midpoint of each bin interval (rescale back)
            mid = (e[idx] + e[idx + 1]) / 2
            out[c] = (mid * self._rng[c] + self._mins[c])
        return pd.DataFrame(out, index=df_bins.index)

    def fit_transform(self, df: pd.DataFrame, *, columns: list[str] | None = None) -> pd.DataFrame:
        return self.fit(df, columns=columns).transform(df)

    def describe(self) -> str:
        if self.edges_ is None:
            return "<Binner (not fitted)>"
        nfeat = len(self.edges_)
        return f"<Binner strategy={self.cfg.strategy!r} bins={self.cfg.n_bins} features={nfeat}>"
