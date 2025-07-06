"""binner.py – Fast, global discretiser for DT‑GFN."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

__all__ = ["Binner"]

@dataclass
class Binner:
    """
    Converts continuous features into integer bins. Automatically detects
    and handles pre-binned integer features.
    """
    n_bins: int = 64
    _edges: torch.Tensor | None = None
    _is_pre_binned: bool = False

    def fit(self, X: pd.DataFrame) -> "Binner":
        """Learns bin edges from data or detects pre-binned format."""
        if all(self._is_prebinned_col(X[col]) for col in X.columns):
            self._fit_pre_binned(X)
        else:
            self._fit_continuous(X)
        return self

    def _fit_pre_binned(self, X: pd.DataFrame):
        self._is_pre_binned = True
        max_bins = int(X.max().max()) + 1
        self.n_bins = max_bins
        # For pre-binned, edges are just midpoints for potential inversion
        self._edges = torch.arange(-0.5, max_bins, 1.0).unsqueeze(0).repeat(X.shape[1], 1)

    def _fit_continuous(self, X: pd.DataFrame):
        self._is_pre_binned = False
        all_edges = []
        for col in X.columns:
            quantiles = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.int8)
            # Use PyTorch for quantile calculation for consistency
            col_tensor = torch.from_numpy(X[col].to_numpy(dtype=np.int8))
            edges = torch.quantile(col_tensor, torch.from_numpy(quantiles), interpolation='linear')
            edges = torch.unique(edges)
            edges[0] = -torch.inf
            edges[-1] = torch.inf
            all_edges.append(edges)
        
        # Pad shorter edge tensors to match the longest one
        max_len = max(len(e) for e in all_edges)
        padded_edges = [torch.nn.functional.pad(e, (0, max_len - len(e)), 'constant', float('inf')) for e in all_edges]
        self._edges = torch.stack(padded_edges, dim=0)

    def transform(self, X: pd.DataFrame | np.ndarray) -> torch.Tensor:
        """Transforms data into integer bin IDs."""
        if self._edges is None:
            raise RuntimeError("Binner has not been fitted. Call fit() first.")
        
        X_np = X.to_numpy(dtype=np.int8) if isinstance(X, pd.DataFrame) else X
        X_tensor = torch.from_numpy(X_np)

        if self._is_pre_binned:
            return X_tensor.to(torch.int8)

        binned_cols = []
        for i in range(X_tensor.shape[1]):
            # Subtract 1 because bucketize is 1-indexed relative to the bins
            ids = torch.bucketize(X_tensor[:, i], self._edges[i, :], right=False) - 1
            binned_cols.append(ids.clamp(0, self.n_bins - 1))
        
        return torch.stack(binned_cols, dim=1)

    def threshold(self, feature_id: int, bin_id: int) -> float:
        """Returns the upper edge of a bin for a given feature."""
        if self._edges is None:
            raise RuntimeError("Binner has not been fitted.")
        # The bin_id corresponds to the left edge, so bin_id + 1 is the right edge
        return self._edges[feature_id, bin_id + 1].item()

    @staticmethod
    def _is_prebinned_col(col: pd.Series) -> bool:
        """Checks if a column appears to be pre-binned (0 to K-1 integers)."""
        if not pd.api.types.is_integer_dtype(col.dtype):
            return False
        unique_vals = col.unique()
        return unique_vals.min() == 0 and len(unique_vals) == unique_vals.max() + 1