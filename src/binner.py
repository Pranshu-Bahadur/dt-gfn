"""binner.py – Fast, global discretiser for DT‑GFN
==================================================
This module converts **continuous** feature matrices into **integer bin
IDs** so that GFlowNet logic can operate on integers only.  It
*automatically detects* if the incoming dataframe is **already
pre‑binned** (e.g. Numerai’s `*_bin` features) and, if so, skips the
quantile procedure entirely.

Public API
----------
::

    binner = Binner()
    binner.fit(df_train)        # learns edges *or* records pre‑binned cfg
    X_bin  = binner.transform(df_live)
    thr    = binner.threshold(feature_id, bin_id)  # float upper edge

Implementation notes
--------------------
* **Detection rule** – A column is considered *pre‑binned* when:
  1. dtype is *integer* *or* *categorical*, **and**
  2. values form the contiguous set `{0, 1, …, K‑1}` for some K ≤ 512.
  When *all* columns satisfy this, the dataset is treated as pre‑binned.
* **Edge construction** – For pre‑binned data, edges are
  `[‑inf, 0.5, 1.5, …, K‑0.5, +inf]` so that `bucketize` mirrors the
  identity mapping.
* **Non‑pre‑binned flow** – Falls back to per‑feature quantile binning
  (default 64 bins) using numpy, storing edges as a single tensor of
  shape ``[F, B+1]``.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import torch

__all__ = ["Binner"]


@dataclass(slots=True, frozen=True)
class Binner:
    n_bins: int = 64
    edges: torch.Tensor | None = None  # shape [F, B+1]
    pre_binned: bool = False

    # ----------------------------- fit ---------------------------------
    def fit(self, X: pd.DataFrame) -> "Binner":
        """Learns global bin edges or records pre‑binned config.

        *Detection* happens once here – the transform step assumes the
        same regime (continuous vs pre‑binned).
        """
        col_is_pre = X.apply(self._is_prebinned_col)
        if col_is_pre.all():
            object.__setattr__(self, "pre_binned", True)
            per_feat_bins = X.apply(lambda col: col.max() + 1)
            max_bins = per_feat_bins.max()
            # Build a ragged edge tensor by padding with +inf at the end
            edge_list = []
            for bins in per_feat_bins:
                e = torch.tensor(
                    [-torch.inf] + [i + 0.5 for i in range(bins)] + [torch.inf]
                )
                if bins < max_bins:
                    pad = torch.full((max_bins - bins,), torch.inf)
                    e = torch.cat([e, pad])
                edge_list.append(e)
            edges = torch.stack(edge_list, dim=0)  # [F, max_bins+2]
            object.__setattr__(self, "edges", edges)
            # record the *largest* K so transform() can bucketize quickly
            object.__setattr__(self, "n_bins", int(max_bins))
            return self

        # -------- standard quantile binning --------
        quant_edges = []
        for col in X.columns:
            # numpy quantile is fast and works on float32
            qs = np.linspace(0.0, 1.0, self.n_bins + 1, dtype=np.float32)
            edges = np.quantile(X[col].to_numpy(dtype=np.float32), qs, interpolation="midpoint")
            # ensure strict monotonicity (possible duplicates on flat areas)
            edges = np.unique(edges)
            # prepend -inf, append +inf
            edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf])).astype(np.float32)
            quant_edges.append(torch.from_numpy(edges))
        self_edges = torch.stack(quant_edges, dim=0)  # shape [F, B+1]
        object.__setattr__(self, "edges", self_edges)
        object.__setattr__(self, "pre_binned", False)
        return self

    # --------------------------- transform -----------------------------
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> torch.LongTensor:
        """Returns ``[N, F]`` tensor of *bin IDs* (ints)."""
        assert self.edges is not None, "Binner not fitted. Call fit() first."
        if self.pre_binned:
            # Fast path: assume values are already 0..K-1 and correct.
            return torch.as_tensor(X.to_numpy(copy=False), dtype=torch.long)

        # Continuous path – bucketize per feature
        X_arr = torch.as_tensor(X.to_numpy(dtype=np.float32, copy=False))
        bin_ids = []
        for f in range(X_arr.shape[1]):
            ids = torch.bucketize(X_arr[:, f], self.edges[f]) - 1  # shift because of -inf pad
            bin_ids.append(ids.unsqueeze(1))
        return torch.cat(bin_ids, dim=1)

    # -------------------- helper: threshold lookup ---------------------
    def threshold(self, feature_id: int, bin_id: int) -> float:
        """Returns the *upper edge* as float for logging / debugging."""
        assert self.edges is not None, "Binner not fitted."
        return float(self.edges[feature_id, bin_id + 1])

    # ---------------------- static utilities ---------------------------
    @staticmethod
    def _is_prebinned_col(col: pd.Series) -> bool:
        if not pd.api.types.is_integer_dtype(col):
            return False
        vals = col.to_numpy(copy=False)
        if vals.min() != 0:
            return False
        K = vals.max() + 1
        if K > 512:  # arbitrary safety upper‑bound
            return False
        # contiguous check: all ints from 0 to K‑1 appear at least once
        return len(np.unique(vals)) == K
"

