# src/env.py
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


class TabularEnv:
    """
    Environment wrapper that:
      • fits per-feature binning on train,
      • treats ≤2-unique columns as binary,
      • applies Exclusive Feature Bundling (post-binning),
      • exposes effective bins for the active representation (bundled or original),
      • provides X_full (binned → bundled if enabled) and y_full tensors,
      • supplies rollout helpers.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_bins: int,
        task: str,
        binning_strategy: str = "quantile",  # "quantile" | "global_uniform" | "lgbm_like"
        device: str = "cpu",
        # EFB
        use_efb: bool = True,
        efb_conflict_rate: float = 0.0,      # 0.0 = strict exclusivity
        efb_sort_by: str = "nnz",            # "nnz" | "var"
    ):
        assert task in ("classification", "regression")
        assert binning_strategy in ("quantile", "global_uniform", "lgbm_like")
        self.device = torch.device(device)
        self.task = task

        # Keep original schema for binning always
        self.orig_feature_cols = list(feature_cols)

        self.target_col = target_col
        self.n_bins = int(n_bins)
        self.binning_strategy = binning_strategy

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

        # fit bin edges on train
        self.bin_edges, per_feat_bins, self.binary_mask = self._fit_bin_edges(df_train[self.orig_feature_cols])
        self.per_feat_bins_orig = per_feat_bins.clone()

        # bin train (pre-EFB)
        Xb_np = self._bin_dataframe(df_train[self.orig_feature_cols])

        # EFB
        self.use_efb = bool(use_efb)
        self.efb_conflict_rate = float(efb_conflict_rate)
        self.efb_sort_by = str(efb_sort_by)

        if self.use_efb:
            self._build_efb_plan(Xb_np)               # uses *original* feature space
            Xb_np = self._apply_efb(Xb_np)            # map to bundled features
            self.per_bundle_bins = torch.as_tensor(self._bundle_bins, dtype=torch.long)
            self.per_feat_bins = self.per_bundle_bins # effective bins for active X representation
            self.feature_cols = [f"bundle_{i}" for i in range(len(self._bundles))]
            self.binary_mask_bundled = torch.zeros(len(self._bundles), dtype=torch.bool)
        else:
            self.per_feat_bins = self.per_feat_bins_orig
            self.feature_cols = list(self.orig_feature_cols)

        self.X_full = torch.as_tensor(Xb_np, device=self.device, dtype=torch.long)
        self.F = self.X_full.size(1)

        # rollout state
        self.paths: List = []
        self.open_leaves: int = 1
        self.done: bool = False
        self.idxs: torch.Tensor = torch.arange(self.X_full.size(0), device=self.device)
        self._n_splits: int = 0

    # ---------------- Binning ----------------

    @staticmethod
    def _two_unique_midpoint(u: np.ndarray) -> float:
        a, b = float(u[0]), float(u[1])
        return (a + b) * 0.5

    def _fit_bin_edges(self, Xdf: pd.DataFrame) -> Tuple[List[torch.Tensor], torch.LongTensor, torch.BoolTensor]:
        edges: List[torch.Tensor] = []
        eff_bins: List[int] = []
        binary_mask: List[bool] = []

        max_bins = max(1, int(self.n_bins))

        for col in Xdf.columns:
            x = pd.to_numeric(Xdf[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            x_nn = x[~np.isnan(x)]
            uniq = np.unique(x_nn)

            if uniq.size <= 1:
                edges.append(torch.tensor([], dtype=torch.float32))
                eff_bins.append(1)
                binary_mask.append(False)
                continue

            if uniq.size == 2:
                cut = self._two_unique_midpoint(uniq)
                edges.append(torch.tensor([cut], dtype=torch.float32))
                eff_bins.append(2)
                binary_mask.append(True)
                continue

            if self.binning_strategy == "global_uniform":
                lo, hi = float(np.nanmin(x_nn)), float(np.nanmax(x_nn))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    cuts = np.array([], dtype=np.float64)
                else:
                    cuts = np.linspace(lo, hi, num=max_bins + 1, endpoint=True)[1:-1].astype(np.float64)
                    cuts = np.unique(cuts)
                edges.append(torch.from_numpy(cuts.astype(np.float32)))
                eff_bins.append(int(cuts.size + 1))
                binary_mask.append(False)

            elif self.binning_strategy == "quantile":
                qs = np.linspace(0.0, 1.0, num=max_bins + 1, endpoint=True)[1:-1]
                if qs.size == 0:
                    cuts = np.array([], dtype=np.float64)
                else:
                    qv = np.nanquantile(x_nn, qs, method="linear").astype(np.float64)
                    cuts = np.unique(qv)
                    cuts = cuts[(cuts > uniq[0]) & (cuts < uniq[-1])]
                edges.append(torch.from_numpy(cuts.astype(np.float32)))
                eff_bins.append(int(cuts.size + 1))
                binary_mask.append(False)

            else:  # "lgbm_like"
                vals, cnts = np.unique(x_nn, return_counts=True)
                N = float(cnts.sum())
                if vals.size <= 1:
                    cuts = np.array([], dtype=np.float64)
                else:
                    cdf = np.cumsum(cnts) / N
                    targets = np.linspace(0.0, 1.0, num=max_bins + 1, endpoint=True)[1:-1]
                    idx = np.searchsorted(cdf, targets, side="left")
                    idx = np.clip(idx, 0, vals.size - 1)
                    cuts = np.unique(vals[idx].astype(np.float64))
                    cuts = cuts[(cuts > vals[0]) & (cuts < vals[-1])]
                edges.append(torch.from_numpy(cuts.astype(np.float32)))
                eff_bins.append(int(cuts.size + 1))
                binary_mask.append(False)

        per_feat_bins = torch.as_tensor(eff_bins, dtype=torch.long)
        binary_mask_t = torch.as_tensor(binary_mask, dtype=torch.bool)
        return edges, per_feat_bins, binary_mask_t

    def _bucketize_col(self, x: np.ndarray, cutpoints: torch.Tensor, is_binary: bool) -> np.ndarray:
        cp = cutpoints.detach().cpu().numpy()
        x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64, copy=False)

        if is_binary:
            thr = cp[0] if cp.size == 1 else 0.5
            out = (x > thr).astype(np.int64)
            out[np.isnan(x)] = 0
            return out

        if cp.size == 0:
            out = np.zeros_like(x, dtype=np.int64)
            out[np.isnan(x)] = 0
            return out

        out = np.digitize(x, cp, right=False).astype(np.int64)
        out[np.isnan(x)] = 0
        return out

    def _bin_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        Xnp = df.to_numpy(copy=False)
        D = Xnp.shape[1]
        out = np.empty((Xnp.shape[0], D), dtype=np.int64)
        for j, col in enumerate(df.columns):
            out[:, j] = self._bucketize_col(Xnp[:, j], self.bin_edges[j], bool(self.binary_mask[j]))
        return out

    # ---------------- EFB (post-binning on original space) ----------------

    def _build_efb_plan(self, Xb: np.ndarray) -> None:
        """
        Build bundles over original features using strict or near-strict exclusivity.
        Active entry = bin != mode(bin) for that feature.
        """
        n, d = Xb.shape
        self._efb_mode = np.zeros(d, dtype=np.int64)
        self._efb_maps: List[np.ndarray] = [None] * d
        self._efb_nnz = np.zeros(d, dtype=np.int64)

        # per-feature mode and remap (0 for mode, 1.. for others)
        for j in range(d):
            bj, cnt = np.unique(Xb[:, j], return_counts=True)
            mode_val = bj[np.argmax(cnt)]
            self._efb_mode[j] = int(mode_val)
            self._efb_nnz[j] = int((Xb[:, j] != mode_val).sum())

            Bj = int(self.per_feat_bins_orig[j].item())
            remap = np.zeros(Bj, dtype=np.int32)
            cur = 1
            for b in range(Bj):
                if b == mode_val:
                    remap[b] = 0
                else:
                    remap[b] = cur
                    cur += 1
            self._efb_maps[j] = remap

        active = [(Xb[:, j] != self._efb_mode[j]).astype(np.uint8) for j in range(d)]

        # sort small → large by criterion
        if self.efb_sort_by == "var":
            crit = [np.var(Xb[:, j]) for j in range(d)]
        else:
            crit = list(self._efb_nnz)
        order = np.argsort(crit)

        bundles: List[List[int]] = []
        bundle_masks: List[np.ndarray] = []
        max_conflicts = int(np.floor(self.efb_conflict_rate * n))

        for j in order:
            aj = active[j]
            placed = False
            for bi in range(len(bundles)):
                conflict = int(np.bitwise_and(bundle_masks[bi], aj).sum())
                if conflict <= max_conflicts:
                    bundles[bi].append(j)
                    bundle_masks[bi] = np.bitwise_or(bundle_masks[bi], aj)
                    placed = True
                    break
            if not placed:
                bundles.append([j])
                bundle_masks.append(aj.copy())

        # offsets: 0 reserved for "inactive"
        offsets = np.zeros(d, dtype=np.int64)
        bundle_bins: List[int] = []
        for group in bundles:
            off = 1
            for j in group:
                Bj = int(self.per_feat_bins_orig[j].item())
                span_eff = max(0, Bj - 1)  # positives after remap
                offsets[j] = off
                off += span_eff
            bundle_bins.append(int(off))  # bins = 0 + sum(span_eff)

        self._bundles = bundles
        self._bundle_offsets_per_feat = offsets
        self._bundle_bins = np.asarray(bundle_bins, dtype=np.int64)
        self._efb_plan_built = True

    def _apply_efb(self, Xb: np.ndarray) -> np.ndarray:
        """
        Map original binned matrix to bundled matrix.
        Strict mode (conflict_rate=0) asserts no collisions in a bundle.
        """
        assert getattr(self, "_efb_plan_built", False), "EFB plan not built"
        n, d = Xb.shape
        out = np.zeros((n, len(self._bundles)), dtype=np.int64)

        for bi, group in enumerate(self._bundles):
            z = np.zeros(n, dtype=np.int64)
            active_sum = np.zeros(n, dtype=np.uint8)
            for j in group:
                remap = self._efb_maps[j]
                v = remap[Xb[:, j]]         # 0 for mode, 1..Bj-1
                is_act = (v > 0).astype(np.uint8)
                if self.efb_conflict_rate == 0.0:
                    active_sum += is_act
                off = int(self._bundle_offsets_per_feat[j])
                z += is_act * (off + v)     # keep 0 for inactive
            if self.efb_conflict_rate == 0.0 and np.any(active_sum > 1):
                raise ValueError("EFB strict mode detected collisions. Increase efb_conflict_rate or disable EFB.")
            out[:, bi] = z
        return out

    # ---------------- Public helpers ----------------

    def _featurise(self, df_new: pd.DataFrame, df_fit: pd.DataFrame, feature_cols: List[str], n_bins: int) -> torch.Tensor:
        """
        Always bin using original feature schema, then apply EFB mapping if enabled.
        """
        Xb_pre = self._bin_dataframe(df_new[self.orig_feature_cols])
        Xb = self._apply_efb(Xb_pre) if self.use_efb else Xb_pre
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

    def step(self, action: Tuple[str, int]):
        kind, _ = action
        if kind in ("th", "thr", "threshold"):
            self._n_splits += 1

    def get_prior(self, beta: float) -> torch.Tensor:
        return torch.tensor(-beta * float(self._n_splits), device=self.device, dtype=torch.float32)
