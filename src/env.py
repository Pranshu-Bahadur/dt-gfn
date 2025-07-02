# dtgfn/env.py

from __future__ import annotations
from collections import deque
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from src.binning import Binner, BinConfig


class TabularEnv:
    """
    A tabular decision‐tree environment for GFN rollouts, with optional
    pre-binned input support (skips fitting/transform) and int8 storage.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        bin_config: BinConfig,
        device: str = "cpu",
    ):
        self.device = device
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Split off the feature matrix
        df_feat = df_train[self.feature_cols]

        # Detect prebinned: all integer dtypes
        if df_feat.dtypes.apply(pd.api.types.is_integer_dtype).all():
            # Pre-binned: skip Binner entirely
            arr = df_feat.values
            max_idx = int(arr.max())
            self.num_bins = max_idx + 1

            # store X_full as int8
            X_arr = arr.astype(np.int8)
            self.X_full = torch.tensor(X_arr, dtype=torch.int8, device=self.device)

            # no binner needed
            self.binner = None

        else:
            # Regular flow: fit Binner then transform
            self.binner = Binner(bin_config)
            X_df = self.binner.fit_transform(df_feat)
            self.num_bins = bin_config.n_bins

            # store as int8
            X_arr = X_df.values.astype(np.int8)
            self.X_full = torch.tensor(X_arr, dtype=torch.int8, device=self.device)

        # Store true targets
        self.y_full = torch.tensor(
            df_train[self.target_col].values,
            dtype=torch.float32,
            device=self.device,
        )
        self.y = self.y_full.clone()

        # reset() will initialize paths, idxs, etc.
        self.paths: List[Tuple[str, int]]
        self.open_leaves: int
        self.done: bool

    def featurise(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Bin a new DataFrame using the same edges as training, or
        if prebinned, just cast to int8.
        Returns a (n_rows, n_features) tensor on self.device.
        """
        df_feat = df[self.feature_cols]
        if self.binner is None:
            arr = df_feat.values.astype(np.int8)
            return torch.tensor(arr, dtype=torch.int8, device=self.device)

        X_df = self.binner.transform(df_feat)
        arr = X_df.values.astype(np.int8)
        return torch.tensor(arr, dtype=torch.int8, device=self.device)

    def reset(self, batch_size: int) -> None:
        """
        Start a new rollout: sample `batch_size` examples uniformly.
        """
        n = self.y_full.size(0)
        idxs = torch.randperm(n, device=self.device)[:batch_size]
        self.idxs = idxs
        self.paths = []
        self.open_leaves = 1
        self.done = False

    def step(self, action: Tuple[str, int]) -> None:
        """
        Take a split or leaf action.
        """
        kind, _ = action
        self.paths.append(action)

        if kind == "feature":
            self.open_leaves += 1
        else:
            self.open_leaves -= 1

        # done if no leaves left or path too long
        if self.open_leaves == 0 or len(self.paths) > 8192:
            self.done = True

    def evaluate(
        self, current_beta: float
    ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute (R, prior, pred, y_t) for the current rollout.
        """
        # prior penalty: one unit per split
        num_splits = sum(1 for kind, _ in self.paths if kind == "feature")
        prior = -current_beta * num_splits

        y_t = self.y_full[self.idxs]
        X_batch = self.X_full[self.idxs]

        # if tree unfinished, fallback to base MSE
        if not self.paths or self.open_leaves != 0:
            mse = ((y_t - y_t.mean()) ** 2).mean().item()
            R = 1.0 / (1.0 + mse)
            return R, torch.tensor([prior], device=self.device), None, y_t

        # build tree recursively (you'll fill in threshold logic)
        path_iter = iter(self.paths)

        def build_tree():
            try:
                kind, idx = next(path_iter)
            except StopIteration:
                return None
            if kind == "feature":
                left = build_tree()
                right = build_tree()
                return {"f": idx, "L": left, "R": right}
            return None  # leaf

        tree = build_tree()
        if tree is None:
            mse = ((y_t - y_t.mean()) ** 2).mean().item()
            R = 1.0 / (1.0 + mse)
            return R, torch.tensor([prior], device=self.device), None, y_t

        # traverse to produce pred (stub: predict mean)
        pred = torch.empty_like(y_t)
        stack = [(tree, torch.arange(y_t.size(0), device=self.device))]
        while stack:
            node, idxs = stack.pop()
            if not idxs.numel() or node is None:
                continue
            # placeholder split logic; replace with your threshold mask
            mask = torch.zeros_like(idxs, dtype=torch.bool)
            stack.append((node["R"], idxs[~mask]))
            stack.append((node["L"], idxs[mask]))

        mse = ((pred - y_t) ** 2).mean().item()
        R = 1.0 / (1.0 + mse)
        return R, torch.tensor([prior], device=self.device), pred, y_t
