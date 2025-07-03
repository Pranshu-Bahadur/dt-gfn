# src/env.py

from __future__ import annotations
from typing import List, Tuple

import pandas as pd
import torch

from src.binning import Binner, BinConfig
from src.utils import sequence_to_predictor
from src.tokenizer import Tokenizer


class TabularEnv:
    """
    A tabular decision‐tree environment for GFN rollouts, with optional
    pre-binned input support (skips fitting/transform) and int8 storage.
    Now uses sequence_to_predictor() for both reward and inference.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        bin_config: BinConfig,
        tokenizer: Tokenizer,
        device: str = "cpu",
    ):
        self.device = device
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.tokenizer = tokenizer

        # Prepare features
        df_feat = df_train[self.feature_cols]

        # Detect prebinned: all integer dtypes
        if df_feat.dtypes.apply(pd.api.types.is_integer_dtype).all():
            # Pre-binned: skip Binner entirely
            arr = df_feat.values
            max_idx = int(arr.max())
            self.num_bins = max_idx + 1

            # store X_full as int8
            X_arr = arr.astype("int8")
            self.X_full = torch.tensor(X_arr, dtype=torch.int8, device=self.device)

            self.binner = None
        else:
            # Fit a Binner then transform
            self.binner = Binner(bin_config)
            X_df = self.binner.fit_transform(df_feat)
            self.num_bins = bin_config.n_bins

            X_arr = X_df.values.astype("int8")
            self.X_full = torch.tensor(X_arr, dtype=torch.int8, device=self.device)

        # Targets
        self.y_full = torch.tensor(
            df_train[self.target_col].values,
            dtype=torch.float32,
            device=self.device,
        )
        self.y = self.y_full.clone()

        # Will be set in reset()
        self.paths: List[Tuple[str, int]]
        self.open_leaves: int
        self.done: bool

    def featurise(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Bin a new DataFrame using the same edges as training, or
        if prebinned, just cast to int8.
        """
        df_feat = df[self.feature_cols]
        if self.binner is None:
            arr = df_feat.values.astype("int8")
            return torch.tensor(arr, dtype=torch.int8, device=self.device)

        X_df = self.binner.transform(df_feat)
        arr = X_df.values.astype("int8")
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
        Take a split or leaf action (kind, index).
        """
        kind, _ = action
        self.paths.append(action)

        if kind == "feature":
            self.open_leaves += 1
        else:
            self.open_leaves -= 1

        if self.open_leaves == 0 or len(self.paths) > 8192:
            self.done = True

    def evaluate(
        self, current_beta: float
    ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute (R, prior, pred, y_t) for the current rollout using
        sequence_to_predictor so the env matches inference exactly.
        """
        # Prior penalty
        num_splits = sum(1 for kind, _ in self.paths if kind == "feature")
        prior = -current_beta * num_splits

        # Gather batch
        y_t = self.y_full[self.idxs]
        X_batch = self.X_full[self.idxs]

        # If rollout not complete, fallback to base MSE
        if not self.paths or self.open_leaves != 0:
            mse = ((y_t - y_t.mean()) ** 2).mean().item()
            R = 1.0 / (1.0 + mse)
            return R, torch.tensor([prior], device=self.device), None, y_t

        # Reconstruct token-ID trajectory: [BOS, feat, th, feat, th, ..., EOS]
        tok = self.tokenizer
        traj_ids: List[int] = [tok.BOS]
        for kind, idx in self.paths:
            if kind == "feature":
                traj_ids.append(tok.encode_feature(idx))
            elif kind == "threshold":
                traj_ids.append(tok.encode_threshold(idx))
            else:  # leaf
                traj_ids.append(tok.encode_leaf(idx))
        traj_ids.append(tok.EOS)

        # Build predictor and get predictions for this batch
        pred_fn = sequence_to_predictor(traj_ids, X_batch, y_t, tok)
        pred = pred_fn(X_batch)

        # Compute reward
        mse = ((pred - y_t) ** 2).mean().item()
        R = 1.0 / (1.0 + mse)
        return R, torch.tensor([prior], device=self.device), pred, y_t
