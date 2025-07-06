"""wrapper.py – scikit-learn-compatible DT-GFN regressor
   (dynamic-sampling version: a new ensemble is drawn on every predict)
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from .binner     import Binner
from .tokenizer  import Tokenizer
from .reward     import RewardConfig
from .trainer    import DTGFNTrainer, TrainerConfig, Tree


class DTGFNRegressor(BaseEstimator, RegressorMixin):
    # ---------- constructor hyper-params -------------------------------------------------
    def __init__(
        self,
        n_bins: int = 64,
        hidden_dim: int = 128,
        depth_limit: int = 6,
        epochs: int = 5,
        replay_size: int = 4096,
        batch_size: int = 1024,
        lr: float = 1e-3,
        epsilon: float = 0.2,
        # ► sampling parameters (used every predict / score call)
        num_trees: int = 400,
        leaves_per_tree: int = 8,
        device: str | None = None,
        beta: float | None = None,
    ):
        # training
        self.n_bins = n_bins
        self.hidden_dim = hidden_dim
        self.depth_limit = depth_limit
        self.epochs = epochs
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        # inference-time sampling
        self.num_trees = num_trees
        self.leaves_per_tree = leaves_per_tree
        # misc
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta  # tree size penalty

        # objects populated by fit()
        self._binner:    Binner      | None = None
        self._tokenizer: Tokenizer   | None = None
        self._trainer:   DTGFNTrainer| None = None

    # ---------- sklearn API --------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        # ----- numpy → tensors -----
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        # ----- binning -----
        self._binner = Binner(n_bins=self.n_bins)
        X_bin = self._binner.fit_transform(X)
        X_bin_t = torch.as_tensor(X_bin, dtype=torch.long)
        y_t     = torch.as_tensor(y,     dtype=torch.float32)
        # ----- tokenizer -----
        self._tokenizer = Tokenizer(
            n_features = X_bin.shape[1],
            n_bins     = int(X_bin.max()) + 1,
        )
        # ----- configs -----
        r_cfg = RewardConfig(beta=self.beta or float(np.log(4)))
        t_cfg = TrainerConfig(
            hidden_dim      = self.hidden_dim,
            depth_limit     = self.depth_limit,
            epochs          = self.epochs,
            batch_size      = self.batch_size,
            replay_size     = self.replay_size,
            lr              = self.lr,
            epsilon         = self.epsilon,
            leaves_per_tree = self.leaves_per_tree,
            device          = self.device,
        )
        # ----- train -----
        self._trainer = DTGFNTrainer(
            X_bin_t, y_t, self._tokenizer, r_cfg, t_cfg
        )
        self._trainer.train()
        return self

    # ---------- internal helpers ---------------------------------------------------------
    def _sample_ensemble(self) -> list[Tree]:
        assert self._trainer is not None, "Model not fitted yet"
        return self._trainer.sample_ensemble(
            num_trees        = self.num_trees,
            leaves_per_tree  = self.leaves_per_tree
        )

    def _predict_on_bin(self, X_bin_t: torch.LongTensor) -> torch.Tensor:
        """Average prediction over a freshly-sampled ensemble."""
        ensemble = self._sample_ensemble()
        preds = torch.zeros(
            X_bin_t.shape[0], dtype=torch.float32, device=self.device
        )
        for tree in ensemble:
            preds += self._trainer._predict_tree(tree, X_bin_t)
        return preds / len(ensemble)

    # ---------- public predict / score ---------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._binner is None:
            raise RuntimeError("DTGFNRegressor must be fitted first")
        X_bin = self._binner.transform(X)
        X_bin_t = torch.as_tensor(X_bin, dtype=torch.long, device=self.device)
        preds_t = self._predict_on_bin(X_bin_t).cpu()
        return preds_t.numpy()

    def score_corr(self, X: np.ndarray, y: np.ndarray) -> float:
        """Numerai-style Pearson (z-scored) correlation."""
        y_pred = self.predict(X)
        y_pred = (y_pred - y_pred.mean()) / (y_pred.std() + 1e-8)
        y      = (y      - y.mean()     ) / (y     .std() + 1e-8)
        return float(np.dot(y_pred, y) / len(y))

