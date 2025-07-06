"""wrapper.py – scikit-learn-compatible DT-GFN regressor."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from src.binner import Binner
from src.tokenizer import Tokenizer
from src.reward import RewardConfig
from src.trainer import DTGFNTrainer, TrainerConfig, Tree

class DTGFNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.config_kwargs = kwargs
        self._trainer: DTGFNTrainer | None = None
        self._binner: Binner | None = None
        self._tokenizer: Tokenizer | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the DT-GFN model."""
        # 1. Binner
        self._binner = Binner(n_bins=self.get_params().get('n_bins', 64))
        X_bin_tensor = self._binner.fit(X).transform(X)
        y_tensor = torch.from_numpy(y.to_numpy(dtype=np.float32))

        # 2. Tokenizer
        self._tokenizer = Tokenizer(
            num_features=X.shape[1],
            num_bins=self._binner.n_bins
        )
        
        # 3. Configs
        reward_cfg = RewardConfig(beta=self.get_params().get('beta', 2.0))
        trainer_cfg = TrainerConfig(**{k: v for k, v in self.get_params().items() if k in TrainerConfig.__annotations__})

        # 4. Trainer
        self._trainer = DTGFNTrainer(
            X_bin=X_bin_tensor,
            y=y_tensor,
            tokenizer=self._tokenizer,
            reward_cfg=reward_cfg,
            t_cfg=trainer_cfg
        )
        self._trainer.train()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions for new data."""
        if not all([self._trainer, self._binner]):
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_bin_tensor = self._binner.transform(X).to(self._trainer.device)
        
        # Sample a new ensemble for each prediction call
        num_trees = self.get_params().get('num_trees', 100)
        leaves_per_tree = self.get_params().get('leaves_per_tree', 8)
        ensemble = self._trainer.sample_ensemble(num_trees, leaves_per_tree)
        
        if not ensemble:
            # Fallback to mean prediction if no valid trees are sampled
            return np.full(X.shape[0], self._trainer.y.mean().item())

        total_preds = torch.zeros(X.shape[0], device=self._trainer.device)
        for tree in ensemble:
            total_preds += self._trainer._predict_tree(tree, X_bin_tensor)
            
        return total_preds.cpu().numpy()

    def get_params(self, deep=True):
        return self.config_kwargs

    def set_params(self, **params):
        self.config_kwargs.update(params)
        # Re-create config objects if trainer exists, to reflect changes
        if self._trainer:
            self._trainer.cfg = TrainerConfig(**{k:v for k,v in self.get_params().items() if k in TrainerConfig.__annotations__})
        return self