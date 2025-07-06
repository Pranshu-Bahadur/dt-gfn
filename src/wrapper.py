# src/wrapper.py

from __future__ import annotations
from typing import Any, Dict, Optional, List

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.trainer import Trainer, Config


class DTGFNRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper around the updated DT-GFN Trainer.
    
    This wrapper allows for two prediction modes:
    1. 'ensemble': Uses the pre-trained boosted ensemble from the replay buffer (default).
    2. 'policy': Generates a new ensemble of trees directly from the policy network at inference time.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, **config_kwargs):
        """
        Initializes the DTGFNRegressor.

        Args:
            feature_cols (Optional[List[str]]): A list of feature column names. 
                                                If None, all columns in X during fit (except 'target') will be used.
            **config_kwargs: Hyperparameters for the Trainer's Config object.
        """
        # Store kwargs so get_params works
        self.feature_cols = feature_cols
        self._cfg_kwargs: Dict[str, Any] = config_kwargs
        self._trainer: Optional[Trainer] = None
        self.df_train_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # scikit-learn API
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the model. `X` must be a pandas DataFrame; `y` may be a Series/
        array, or you may include the target column in `X` and omit `y`.
        """
        # Build training DataFrame with target column present
        if y is not None:
            df_train = X.copy()
            df_train["target"] = y
            target_col = "target"
        else:
            if "target" not in X.columns:
                raise ValueError(
                    "If `y` is not provided, X must contain a 'target' column."
                )
            df_train = X.copy()
            target_col = "target"

        self.df_train_ = df_train.copy()
        
        # Determine feature columns
        if self.feature_cols:
            feature_cols = self.feature_cols
        else:
            feature_cols = [c for c in X.columns if c != target_col]
        
        # Build Config from kwargs + derived fields
        cfg = Config(
            feature_cols=feature_cols,
            target_col=target_col,
            **self._cfg_kwargs,
        )

        # Train
        self._trainer = Trainer(cfg).fit(df_train)
        return self  # sklearn convention

    def predict(self, X: pd.DataFrame, predict_mode: str = "ensemble", n_trees: int = 100) -> pd.Series:
        """
        Make predictions using the fitted model.

        Args:
            X (pd.DataFrame): The input features for which to make predictions.
            predict_mode (str): The prediction strategy. Can be one of:
                                'ensemble': Use the boosted ensemble created during training (default).
                                'policy': Generate `n_trees` on-the-fly from the policy network.
            n_trees (int): The number of trees to generate if `predict_mode` is 'policy'.

        Returns:
            A pandas Series containing the predictions.
        """
        if self._trainer is None or self.df_train_ is None:
            raise RuntimeError("DTGFNRegressor has not been fitted yet. Call fit() first.")

        if predict_mode == "ensemble":
            predictions = self._trainer.predict(X, self.df_train_)
        elif predict_mode == "policy":
            predictions = self._trainer.predict_from_policy(X, self.df_train_, n_trees=n_trees)
        else:
            raise ValueError(f"Unknown predict_mode: '{predict_mode}'. Must be 'ensemble' or 'policy'.")
        
        return pd.Series(predictions, index=X.index)


    # ------------------------------------------------------------------ #
    # Parameter handling for GridSearchCV / cloning
    # ------------------------------------------------------------------ #
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Gets parameters for this estimator."""
        params = self._cfg_kwargs.copy()
        params['feature_cols'] = self.feature_cols
        return params

    def set_params(self, **params):
        """Sets the parameters of this estimator."""
        if 'feature_cols' in params:
            self.feature_cols = params.pop('feature_cols')
        
        self._cfg_kwargs.update(params)
        
        # Also update the config in the trainer if it exists
        if self._trainer:
            for k, v in params.items():
                setattr(self._trainer.cfg, k, v)

        return self