# src/wrappers/sklearn.py

from __future__ import annotations
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.trainer import Trainer, Config


class DTGFNRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper around the updated DT-GFN Trainer.
    """

    def __init__(self, **config_kwargs):
        # Store kwargs so get_params works
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
            df_train = X
            target_col = "target"

        self.df_train_ = df_train.copy()
        
        # Build Config from kwargs + derived fields
        cfg = Config(
            #feature_cols=[c for c in df_train.columns if c != target_col],
            target_col=target_col,
            **self._cfg_kwargs,
        )

        # Train
        self._trainer = Trainer(cfg).fit(df_train)
        return self  # sklearn convention

    def predict(self, X: pd.DataFrame):
        if self._trainer is None or self.df_train_ is None:
            raise RuntimeError("DTGFNRegressor has not been fitted yet.")
        return self._trainer.predict(X, self.df_train_)

    # ------------------------------------------------------------------ #
    # Parameter handling for GridSearchCV / cloning
    # ------------------------------------------------------------------ #
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self._cfg_kwargs.copy()

    def set_params(self, **params):
        self._cfg_kwargs.update(params)
        return self