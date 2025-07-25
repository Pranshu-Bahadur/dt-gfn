from __future__ import annotations
from typing import Any, Dict, Optional, List

import pandas as pd
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from src.trainer import Trainer, Config


class DTGFN(BaseEstimator):
    """
    Scikit-learn compatible wrapper for the DT-GFN Trainer.
    
    This wrapper can be used for both regression and classification tasks.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, **config_kwargs):
        """
        Initializes the DTGFN model.

        Args:
            feature_cols (Optional[List[str]]): A list of feature column names. 
                                                If None, all columns in X during fit will be used.
            **config_kwargs: Hyperparameters for the Trainer's Config object.
        """
        self.feature_cols = feature_cols
        self._cfg_kwargs: Dict[str, Any] = config_kwargs
        self._trainer: Optional[Trainer] = None
        self.df_train_: Optional[pd.DataFrame] = None
        self.task = self._cfg_kwargs.get("task", "classification")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the model. `X` must be a pandas DataFrame; `y` may be a Series/
        array, or you may include the target column in `X` and omit `y`.
        """
        # Get target column name from config, default to 'target' if not provided
        target_col = self._cfg_kwargs.get("target_col", "target")

        # Build training DataFrame with target column present
        if y is not None:
            df_train = X.copy()
            df_train[target_col] = y
        else:
            if target_col not in X.columns:
                raise ValueError(
                    f"If `y` is not provided, X must contain the target column '{target_col}'."
                )
            df_train = X.copy()

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
        self.task = cfg.task

        # Train
        self._trainer = Trainer(cfg).fit(df_train)
        return self

    def predict(self, X: pd.DataFrame, predict_mode: str = "ensemble", n_trees: Optional[int] = None) -> np.ndarray:
        """
        Make predictions using the fitted model.

        For regression, this returns the predicted values.
        For classification, this returns the class labels.
        """
        if self._trainer is None:
            raise RuntimeError("DTGFN has not been fitted yet. Call fit() first.")
        
        if self.task == "regression":
            return self._trainer.predict(df_test=X, df_train=self.df_train_, use_policy=(predict_mode == "policy"), policy_inference_trees=n_trees)
        else:
            probas = self.predict_proba(X, predict_mode, n_trees)
            if self._trainer.cfg.n_classes == 2:
                # Get the index of the highest probability for each sample
                return probas.argmax(axis=1)
            else:
                return probas.argmax(axis=1)


    def predict_proba(self, X: pd.DataFrame, predict_mode: str = "ensemble", n_trees: Optional[int] = None) -> np.ndarray:
        """
        Predict class probabilities.
        """
        if self._trainer is None:
            raise RuntimeError("DTGFN has not been fitted yet. Call fit() first.")
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification tasks.")

        logits = self._trainer.predict(df_test=X, df_train=self.df_train_, use_policy=(predict_mode == "policy"), policy_inference_trees=n_trees)
        
        if self._trainer.cfg.n_classes == 2:
            # logits are 1D for binary classification
            probas = torch.sigmoid(torch.from_numpy(logits)).numpy()
            # Ensure probas is a column vector before stacking
            if probas.ndim == 1:
                probas = probas.reshape(-1, 1)
            # Create a (n_samples, 2) array with P(class 0) and P(class 1)
            return np.hstack([1 - probas, probas])
        else:
            # logits are (n_samples, n_classes) for multi-class
            return torch.softmax(torch.from_numpy(logits), dim=1).numpy()


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

class DTGFNRegressor(DTGFN, RegressorMixin):
    def __init__(self, feature_cols: Optional[List[str]] = None, **config_kwargs):
        config_kwargs["task"] = "regression"
        super().__init__(feature_cols, **config_kwargs)

class DTGFNClassifier(DTGFN, ClassifierMixin):
    def __init__(self, feature_cols: Optional[List[str]] = None, **config_kwargs):
        config_kwargs["task"] = "classification"
        super().__init__(feature_cols, **config_kwargs)