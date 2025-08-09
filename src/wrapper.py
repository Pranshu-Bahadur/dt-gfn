from __future__ import annotations
from typing import Any, Dict, Optional, List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from src.trainer import Trainer, Config


class DTGFN(BaseEstimator):
    """
    Scikit-learn compatible wrapper for the DT-GFN Trainer.
    Works for both regression and classification.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, **config_kwargs):
        """
        Args:
            feature_cols: list of feature column names. If None, all columns in X (except target_col) are used.
            **config_kwargs: passed into Config(...)
        """
        self.feature_cols = feature_cols
        self._cfg_kwargs: Dict[str, Any] = config_kwargs
        self._trainer: Optional[Trainer] = None
        self.df_train_: Optional[pd.DataFrame] = None
        self.task = self._cfg_kwargs.get("task", "classification")

    # -------------------------
    # Fit
    # -------------------------
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the model. `X` must be a pandas DataFrame; `y` may be a Series/array,
        or you may include the target column in `X` and omit `y`.
        """
        target_col = self._cfg_kwargs.get("target_col", "target")

        if y is not None:
            df_train = X.copy()
            df_train[target_col] = y
        else:
            if target_col not in X.columns:
                raise ValueError(f"If `y` is not provided, X must contain the target column '{target_col}'.")
            df_train = X.copy()

        self.df_train_ = df_train.copy()

        feature_cols = self.feature_cols or [c for c in X.columns if c != target_col]

        cfg = Config(
            feature_cols=feature_cols,
            target_col=target_col,
            **self._cfg_kwargs,
        )
        self.task = cfg.task

        self._trainer = Trainer(cfg).fit(df_train)
        return self

    # -------------------------
    # Predict (labels for clf, values for reg)
    # -------------------------
    def predict(
        self,
        X: pd.DataFrame,
        predict_mode: str = "ensemble",           # "ensemble" | "policy"
        n_trees: Optional[int] = None,
        **predict_kwargs,                          # passthrough to Trainer.predict (e.g., predictor_mode, reward_type, ensemble_reward, rf_or_boost)
    ) -> np.ndarray:
        """
        Predict on X.

        Args:
            predict_mode: "ensemble" (use learned ensemble) or "policy" (generate trees at inference)
            n_trees: number of policy-generated trees (only used when predict_mode == "policy")
            **predict_kwargs: forwarded to Trainer.predict (advanced inference options)

        Returns:
            Regression: 1D np.ndarray of predictions
            Classification: 1D np.ndarray of class labels
        """
        if self._trainer is None:
            raise RuntimeError("DTGFN has not been fitted yet. Call fit() first.")

        if self.task == "regression":
            preds = self._trainer.predict(
                df_test=X,
                df_train=self.df_train_,
                use_policy=(predict_mode == "policy"),
                policy_inference_trees=n_trees,
                **predict_kwargs,
            )
            return preds  # already numpy
        else:
            probas = self.predict_proba(X, predict_mode=predict_mode, n_trees=n_trees, **predict_kwargs)
            return probas.argmax(axis=1)

    # -------------------------
    # Predict probabilities (classification only)
    # -------------------------
    def predict_proba(
        self,
        X: pd.DataFrame,
        predict_mode: str = "ensemble",
        n_trees: Optional[int] = None,
        **predict_kwargs,                          # passthrough to Trainer.predict
    ) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        """
        if self._trainer is None:
            raise RuntimeError("DTGFN has not been fitted yet. Call fit() first.")
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification tasks.")

        # Trainer already returns probabilities for classification
        probas = self._trainer.predict(
            df_test=X,
            df_train=self.df_train_,
            use_policy=(predict_mode == "policy"),
            policy_inference_trees=n_trees,
            **predict_kwargs,
        )
        return probas  # np.ndarray [n_samples, n_classes]

    # -------------------------
    # sklearn param plumbing
    # -------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = self._cfg_kwargs.copy()
        params["feature_cols"] = self.feature_cols
        return params

    def set_params(self, **params):
        if "feature_cols" in params:
            self.feature_cols = params.pop("feature_cols")

        self._cfg_kwargs.update(params)

        # live-update trainer config if already created
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
