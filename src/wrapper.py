from __future__ import annotations
from dataclasses import replace
from typing import Optional, Union, Iterable
import numpy as np
import pandas as pd
import torch

from src.trainer import Trainer, Config

class DTGFNClassifier:
    """
    Thin sklearn-ish wrapper around the Trainer for classification tasks.
    """

    def __init__(
        self,
        *,
        n_bins: int = 255,
        binning_strategy: str = "quantile",
        updates: int = 50,
        rollouts: int = 60,
        max_depth: int = 7,
        top_k_trees: int = 10,
        num_parallel: int = 32,
        boosting_lr: float = 0.1,
        redundancy_aware: bool = False,
        device: Optional[str] = None,
        batch_size: int = 8192,
        random_forest: bool = True,
        reward_function: str = "bayesian",
        min_child_size: int = 20,
        min_gain: float = 0.0,
        policy_inference_trees: int = 500,
        policy_predictor_mode: str = "mean",   # default to mean at inference for stability
        lr: float = 1e-4,
        lstm_hidden: int = 256,
        mlp_layers: int = 3,
        mlp_width: int = 256,
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._random_forest = bool(random_forest)
        self._cfg_kwargs = dict(
            n_bins=n_bins,
            binning_strategy=binning_strategy,
            updates=updates,
            rollouts=rollouts,
            max_depth=max_depth,
            top_k_trees=top_k_trees,
            num_parallel=num_parallel,
            boosting_lr=boosting_lr,
            redundancy_aware=redundancy_aware,
            device=self._device,
            batch_size=batch_size,
            random_forest=self._random_forest,
            reward_function=reward_function,
            min_child_size=min_child_size,
            min_gain=min_gain,
            policy_inference_trees=policy_inference_trees,
            policy_predictor_mode=policy_predictor_mode,
            lr=lr,
            lstm_hidden=lstm_hidden,
            mlp_layers=mlp_layers,
            mlp_width=mlp_width,
        )

        self._trainer: Optional[Trainer] = None
        self._df_train: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[list] = None
        self.classes_: Optional[np.ndarray] = None

    # -----------------------------
    # Fit
    # -----------------------------
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "DTGFNClassifier":
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name="Cover_Type")
        elif isinstance(y, pd.Series):
            y = y.rename("Cover_Type")

        df_train = X.copy()
        df_train["Cover_Type"] = y.values

        feature_cols = [c for c in df_train.columns if c != "target"]

        cfg = Config(
            feature_cols=feature_cols,
            target_col="Cover_Type",
            task="classification",
            #device=self._device,
            **{k: v for k, v in self._cfg_kwargs.items() if k in Config.__dataclass_fields__}
        )
        # also push non-config extras
        cfg.batch_size = self._cfg_kwargs["batch_size"]
        cfg.policy_predictor_mode = self._cfg_kwargs["policy_predictor_mode"]

        trainer = Trainer(cfg)
        trainer.fit(df_train)

        self._trainer = trainer
        self._df_train = df_train
        self._feature_cols = feature_cols
        self.classes_ = trainer.classes_
        return self

    # -----------------------------
    # Predict / Predict_proba
    # -----------------------------
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        *,
        predict_mode: str = "policy",          # "policy" | "ensemble"
        n_trees: Optional[int] = None,
        infer_reward: Optional[str] = None,
        algorithm: str = "boosting",           # "rf" | "boosting"
        policy_predictor_mode: Optional[str] = None,
    ) -> np.ndarray:
        assert self._trainer is not None and self._df_train is not None, "Call fit() first."
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._feature_cols)

        use_policy = (predict_mode == "policy")
        preds = self._trainer.predict(
            df_test=X,
            df_train=self._df_train,
            use_policy=use_policy,
            policy_inference_trees=n_trees,
            policy_predictor_mode=(policy_predictor_mode or self._cfg_kwargs["policy_predictor_mode"]),
            infer_reward=infer_reward,
            algorithm=algorithm,
        )
        return preds

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        *,
        predict_mode: str = "policy",
        n_trees: Optional[int] = None,
        infer_reward: Optional[str] = None,
        algorithm: str = "rf",                 # default to RF hard preds unless specified
        policy_predictor_mode: Optional[str] = None,
    ) -> np.ndarray:
        P = self.predict_proba(
            X,
            predict_mode=predict_mode,
            n_trees=n_trees,
            infer_reward=infer_reward,
            algorithm=algorithm,
            policy_predictor_mode=policy_predictor_mode,
        )
        if P.ndim == 2:
            # classification probs
            yhat_ids = np.argmax(P, axis=1)
            if self.classes_ is not None:
                return self.classes_[yhat_ids]
            return yhat_ids.astype(int)
        return P

    # sklearn compat
    def get_params(self, deep=True):
        return dict(**self._cfg_kwargs)

    def set_params(self, **params):
        self._cfg_kwargs.update(params)
        return self
