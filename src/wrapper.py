# src/wrapper.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union
import numpy as np
import pandas as pd

from .trainer import Trainer, Config  # expects Config is a dataclass

# reserved internal label column (prevents name collisions)
TARGET_COL = "__target__"


def _to_df(X: Union[pd.DataFrame, np.ndarray, Iterable]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        n = X.shape[1] if X.ndim == 2 else 1
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(n)])
    df = pd.DataFrame(X)
    if df.columns is None or any(c is None for c in df.columns):
        df.columns = [f"f{i}" for i in range(df.shape[1])]
    return df


class DTGFNClassifier:
    """
    Thin sklearn-ish wrapper around the DT-GFN Trainer.

    Fixes:
      • Never leaks the target into features (uses reserved TARGET_COL).
      • Freezes feature order at fit-time and reuses it for inference.
      • Encodes labels to 0..C-1 and maps predictions back to original labels.
      • Compatible with Trainer.infer_* and legacy Trainer.predict(...) interfaces.
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
        policy_predictor_mode: str = "mean",
        lr: float = 1e-4,
        lstm_hidden: int = 256,
        mlp_layers: int = 3,
        mlp_width: int = 256,
        rollout_temperature: float = 1.0,
        beta: Optional[float] = None,
    ):
        self._cfg = dict(
            n_bins=n_bins,
            binning_strategy=binning_strategy,
            updates=updates,
            rollouts=rollouts,
            max_depth=max_depth,
            top_k_trees=top_k_trees,
            num_parallel=num_parallel,
            boosting_lr=boosting_lr,
            redundancy_aware=redundancy_aware,
            device=device,
            batch_size=batch_size,
            random_forest=random_forest,
            reward_function=reward_function,
            min_child_size=min_child_size,
            min_gain=min_gain,
            policy_inference_trees=policy_inference_trees,
            policy_predictor_mode=policy_predictor_mode,
            lr=lr,
            lstm_hidden=lstm_hidden,
            mlp_layers=mlp_layers,
            mlp_width=mlp_width,
            rollout_temperature=rollout_temperature,
            beta=beta,
        )

        # set by fit()
        self._trainer: Optional[Trainer] = None
        self._df_train: Optional[pd.DataFrame] = None
        self.feature_cols_: Optional[list[str]] = None
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None  # original label space

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, Iterable],
        y: Union[pd.Series, np.ndarray, Iterable],
    ) -> "DTGFNClassifier":
        df_X = _to_df(X)
        self.n_features_in_ = df_X.shape[1]

        # convert y to np array
        y_arr = np.asarray(y)
        # classification: integers or bools
        is_classif = np.issubdtype(y_arr.dtype, np.integer) or np.array_equal(
            np.unique(y_arr), [0, 1]
        )

        if not is_classif:
            # force int if covertype/poker comes in as object/strings of ints
            try:
                y_arr_int = y_arr.astype(np.int64)
                is_classif = True
                y_arr = y_arr_int
            except Exception:
                pass

        if is_classif:
            classes = np.unique(y_arr)
            inv_map = {cls: i for i, cls in enumerate(classes)}
            y_enc = np.vectorize(inv_map.get, otypes=[np.int64])(y_arr).astype(np.int64)
            self.classes_ = classes
            task = "classification"
        else:
            y_enc = y_arr.astype(np.float32)
            self.classes_ = None
            task = "regression"

        # build training frame with reserved target name
        df_train = df_X.copy()
        if TARGET_COL in df_train.columns:
            df_train = df_train.drop(columns=[TARGET_COL])
        df_train[TARGET_COL] = y_enc

        # freeze feature order
        self.feature_cols_ = [c for c in df_train.columns if c != TARGET_COL]

        # filter cfg to Config dataclass fields and inject required keys
        cfg_kwargs = dict(self._cfg)
        # If Config is a dataclass we can query its fields
        allowed = set(getattr(Config, "__dataclass_fields__", {}).keys())
        filtered = {k: v for k, v in cfg_kwargs.items() if (allowed and k in allowed) or not allowed}

        # required fields (override if present)
        filtered["feature_cols"] = self.feature_cols_
        filtered["target_col"] = TARGET_COL
        filtered["task"] = task

        # choose a device if none provided
        if "device" in allowed and (filtered.get("device") is None):
            import torch

            filtered["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = Config(**filtered)

        trainer = Trainer(cfg)
        trainer.fit(df_train)

        self._trainer = trainer
        self._df_train = df_train
        return self

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _align_infer_df(self, X: Union[pd.DataFrame, np.ndarray, Iterable]) -> pd.DataFrame:
        if self.feature_cols_ is None:
            raise RuntimeError("Model is not fitted.")
        df = _to_df(X).copy()
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
        missing = [c for c in self.feature_cols_ if c not in df.columns]
        if missing:
            raise ValueError(f"Inference data is missing columns: {missing}")
        return df.loc[:, self.feature_cols_]

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray, Iterable],
        *,
        predict_mode: str = "policy",           # "policy" | "ensemble"
        n_trees: Optional[int] = None,
        infer_reward: Optional[str] = "bayesian",
        algorithm: str = "rf",                  # "rf" | "boosting"
        policy_predictor_mode: Optional[str] = None,
    ) -> np.ndarray:
        if self._trainer is None or self._df_train is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if self.classes_ is None:
            raise RuntimeError("predict_proba is only valid for classification tasks.")

        df = self._align_infer_df(X)

        # prefer new API if available
        if hasattr(self._trainer, "infer_proba"):
            P = self._trainer.infer_proba(
                df,
                n_trees=(n_trees or self._cfg.get("policy_inference_trees", 500)),
                mode=predict_mode,
                infer_reward=(infer_reward or self._cfg.get("reward_function", "bayesian")),
                algorithm=algorithm,
                policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "mean")),
            )
        else:
            # legacy API
            P = self._trainer.predict(
                df_test=df,
                df_train=self._df_train,
                use_policy=(predict_mode == "policy"),
                policy_inference_trees=(n_trees or self._cfg.get("policy_inference_trees", 500)),
                policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "mean")),
                infer_reward=(infer_reward or self._cfg.get("reward_function", "bayesian")),
                algorithm=algorithm,
            )
        return np.asarray(P)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, Iterable],
        *,
        predict_mode: str = "policy",
        n_trees: Optional[int] = None,
        infer_reward: Optional[str] = "bayesian",
        algorithm: str = "rf",
        policy_predictor_mode: Optional[str] = None,
    ) -> np.ndarray:
        if self._trainer is None or self._df_train is None:
            raise RuntimeError("Call fit() before predict().")

        if self.classes_ is not None:
            P = self.predict_proba(
                X,
                predict_mode=predict_mode,
                n_trees=n_trees,
                infer_reward=infer_reward,
                algorithm=algorithm,
                policy_predictor_mode=policy_predictor_mode,
            )
            y_enc = np.argmax(P, axis=1).astype(int)
            return self.classes_[y_enc]
        else:
            # regression path
            df = self._align_infer_df(X)
            if hasattr(self._trainer, "infer_regression"):
                y = self._trainer.infer_regression(
                    df,
                    n_trees=(n_trees or self._cfg.get("policy_inference_trees", 500)),
                    mode=predict_mode,
                    infer_reward=(infer_reward or self._cfg.get("reward_function", "bayesian")),
                    algorithm=algorithm,
                    policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "mean")),
                )
            else:
                y = self._trainer.predict(
                    df_test=df,
                    df_train=self._df_train,
                    use_policy=(predict_mode == "policy"),
                    policy_inference_trees=(n_trees or self._cfg.get("policy_inference_trees", 500)),
                    policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "mean")),
                    infer_reward=(infer_reward or self._cfg.get("reward_function", "bayesian")),
                    algorithm=algorithm,
                )
            return np.asarray(y, dtype=np.float32)

    # ------------------------------------------------------------------
    # sklearn-ish
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self._cfg)

    def set_params(self, **params: Any) -> "DTGFNClassifier":
        self._cfg.update(params)
        return self
