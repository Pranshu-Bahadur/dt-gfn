# src/wrapper.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union, List
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

    • Never leaks the target into features (uses reserved TARGET_COL)
    • Freezes feature order at fit-time and reuses it at inference
    • Encodes labels to 0..C-1 and maps predictions back to original labels
    • Forwards all training/inference/binning/regularization/viz knobs to Trainer.Config
    """

    def __init__(
        self,
        *,
        # --- Core dataset/binning ---
        n_bins: int = 255,
        binning_strategy: str = "lgbm_like",   # "lgbm_like" | "quantile" | "global_uniform"
        # LightGBM-like binning controls (env may use these when strategy == "lgbm_like")
        min_data_in_bin: Optional[int] = None,
        subsample_for_bin: Optional[int] = None,
        # Optional per-feature overrides: {"feature": {"type": "binary"/"continuous", "n_bins": K}}
        per_feature_binning: Optional[Dict[str, Dict[str, Any]]] = None,

        # --- Training budget / structure ---
        updates: int = 50,
        rollouts: int = 60,              # target unique rollouts per update (if enforced)
        max_depth: int = 7,
        top_k_trees: int = 10,
        num_parallel: int = 32,
        batch_size: int = 8192,

        # --- Mode & rewards ---
        random_forest: bool = True,
        boosting_lr: float = 0.1,
        reward_function: str = "bayesian",
        infer_reward_function: Optional[str] = None,

        # --- Tree feasibility ---
        min_child_size: int = 20,
        min_gain: float = 0.0,

        # --- Policy inference ---
        policy_inference_trees: int = 500,
        policy_predictor_mode: str = "dirichlet_sample",  # "dirichlet_sample" | "dirichlet" | "mean"
        rollout_temperature: float = 0.0,

        # --- Network / opt ---
        lr: float = 1e-4,
        lstm_hidden: int = 256,
        mlp_layers: int = 3,
        mlp_width: int = 256,
        backward_policy: str = "uniform",  # "uniform" | "network" (usually uniform)
        beta: Optional[float] = None,
        prior_scale: float = 0.5,
        device: Optional[str] = None,

        # --- Redundancy / uniqueness / STOP ---
        redundancy_aware: bool = True,
        redundancy_lambda_intra: float = 1.0,
        redundancy_lambda_inter: float = 0.25,
        redundancy_decay: float = 0.995,
        redundancy_ngram: int = 8,
        enforce_unique_rollouts: bool = True,
        unique_rollouts_max_rounds_factor: int = 50,
        dedup_sequences: bool = True,

        allow_early_stop: bool = False,
        min_decisions_before_stop: int = 1,
        stop_bias: float = 0.0,

        # --- Replay extras ---
        replay_novelty_bonus: float = 0.10,
        replay_metric_mode: str = "off",   # "off" | "acc_corr"
        replay_metric_alpha: float = 1.0,
        replay_metric_power: float = 1.0,
        replay_metric_refresh: int = 1000,

        # --- TB stabilization & live viz ---
        tb_reward_temperature: float = 10.0,
        tb_reward_standardize: bool = True,
        show_best_tree_acc: bool = True,
        viz_every: int = 0,
        viz_dir: str = "runs/trees",
        viz_format: str = "png",

        # --- System / throughput ---
        amp: bool = True,
        eval_on_cpu: bool = False,
        metric_sample_size: int = 20000,
        eval_batch_size: int = 16384,

        # --- Reward scope ---
        training_reward_scope: str = "per_tree",  # "per_tree" | "ensemble"
        ensemble_reward_metric: str = "mse",

        # --- Leaf discouragement & balance knobs ---
        leaf_penalty_strength: float = 4.0,
        leaf_penalty_decay: float = 0.85,
        leaf_penalty_min_depth: int = 2,
        leaf_bias: float = -0.2,
        leaf_cooldown_steps: int = 1,
        threshold_balance_gamma: float = 0.0,
    ):
        # Store everything; we'll filter by Config at fit()
        self._cfg: Dict[str, Any] = dict(
            # data/binning
            n_bins=n_bins,
            binning_strategy=binning_strategy,
            min_data_in_bin=min_data_in_bin,
            subsample_for_bin=subsample_for_bin,
            per_feature_binning=per_feature_binning,

            # training budget
            updates=updates,
            rollouts=rollouts,
            max_depth=max_depth,
            top_k_trees=top_k_trees,
            num_parallel=num_parallel,
            batch_size=batch_size,

            # mode/rewards
            random_forest=random_forest,
            boosting_lr=boosting_lr,
            reward_function=reward_function,
            infer_reward_function=infer_reward_function,

            # feasibility
            min_child_size=min_child_size,
            min_gain=min_gain,

            # policy inference
            policy_inference_trees=policy_inference_trees,
            policy_predictor_mode=policy_predictor_mode,
            rollout_temperature=rollout_temperature,

            # network/opt
            lr=lr,
            lstm_hidden=lstm_hidden,
            mlp_layers=mlp_layers,
            mlp_width=mlp_width,
            backward_policy=backward_policy,
            beta=beta,
            prior_scale=prior_scale,
            device=device,

            # redundancy / uniqueness / stop
            redundancy_aware=redundancy_aware,
            redundancy_lambda_intra=redundancy_lambda_intra,
            redundancy_lambda_inter=redundancy_lambda_inter,
            redundancy_decay=redundancy_decay,
            redundancy_ngram=redundancy_ngram,
            enforce_unique_rollouts=enforce_unique_rollouts,
            unique_rollouts_max_rounds_factor=unique_rollouts_max_rounds_factor,
            dedup_sequences=dedup_sequences,
            allow_early_stop=allow_early_stop,
            min_decisions_before_stop=min_decisions_before_stop,
            stop_bias=stop_bias,

            # replay extras
            replay_novelty_bonus=replay_novelty_bonus,
            replay_metric_mode=replay_metric_mode,
            replay_metric_alpha=replay_metric_alpha,
            replay_metric_power=replay_metric_power,
            replay_metric_refresh=replay_metric_refresh,

            # TB/viz
            tb_reward_temperature=tb_reward_temperature,
            tb_reward_standardize=tb_reward_standardize,
            show_best_tree_acc=show_best_tree_acc,
            viz_every=viz_every,
            viz_dir=viz_dir,
            viz_format=viz_format,

            # system / throughput
            amp=amp,
            eval_on_cpu=eval_on_cpu,
            metric_sample_size=metric_sample_size,
            eval_batch_size=eval_batch_size,

            # reward scope
            training_reward_scope=training_reward_scope,
            ensemble_reward_metric=ensemble_reward_metric,

            # leaf discouragement & balance
            leaf_penalty_strength=leaf_penalty_strength,
            leaf_penalty_decay=leaf_penalty_decay,
            leaf_penalty_min_depth=leaf_penalty_min_depth,
            leaf_bias=leaf_bias,
            leaf_cooldown_steps=leaf_cooldown_steps,
            threshold_balance_gamma=threshold_balance_gamma,
        )

        # set by fit()
        self._trainer: Optional[Trainer] = None
        self._df_train: Optional[pd.DataFrame] = None
        self.feature_cols_: Optional[List[str]] = None
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

        # classification if integer-like or boolean {0,1}
        is_classif = np.issubdtype(y_arr.dtype, np.integer)
        if not is_classif:
            # special-case: ints encoded as strings/objects (e.g., covertype)
            try:
                y_arr = y_arr.astype(np.int64)
                is_classif = True
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

        # filter _cfg to Config dataclass fields; inject required
        cfg_kwargs = dict(self._cfg)
        allowed = set(getattr(Config, "__dataclass_fields__", {}).keys())
        filtered = {k: v for k, v in cfg_kwargs.items() if k in allowed}

        # required fields
        filtered["feature_cols"] = self.feature_cols_
        filtered["target_col"] = TARGET_COL
        filtered["task"] = task

        # choose device if none provided
        if ("device" in allowed) and (filtered.get("device") is None):
            import torch
            filtered["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = Config(**filtered)
        trainer = Trainer(cfg).fit(df_train)

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
    # predict_proba (classification)
    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray, Iterable],
        *,
        predict_mode: str = "policy",   # "policy" | "ensemble"
        n_trees: Optional[int] = None,
        infer_reward: Optional[str] = "bayesian",
        algorithm: str = "rf",          # "rf" | "boosting"
        policy_predictor_mode: Optional[str] = None,
    ) -> np.ndarray:
        if self._trainer is None or self._df_train is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if self.classes_ is None:
            raise RuntimeError("predict_proba is only valid for classification tasks.")

        df = self._align_infer_df(X)

        # prefer newer API if present
        if hasattr(self._trainer, "infer_proba"):
            P = self._trainer.infer_proba(
                df,
                n_trees=(n_trees or self._cfg.get("policy_inference_trees", 500)),
                mode=predict_mode,
                infer_reward=(infer_reward or self._cfg.get("reward_function", "bayesian")),
                algorithm=algorithm,
                policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "dirichlet_sample")),
            )
        else:
            # legacy / mainline API
            P = self._trainer.predict(
                df_test=df,
                df_train=self._df_train,
                use_policy=(predict_mode == "policy"),
                policy_inference_trees=(n_trees or self._cfg.get("policy_inference_trees", 500)),
                policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "dirichlet_sample")),
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
                    policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "dirichlet_sample")),
                )
            else:
                y = self._trainer.predict(
                    df_test=df,
                    df_train=self._df_train,
                    use_policy=(predict_mode == "policy"),
                    policy_inference_trees=(n_trees or self._cfg.get("policy_inference_trees", 500)),
                    policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "dirichlet_sample")),
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
        """
        Update stored config. Any keys that exist in Trainer.Config
        will be forwarded at the next fit(); other keys are retained here.
        """
        self._cfg.update(params)
        return self


class DTGFNRegressor:
    """
    Regressor twin of DTGFNClassifier.

    • Uses the same configuration surface
    • Always treats the task as regression (even if y is integer-typed)
    • Returns float32 predictions
    """

    def __init__(self, **kwargs: Any):
        # Reuse the exact same knob surface as DTGFNClassifier
        self._clf_stub = DTGFNClassifier(**kwargs)
        # Internal mirrors for convenience
        self._trainer: Optional[Trainer] = None
        self._df_train: Optional[pd.DataFrame] = None
        self.feature_cols_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None

    # Proxy cfg dict accessors for consistency
    @property
    def _cfg(self) -> Dict[str, Any]:
        return self._clf_stub._cfg

    # ------------------------------------------------------------------
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, Iterable],
        y: Union[pd.Series, np.ndarray, Iterable],
    ) -> "DTGFNRegressor":
        df_X = _to_df(X)
        self.n_features_in_ = df_X.shape[1]

        # force regression target
        y_arr = np.asarray(y, dtype=np.float32)

        df_train = df_X.copy()
        if TARGET_COL in df_train.columns:
            df_train = df_train.drop(columns=[TARGET_COL])
        df_train[TARGET_COL] = y_arr

        # freeze feature order
        self.feature_cols_ = [c for c in df_train.columns if c != TARGET_COL]

        # filter to Config
        cfg_kwargs = dict(self._cfg)
        allowed = set(getattr(Config, "__dataclass_fields__", {}).keys())
        filtered = {k: v for k, v in cfg_kwargs.items() if k in allowed}

        filtered["feature_cols"] = self.feature_cols_
        filtered["target_col"] = TARGET_COL
        filtered["task"] = "regression"

        # choose device if none provided
        if ("device" in allowed) and (filtered.get("device") is None):
            import torch
            filtered["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = Config(**filtered)
        trainer = Trainer(cfg).fit(df_train)

        self._trainer = trainer
        self._df_train = df_train
        return self

    # ------------------------------------------------------------------
    def _align_infer_df(self, X: Union[pd.DataFrame, np.ndarray, Iterable]) -> pd.DataFrame:
        # reuse classifier helper semantics
        return self._clf_stub._align_infer_df(X)

    # ------------------------------------------------------------------
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, Iterable],
        *,
        predict_mode: str = "policy",
        n_trees: Optional[int] = None,
        infer_reward: Optional[str] = None,
        algorithm: str = "rf",
        policy_predictor_mode: Optional[str] = None,
    ) -> np.ndarray:
        if self._trainer is None or self._df_train is None:
            raise RuntimeError("Call fit() before predict().")

        df = self._align_infer_df(X)

        # Prefer trainer.infer_regression if available
        if hasattr(self._trainer, "infer_regression"):
            y = self._trainer.infer_regression(
                df,
                n_trees=(n_trees or self._cfg.get("policy_inference_trees", 500)),
                mode=predict_mode,
                infer_reward=(infer_reward or self._cfg.get("infer_reward_function")),
                algorithm=algorithm,
                policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "dirichlet_sample")),
            )
        else:
            y = self._trainer.predict(
                df_test=df,
                df_train=self._df_train,
                use_policy=(predict_mode == "policy"),
                policy_inference_trees=(n_trees or self._cfg.get("policy_inference_trees", 500)),
                policy_predictor_mode=(policy_predictor_mode or self._cfg.get("policy_predictor_mode", "dirichlet_sample")),
                infer_reward=(infer_reward or self._cfg.get("infer_reward_function")),
                algorithm=algorithm,
            )
        return np.asarray(y, dtype=np.float32)

    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self._cfg)

    def set_params(self, **params: Any) -> "DTGFNRegressor":
        self._cfg.update(params)
        return self
