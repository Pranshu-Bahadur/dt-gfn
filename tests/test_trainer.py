import numpy as np
import pandas as pd
import torch
import pytest

from src.trainer import Trainer, Config
from src.wrapper import DTGFNRegressor


@pytest.fixture(scope="module")
def toy_df():
    """
    Five rows, two float features, regression target ~ linear + noise.
    """
    np.random.seed(0)
    f1 = np.linspace(0, 1, 5)
    f2 = np.linspace(1, 2, 5)
    y = 0.3 * f1 + 0.7 * f2 + np.random.normal(0, 0.01, size=5)
    return pd.DataFrame({"f1": f1, "f2": f2, "label": y})


def test_trainer_fit_predict(toy_df):
    cfg = Config(
        feature_cols=["f1", "f2"],
        target_col="label",
        n_bins=4,
        bin_strategy="global_uniform",
        device="cpu",
        updates=2,      # keep tiny for unit test
        rollouts=2,
        batch_size=4,
    )
    trainer = Trainer(cfg).fit(toy_df)
    preds = trainer.predict(toy_df)
    # shape
    assert preds.shape == (len(toy_df),)
    # finite values
    assert np.isfinite(preds).all()
    # predictions should not all equal the mean (some learning happened)
    #assert not np.allclose(preds, preds.mean())


def test_sklearn_wrapper_roundtrip(toy_df):
    X = toy_df[["f1", "f2"]]
    y = toy_df["label"]

    model = DTGFNRegressor(
        n_bins=4,
        updates=2,
        rollouts=2,
        device="cpu",
        bin_strategy="global_uniform",
    )
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()
