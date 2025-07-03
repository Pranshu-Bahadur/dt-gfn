import pytest
import pandas as pd
import torch

from src.env import TabularEnv
from src.binning import BinConfig
from src.tokenizer import Tokenizer


@pytest.fixture
def simple_df():
    # Float features and constant label
    return pd.DataFrame({
        'f1': [0.1, 0.4, 0.6, 0.9],
        'f2': [1.0, 2.0, 3.0, 4.0],
        'label': [5.0, 5.0, 5.0, 5.0],
    })


@pytest.fixture
def prebinned_df():
    # Integer “already binned” features and constant label
    return pd.DataFrame({
        'f1': [0, 1, 2, 3],
        'f2': [3, 2, 1, 0],
        'label': [7.0, 7.0, 7.0, 7.0],
    })


def test_prebinned_detection(prebinned_df):
    df = prebinned_df
    cfg = BinConfig(n_bins=10)  # bin_config is ignored for prebinned
    tok = Tokenizer(num_features=2, num_bins=cfg.n_bins)
    env = TabularEnv(
        df,
        feature_cols=['f1', 'f2'],
        target_col='label',
        bin_config=cfg,
        tokenizer=tok,
    )

    # Binner is skipped
    assert env.binner is None
    # num_bins = max(bin index) + 1
    assert env.num_bins == 4

    # X_full matches original and is int8
    expected = torch.tensor(df[['f1', 'f2']].values, dtype=torch.int8)
    assert torch.equal(env.X_full.cpu(), expected)
    assert env.X_full.dtype == torch.int8

    # featurise also skips binning
    df2 = pd.DataFrame({'f1': [2, 0], 'f2': [1, 3], 'label': [7.0, 7.0]})
    feat = env.featurise(df2)
    expected2 = torch.tensor(df2[['f1', 'f2']].values, dtype=torch.int8)
    assert torch.equal(feat.cpu(), expected2)


def test_non_prebinned_behavior(simple_df):
    df = simple_df
    cfg = BinConfig(n_bins=3, strategy='global_uniform')
    tok = Tokenizer(num_features=2, num_bins=cfg.n_bins)
    env = TabularEnv(
        df,
        feature_cols=['f1', 'f2'],
        target_col='label',
        bin_config=cfg,
        tokenizer=tok,
    )

    # Binner should be active
    assert env.binner is not None
    # num_bins matches config
    assert env.num_bins == cfg.n_bins
    # X_full shape and dtype
    assert env.X_full.shape == (4, 2)
    assert env.X_full.dtype == torch.int8

    # featurise returns same as X_full
    feat = env.featurise(df)
    assert torch.equal(feat.cpu(), env.X_full.cpu())


def test_reset_and_step_and_done(simple_df):
    df = simple_df
    cfg = BinConfig(n_bins=3)
    tok = Tokenizer(num_features=2, num_bins=cfg.n_bins)
    env = TabularEnv(
        df,
        feature_cols=['f1', 'f2'],
        target_col='label',
        bin_config=cfg,
        tokenizer=tok,
    )

    # reset
    env.reset(batch_size=2)
    assert env.paths == []
    assert env.open_leaves == 1
    assert env.done is False
    # idxs in [0, 4)
    assert all(0 <= idx < 4 for idx in env.idxs.tolist())

    # step a split
    env.reset(batch_size=2)
    env.step(('feature', 0))
    assert env.open_leaves == 1
    assert env.done is False

    # step leaf twice to close all leaves
    env.step(('threshold', 1))
    assert env.open_leaves == 2
    assert env.done is False

    env.step(('leaf', 0))
    assert env.open_leaves == 1
    assert env.done is True


def test_evaluate_fallback(simple_df):
    df = simple_df
    cfg = BinConfig(n_bins=3)
    tok = Tokenizer(num_features=2, num_bins=cfg.n_bins)
    env = TabularEnv(
        df,
        feature_cols=['f1', 'f2'],
        target_col='label',
        bin_config=cfg,
        tokenizer=tok,
    )

    # fallback with no paths
    env.reset(batch_size=2)
    R, prior, pred, y_t = env.evaluate(current_beta=0.5)
    assert isinstance(R, float)
    assert torch.is_tensor(prior)
    assert prior.item() == 0.0
    assert pred is None
    # y_t length matches batch
    assert y_t.shape[0] == 2
    # constant label → R == 1.0
    assert pytest.approx(R, rel=1e-6) == 1.0

    # fallback with one split action
    env.reset(batch_size=3)
    env.step(('feature', 1))
    R2, prior2, pred2, y2 = env.evaluate(current_beta=0.5)
    assert prior2.item() == -0.5
    assert pred2 is None
    assert y2.shape[0] == 3

