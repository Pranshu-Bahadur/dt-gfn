import pytest
import numpy as np
import pandas as pd

from dtgfn.binning import BinConfig, Binner


@pytest.fixture
def df_simple():
    # Small synthetic dataset
    return pd.DataFrame({
        'A': [0.0, 1.0, 2.0, 3.0, 4.0],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0]
    })


def test_global_uniform_bins_and_ranges(df_simple):
    cfg = BinConfig(n_bins=3, strategy='global_uniform')
    binner = Binner(cfg)
    # Fit on df_simple
    binner.fit(df_simple)
    # Transform
    binned = binner.transform(df_simple)
    # Should have same shape
    assert isinstance(binned, pd.DataFrame)
    assert binned.shape == df_simple.shape
    # Values should be in 0..n_bins-1
    assert binned.values.min() >= 0
    assert binned.values.max() < cfg.n_bins
    # Edges: each feature has edges length n_bins+1
    for col, edges in binner.edges_.items():
        assert isinstance(edges, np.ndarray)
        assert edges.shape[0] == cfg.n_bins + 1
        # edges are sorted
        assert np.all(np.diff(edges) > 0)

    # inverse_transform recovers approximate values
    inv = binner.inverse_transform(binned)
    # For each col, inverse_transform yields values within original min/max
    for col in df_simple.columns:
        orig_min, orig_max = df_simple[col].min(), df_simple[col].max()
        assert inv[col].min() >= orig_min - 1e-6
        assert inv[col].max() <= orig_max + 1e-6


def test_feature_quantile_reproducibility(df_simple):
    cfg = BinConfig(n_bins=4, strategy='feature_quantile')
    binner1 = Binner(cfg)
    binner2 = Binner(cfg)
    # Fit both on the same data
    binned1 = binner1.fit_transform(df_simple)
    # Save edges, load into binner2 then transform
    edges_copy = {col: edges.copy() for col, edges in binner1.edges_.items()}
    binner2.edges_ = edges_copy
    binner2._mins = binner1._mins
    binner2._rng = binner1._rng
    binned2 = binner2.transform(df_simple)
    # Binned results should match exactly
    pd.testing.assert_frame_equal(binned1, binned2)


def test_invalid_config():
    # n_bins too small
    with pytest.raises(ValueError):
        BinConfig(n_bins=1).validate()
    # unknown strategy
    with pytest.raises(ValueError):
        BinConfig(n_bins=5, strategy='unknown').validate()
