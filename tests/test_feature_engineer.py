"""
Unit tests for src/feature_engineer.py
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineer import (
    DEFAULT_WINDOW_SIZE,
    add_cumulative_wear_features,
    add_normalized_cycle,
    add_operational_clusters,
    add_rolling_features,
    fit_operational_clusters,
)


@pytest.fixture
def synthetic_engine_data() -> pd.DataFrame:
    """Fixture with two distinct engines with varying cycles."""
    return pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "op_setting_1": [-0.0001, -0.0002, -0.0001, 0.0020, 0.0021],
        "op_setting_2": [0.0001, 0.0002, 0.0001, 0.0005, 0.0004],
        "op_setting_3": [100.0, 100.0, 100.0, 60.0, 60.0],
        "s2": [10.0, 20.0, 30.0, 100.0, 200.0],
    })


def test_add_rolling_features_isolation(synthetic_engine_data):
    """Verify rolling statistics do not leak across engine boundaries."""
    window = 2
    res = add_rolling_features(synthetic_engine_data, ["s2"], window=window)

    # Engine 1, cycle 1: mean is 10.0
    assert np.isclose(res.loc[0, f"s2_mean_{window}"], 10.0)
    # Engine 1, cycle 2: mean of (10, 20) is 15.0
    assert np.isclose(res.loc[1, f"s2_mean_{window}"], 15.0)

    # Engine 2, cycle 1: s2 is 100.0. MUST NOT include Engine 1's values!
    assert np.isclose(res.loc[3, f"s2_mean_{window}"], 100.0)
    # Engine 2, cycle 2: mean of (100, 200) is 150.0
    assert np.isclose(res.loc[4, f"s2_mean_{window}"], 150.0)

    # Verify min_periods=1 ensures no NaNs in rolling features
    assert not res[[f"s2_{stat}_{window}" for stat in ["mean", "std", "min", "max"]]].isnull().any().any()


def test_add_cumulative_wear_features(synthetic_engine_data):
    """Verify cumulative wear starts at 0 per engine and is monotonically non-decreasing."""
    res = add_cumulative_wear_features(synthetic_engine_data, ["s2"])
    cum_col = "s2_cum_change"
    assert cum_col in res.columns

    # First cycle of each engine must be 0.0
    assert res.loc[0, cum_col] == 0.0
    assert res.loc[3, cum_col] == 0.0

    # Engine 1 diffs: |20 - 10| = 10, |30 - 20| = 10 -> cumulative: 0, 10, 20
    e1_cum = res[res["engine_id"] == 1][cum_col].tolist()
    assert e1_cum == [0.0, 10.0, 20.0]

    # Engine 2 diffs: |200 - 100| = 100 -> cumulative: 0, 100
    e2_cum = res[res["engine_id"] == 2][cum_col].tolist()
    assert e2_cum == [0.0, 100.0]


def test_add_normalized_cycle(synthetic_engine_data):
    """Verify cycle normalization is in [0, 1] with final cycle equaling 1.0."""
    res = add_normalized_cycle(synthetic_engine_data)
    assert "cycle_norm" in res.columns
    assert (res["cycle_norm"] > 0).all()
    assert (res["cycle_norm"] <= 1.0).all()

    # Engine 1 max cycle is 3 -> cycle 3 normalized is 1.0
    assert np.isclose(res.loc[2, "cycle_norm"], 1.0)
    # Engine 2 max cycle is 2 -> cycle 2 normalized is 1.0
    assert np.isclose(res.loc[4, "cycle_norm"], 1.0)


def test_operational_clusters(synthetic_engine_data):
    """Verify KMeans operational clustering fit and predict."""
    kmeans = fit_operational_clusters(synthetic_engine_data, n_clusters=2)
    res = add_operational_clusters(synthetic_engine_data, kmeans)
    assert "op_cluster" in res.columns
    assert res["op_cluster"].nunique() <= 2
