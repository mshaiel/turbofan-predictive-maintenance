"""
Unit tests for src/preprocessor.py
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import (
    RUL_CLIP,
    add_rul_labels,
    add_test_rul_labels,
    drop_features,
    fit_scaler,
    identify_low_variance_features,
    transform_features,
)


@pytest.fixture
def mock_telemetry_df() -> pd.DataFrame:
    """Fixture providing synthetic engine telemetry data across 2 engines."""
    return pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "s1": [518.67, 518.67, 518.67, 518.67, 518.67],  # zero-variance
        "s2": [642.0, 642.5, 643.0, 641.0, 641.5],       # varying
    })


def test_add_rul_labels_unclipped(mock_telemetry_df):
    """Verify unclipped RUL calculation equals max_cycle - cycle."""
    df_rul = add_rul_labels(mock_telemetry_df, clip_rul=False)
    assert "RUL" in df_rul.columns
    # Engine 1 has max_cycle = 3 -> RUL = [2, 1, 0]
    expected_e1 = [2, 1, 0]
    assert list(df_rul[df_rul["engine_id"] == 1]["RUL"]) == expected_e1
    # Engine 2 has max_cycle = 2 -> RUL = [1, 0]
    expected_e2 = [1, 0]
    assert list(df_rul[df_rul["engine_id"] == 2]["RUL"]) == expected_e2
    assert (df_rul["RUL"] >= 0).all()


def test_add_rul_labels_clipped():
    """Verify piecewise linear RUL clipping at RUL_CLIP threshold."""
    df_long = pd.DataFrame({
        "engine_id": [1] * 200,
        "cycle": list(range(1, 201)),
    })
    df_clipped = add_rul_labels(df_long, clip_rul=True, rul_clip_limit=RUL_CLIP)
    assert df_clipped["RUL"].max() == RUL_CLIP
    # At cycle 1, unclipped RUL is 199, clipped must be 125
    assert df_clipped.loc[0, "RUL"] == RUL_CLIP
    # At cycle 200, RUL must be 0
    assert df_clipped.loc[199, "RUL"] == 0


def test_add_test_rul_labels():
    """Verify test set RUL trajectory calculation using ground-truth offset."""
    test_df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
    })
    rul_series = pd.Series([10, 20], index=[1, 2])
    test_labeled = add_test_rul_labels(test_df, rul_series, clip_rul=False)

    # Engine 1: max observed cycle is 2, ground truth is 10 -> total life 12
    # At cycle 2, RUL should be 10; at cycle 1, RUL should be 11
    e1_rul = test_labeled[test_labeled["engine_id"] == 1]["RUL"].tolist()
    assert e1_rul == [11, 10]


def test_identify_low_variance_features(mock_telemetry_df):
    """Verify zero-variance sensor detection."""
    low_var = identify_low_variance_features(mock_telemetry_df, ["s1", "s2"])
    assert "s1" in low_var
    assert "s2" not in low_var


def test_drop_features(mock_telemetry_df):
    """Verify dropping designated columns."""
    df_dropped = drop_features(mock_telemetry_df, ["s1"])
    assert "s1" not in df_dropped.columns
    assert "s2" in df_dropped.columns


def test_fit_and_transform_scaler(mock_telemetry_df):
    """Verify MinMaxScaler fit on train only and transform functionality."""
    train_df = mock_telemetry_df.copy()
    test_df = pd.DataFrame({
        "engine_id": [3],
        "cycle": [1],
        "s1": [518.67],
        "s2": [642.0],
    })

    scaler = fit_scaler(train_df, ["s2"])
    scaled_train = transform_features(train_df, ["s2"], scaler)
    scaled_test = transform_features(test_df, ["s2"], scaler)

    # In train, min of s2 is 641.0, max is 643.0
    assert scaled_train["s2"].min() == 0.0
    assert scaled_train["s2"].max() == 1.0
    # In test, s2 is 642.0 -> scaled to 0.5
    assert np.isclose(scaled_test.loc[0, "s2"], 0.5)
