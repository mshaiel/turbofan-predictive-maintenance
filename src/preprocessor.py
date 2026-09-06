"""
Preprocessing module for NASA C-MAPSS turbofan engine degradation telemetry.

This module handles RUL ground-truth label generation, piecewise linear clipping,
low-variance sensor detection/removal, and strict train-only feature scaling.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Constants
RUL_CLIP: int = 125
DEFAULT_VARIANCE_THRESHOLD: float = 1e-4

# Known constant/zero-variance sensors in FD001
FD001_LOW_VARIANCE_SENSORS: List[str] = [
    "s1", "s5", "s6", "s10", "s16", "s18", "s19"
]


def add_rul_labels(
    df: pd.DataFrame,
    clip_rul: bool = True,
    rul_clip_limit: int = RUL_CLIP
) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for run-to-failure training data.

    Calculates RUL as (max_cycle - current_cycle) for each engine, with optional
    piecewise linear clipping at `rul_clip_limit`.

    Parameters
    ----------
    df : pd.DataFrame
        Training telemetry DataFrame containing 'engine_id' and 'cycle'.
    clip_rul : bool, optional
        Whether to apply piecewise linear clipping to RUL, by default True.
    rul_clip_limit : int, optional
        Maximum RUL ceiling for clipping, by default RUL_CLIP (125).

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with newly populated 'RUL' column.

    Raises
    ------
    KeyError
        If 'engine_id' or 'cycle' are missing from DataFrame.
    """
    if "engine_id" not in df.columns or "cycle" not in df.columns:
        raise KeyError("DataFrame must contain 'engine_id' and 'cycle' columns.")

    data = df.copy()
    max_cycles = data.groupby("engine_id")["cycle"].max()
    data["RUL"] = data["engine_id"].map(max_cycles) - data["cycle"]

    if clip_rul:
        data["RUL"] = data["RUL"].clip(upper=rul_clip_limit)
        logger.info("Applied piecewise linear RUL clipping at %d cycles", rul_clip_limit)

    return data


def add_test_rul_labels(
    test_df: pd.DataFrame,
    rul_series: pd.Series,
    clip_rul: bool = False,
    rul_clip_limit: int = RUL_CLIP
) -> pd.DataFrame:
    """
    Compute full RUL trajectory for test data using ground-truth RUL offsets.

    In C-MAPSS test sets, engines are observed up to some cycle prior to failure.
    The true RUL at the final observed cycle is provided in rul_series.
    Total failure cycle = last_observed_cycle + true_last_rul.
    RUL at cycle c = total_failure_cycle - c.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test telemetry DataFrame with 'engine_id' and 'cycle'.
    rul_series : pd.Series
        True RUL values indexed by engine_id.
    clip_rul : bool, optional
        Whether to apply piecewise linear clipping to test RUL, by default False.
    rul_clip_limit : int, optional
        Maximum RUL ceiling for clipping, by default RUL_CLIP (125).

    Returns
    -------
    pd.DataFrame
        Test DataFrame with populated 'RUL' trajectory column.
    """
    data = test_df.copy()
    last_cycles = data.groupby("engine_id")["cycle"].max()

    # Total lifetime for each test engine
    total_life = last_cycles + rul_series
    data["RUL"] = data["engine_id"].map(total_life) - data["cycle"]

    if clip_rul:
        data["RUL"] = data["RUL"].clip(upper=rul_clip_limit)

    return data


def identify_low_variance_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = DEFAULT_VARIANCE_THRESHOLD
) -> List[str]:
    """
    Identify features with near-zero variance across the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame.
    feature_cols : List[str]
        List of candidate feature column names.
    threshold : float, optional
        Variance cutoff below which features are flagged, by default 1e-4.

    Returns
    -------
    List[str]
        List of feature column names with variance <= threshold.
    """
    variances = df[feature_cols].var()
    low_var_cols = variances[variances <= threshold].index.tolist()
    logger.info(
        "Identified %d low-variance features (var <= %e): %s",
        len(low_var_cols),
        threshold,
        low_var_cols,
    )
    return low_var_cols


def drop_features(
    df: pd.DataFrame,
    cols_to_drop: List[str]
) -> pd.DataFrame:
    """
    Drop specified feature columns if present.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols_to_drop : List[str]
        Column names to remove.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns removed.
    """
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing_drops)


def fit_scaler(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> MinMaxScaler:
    """
    Fit a MinMaxScaler strictly on training features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame containing feature columns.
    feature_cols : List[str]
        Columns to include in scaling.
    feature_range : Tuple[float, float], optional
        Desired range of transformed data, by default (0.0, 1.0).

    Returns
    -------
    MinMaxScaler
        Fitted scaler instance.
    """
    # Fit strictly on training data to avoid data leakage
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(train_df[feature_cols])  # fit on train only
    logger.info("Successfully fitted MinMaxScaler on %d training features", len(feature_cols))
    return scaler


def transform_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: MinMaxScaler
) -> pd.DataFrame:
    """
    Apply fitted MinMaxScaler to feature columns without refitting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (train, validation, or test).
    feature_cols : List[str]
        Columns to scale.
    scaler : MinMaxScaler
        Pre-fitted scaler instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled feature columns.
    """
    data = df.copy()
    data[feature_cols] = scaler.transform(data[feature_cols])  # transform only - no fitting
    return data


def save_scaler(scaler: MinMaxScaler, filepath: Union[str, Path]) -> Path:
    """
    Persist fitted scaler to disk using joblib.

    Parameters
    ----------
    scaler : MinMaxScaler
        Fitted scaler to serialize.
    filepath : Union[str, Path]
        Destination file path (.pkl or .joblib).

    Returns
    -------
    Path
        Absolute path to saved scaler.
    """
    dest_path = Path(filepath)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, dest_path)
    logger.info("Saved scaler to %s", dest_path)
    return dest_path


def load_scaler(filepath: Union[str, Path]) -> MinMaxScaler:
    """
    Load persisted scaler from disk.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to serialized scaler file.

    Returns
    -------
    MinMaxScaler
        Loaded scaler instance.
    """
    source_path = Path(filepath)
    if not source_path.exists():
        raise FileNotFoundError(f"Scaler file not found at {source_path}")
    scaler = joblib.load(source_path)
    logger.info("Loaded scaler from %s", source_path)
    return scaler
