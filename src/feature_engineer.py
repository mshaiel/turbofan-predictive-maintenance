"""
Feature engineering module for NASA C-MAPSS turbofan degradation telemetry.

Computes temporally-aware time-series degradation features:
- Engine-isolated rolling window statistics (mean, std, min, max) with min_periods=1
- Cumulative absolute change (wear proxy)
- Normalized cycle position
- Operating condition clustering via KMeans (FD002/FD004)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Constants
DEFAULT_WINDOW_SIZE: int = 10
N_OP_CLUSTERS: int = 6
RANDOM_STATE: int = 42
OP_SETTING_COLS: List[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window: int = DEFAULT_WINDOW_SIZE
) -> pd.DataFrame:
    """
    Compute engine-isolated rolling window statistics for sensor telemetry.

    Calculates rolling mean, std, min, and max strictly within each engine's
    lifecycle using groupby('engine_id').transform(). Applies min_periods=1
    and fills first-cycle standard deviation with 0 to prevent NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        Input telemetry DataFrame containing 'engine_id' and sensor columns.
    sensor_cols : List[str]
        List of sensor column names to engineer rolling statistics for.
    window : int, optional
        Window size in cycles, by default DEFAULT_WINDOW_SIZE (10).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling statistic columns.
    """
    data = df.copy()
    grouped = data.groupby("engine_id")

    for col in sensor_cols:
        if col not in data.columns:
            continue
        col_series = grouped[col]
        data[f"{col}_mean_{window}"] = col_series.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        data[f"{col}_std_{window}"] = (
            col_series.transform(
                lambda x: x.rolling(window, min_periods=1).std()
            ).fillna(0.0)
        )
        data[f"{col}_min_{window}"] = col_series.transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        data[f"{col}_max_{window}"] = col_series.transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )

    logger.info(
        "Generated %d rolling features (mean, std, min, max) for window=%d across %d sensors",
        len(sensor_cols) * 4,
        window,
        len(sensor_cols),
    )
    return data


def add_cumulative_wear_features(
    df: pd.DataFrame,
    sensor_cols: List[str]
) -> pd.DataFrame:
    """
    Compute cumulative absolute deviation (wear proxy) for sensors per engine.

    Calculates sum of absolute first-differences along the engine lifecycle,
    proxying accumulated mechanical stress.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame containing 'engine_id' and sensor columns.
    sensor_cols : List[str]
        Sensor column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with f'{col}_cum_change' columns added.
    """
    data = df.copy()
    grouped = data.groupby("engine_id")

    for col in sensor_cols:
        if col not in data.columns:
            continue
        data[f"{col}_cum_change"] = grouped[col].transform(
            lambda x: x.diff().abs().cumsum().fillna(0.0)
        )

    logger.info("Added cumulative wear features for %d sensors", len(sensor_cols))
    return data


def add_normalized_cycle(
    df: pd.DataFrame,
    fleet_max_cycle: Optional[float] = None
) -> pd.DataFrame:
    """
    Compute normalized cycle position.

    If fleet_max_cycle is provided, normalizes by this constant across all engines,
    preventing temporal truncation distribution shifts on unseen test/production data.
    If fleet_max_cycle is None, normalizes relative to maximum observed cycle per engine.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame containing 'engine_id' and 'cycle'.
    fleet_max_cycle : Optional[float], optional
        Maximum cycle across training fleet (e.g. 362.0 for FD001), by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'cycle_norm' column in range [0, 1].
    """
    data = df.copy()
    if fleet_max_cycle is not None and fleet_max_cycle > 0:
        data["cycle_norm"] = data["cycle"] / float(fleet_max_cycle)
    else:
        max_cycles = data.groupby("engine_id")["cycle"].transform("max")
        data["cycle_norm"] = data["cycle"] / max_cycles
    logger.info("Added 'cycle_norm' feature (fleet_max=%s)", fleet_max_cycle)
    return data


def fit_operational_clusters(
    train_df: pd.DataFrame,
    op_cols: Optional[List[str]] = None,
    n_clusters: int = N_OP_CLUSTERS,
    random_state: int = RANDOM_STATE
) -> KMeans:
    """
    Fit a KMeans model on operational setting configurations.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame containing operational setting columns.
    op_cols : Optional[List[str]], optional
        Operational setting column names, by default OP_SETTING_COLS.
    n_clusters : int, optional
        Number of operating condition clusters, by default 6.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    KMeans
        Fitted KMeans clustering model.
    """
    cols = op_cols or OP_SETTING_COLS
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(train_df[cols])  # fit on train only
    logger.info("Fitted KMeans clustering with %d clusters on %s", n_clusters, cols)
    return kmeans


def add_operational_clusters(
    df: pd.DataFrame,
    kmeans: KMeans,
    op_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Assign operating condition cluster IDs using a pre-fitted KMeans model.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry DataFrame.
    kmeans : KMeans
        Pre-fitted KMeans cluster model.
    op_cols : Optional[List[str]], optional
        Operational setting column names, by default OP_SETTING_COLS.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'op_cluster' integer column added.
    """
    cols = op_cols or OP_SETTING_COLS
    data = df.copy()
    data["op_cluster"] = kmeans.predict(data[cols])  # transform only - no fitting
    return data


def build_feature_pipeline(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window: int = DEFAULT_WINDOW_SIZE,
    include_op_clusters: bool = False,
    kmeans_model: Optional[KMeans] = None,
    fleet_max_cycle: Optional[float] = None
) -> Tuple[pd.DataFrame, Optional[KMeans]]:
    """
    Execute end-to-end feature engineering pipeline on telemetry DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or preprocessed telemetry DataFrame.
    sensor_cols : List[str]
        Retained sensor column names.
    window : int, optional
        Rolling window size, by default 10.
    include_op_clusters : bool, optional
        Whether to cluster operational settings (FD002/FD004), by default False.
    kmeans_model : Optional[KMeans], optional
        Pre-fitted KMeans model (for test/inference). If None and include_op_clusters
        is True, a new model is fitted on df.
    fleet_max_cycle : Optional[float], optional
        Maximum cycle across training fleet (e.g. 362 for FD001), by default None.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[KMeans]]
        Tuple of (engineered DataFrame, fitted or used KMeans model).
    """
    logger.info("Starting feature engineering pipeline...")
    data = add_rolling_features(df, sensor_cols, window=window)
    data = add_cumulative_wear_features(data, sensor_cols)
    data = add_normalized_cycle(data, fleet_max_cycle=fleet_max_cycle)

    fitted_kmeans = kmeans_model
    if include_op_clusters:
        if fitted_kmeans is None:
            fitted_kmeans = fit_operational_clusters(data)
        data = add_operational_clusters(data, fitted_kmeans)

    logger.info("Feature engineering complete. Total columns: %d", data.shape[1])
    return data, fitted_kmeans
