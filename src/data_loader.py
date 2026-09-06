"""
Data ingestion module for NASA C-MAPSS turbofan engine degradation dataset.

This module provides reliable utilities for loading raw telemetry files and
RUL ground-truth files into structured pandas DataFrames and Series with
canonical column definitions and rigorous validation.
"""

import logging
from pathlib import Path
from typing import Optional, Set
import pandas as pd

# Module-level logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base directories
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
FALLBACK_DATA_DIR: Path = PROJECT_ROOT

# Canonical C-MAPSS dataset constants
VALID_SUBSETS: Set[str] = {"FD001", "FD002", "FD003", "FD004"}
VALID_SPLITS: Set[str] = {"train", "test"}

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"
]

OPERATIONAL_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]


def _resolve_data_filepath(filename: str, custom_dir: Optional[Path] = None) -> Path:
    """
    Locate a data file in custom_dir, data/raw, or project root.

    Parameters
    ----------
    filename : str
        Target filename to search for (e.g., 'train_FD001.txt').
    custom_dir : Optional[Path], optional
        User-specified directory to prioritize, by default None.

    Returns
    -------
    Path
        Resolved absolute path to the existing file.

    Raises
    ------
    FileNotFoundError
        If the file cannot be located across candidate search directories.
    """
    candidate_dirs = []
    if custom_dir is not None:
        candidate_dirs.append(Path(custom_dir))
    candidate_dirs.extend([RAW_DATA_DIR, FALLBACK_DATA_DIR])

    for directory in candidate_dirs:
        candidate_path = directory / filename
        if candidate_path.is_file():
            return candidate_path

    raise FileNotFoundError(
        f"Data file '{filename}' could not be found in candidate directories: "
        f"{[str(d) for d in candidate_dirs]}"
    )


def load_dataset(
    subset: str = "FD001",
    split: str = "train",
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load a C-MAPSS dataset file into a labeled DataFrame.

    Parameters
    ----------
    subset : str, optional
        One of 'FD001', 'FD002', 'FD003', 'FD004', by default 'FD001'.
    split : str, optional
        One of 'train', 'test', by default 'train'.
    data_dir : Optional[Path], optional
        Custom directory containing the raw text files, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with named columns, trailing empty columns dropped.

    Raises
    ------
    ValueError
        If subset or split arguments are invalid.
    """
    subset_upper = subset.upper()
    split_lower = split.lower()

    if subset_upper not in VALID_SUBSETS:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of {sorted(VALID_SUBSETS)}")
    if split_lower not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of {sorted(VALID_SPLITS)}")

    filename = f"{split_lower}_{subset_upper}.txt"
    filepath = _resolve_data_filepath(filename, custom_dir=data_dir)

    logger.info("Loading C-MAPSS telemetry from %s", filepath)

    # Read raw whitespace-delimited file with no header
    df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="c")

    # Drop any trailing columns resulting from trailing whitespace (usually cols 26, 27)
    if df.shape[1] > len(COLUMNS):
        df = df.iloc[:, : len(COLUMNS)]

    df.columns = COLUMNS

    # Ensure engine_id and cycle are integer types
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    logger.info(
        "Successfully loaded %s (%s): %d rows, %d columns, %d engines",
        subset_upper,
        split_lower,
        df.shape[0],
        df.shape[1],
        df["engine_id"].nunique(),
    )
    return df


def load_rul(
    subset: str = "FD001",
    data_dir: Optional[Path] = None
) -> pd.Series:
    """
    Load ground-truth Remaining Useful Life (RUL) values for test engines.

    Parameters
    ----------
    subset : str, optional
        One of 'FD001', 'FD002', 'FD003', 'FD004', by default 'FD001'.
    data_dir : Optional[Path], optional
        Custom directory containing the RUL text files, by default None.

    Returns
    -------
    pd.Series
        Ground-truth RUL indexed by engine_id (1-indexed matching test engines).

    Raises
    ------
    ValueError
        If subset argument is invalid.
    """
    subset_upper = subset.upper()
    if subset_upper not in VALID_SUBSETS:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of {sorted(VALID_SUBSETS)}")

    filename = f"RUL_{subset_upper}.txt"
    filepath = _resolve_data_filepath(filename, custom_dir=data_dir)

    logger.info("Loading ground-truth RUL from %s", filepath)

    rul_df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="c")
    rul_series = rul_df.iloc[:, 0].astype(int)
    # 1-indexed engine IDs
    rul_series.index = range(1, len(rul_series) + 1)
    rul_series.name = "RUL"
    rul_series.index.name = "engine_id"

    logger.info("Loaded %d ground-truth RUL records for %s", len(rul_series), subset_upper)
    return rul_series
