"""
Comprehensive Exploratory Data Analysis (EDA) generator for NASA C-MAPSS dataset.

Generates high-resolution figures saved into reports/figures/:
1. sensor_variance_fd001.png - Justification for dropping zero-variance sensors
2. engine_lifecycle_dist.png - Lifespan distribution of turbofan engines
3. degradation_profiles.png - Multi-sensor trajectories as engines approach failure
4. sensor_correlation_matrix.png - Cross-sensor collinearity heatmap
5. op_condition_clusters.png - Operational regime clustering for FD002/FD004
"""

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

from src.data_loader import SENSOR_COLUMNS, load_dataset
from src.preprocessor import FD001_LOW_VARIANCE_SENSORS, add_rul_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eda_generator")

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plot styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def plot_sensor_variances(df: pd.DataFrame, subset: str = "FD001") -> Path:
    """Plot variance for all 21 sensors highlighting zero-variance features."""
    logger.info("Generating sensor variance comparison for %s...", subset)
    variances = df[SENSOR_COLUMNS].var().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#e74c3c" if s in FD001_LOW_VARIANCE_SENSORS else "#2980b9" for s in variances.index]
    bars = ax.bar(variances.index, variances.values, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_xlabel("Sensor Channel")
    ax.set_ylabel("Variance (Log Scale)")
    ax.set_title(f"Sensor Telemetry Variance Distribution ({subset})\n(Red = Identified Near-Zero Variance Features to Drop)")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    out_path = FIGURES_DIR / f"sensor_variance_{subset.lower()}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)
    return out_path


def plot_engine_lifecycle_distribution(df: pd.DataFrame, subset: str = "FD001") -> Path:
    """Plot distribution of total operating cycles before failure."""
    logger.info("Generating engine lifecycle distribution for %s...", subset)
    lifecycles = df.groupby("engine_id")["cycle"].max()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(lifecycles, bins=25, kde=True, color="#16a085", edgecolor="black", ax=ax)
    ax.axvline(lifecycles.mean(), color="#c0392b", linestyle="--", linewidth=2,
               label=f"Mean: {lifecycles.mean():.1f} cycles")
    ax.axvline(lifecycles.median(), color="#f39c12", linestyle=":", linewidth=2,
               label=f"Median: {lifecycles.median():.1f} cycles")

    ax.set_xlabel("Total Operational Cycles Until Failure")
    ax.set_ylabel("Engine Count")
    ax.set_title(f"Engine Lifecycle Distribution — {subset} ({len(lifecycles)} Engines)")
    ax.legend()

    out_path = FIGURES_DIR / f"engine_lifecycle_dist_{subset.lower()}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)
    return out_path


def plot_degradation_profiles(df: pd.DataFrame, subset: str = "FD001") -> Path:
    """Plot degradation trajectories for key sensors across sample engines."""
    logger.info("Generating sensor degradation trajectories for %s...", subset)
    df_rul = add_rul_labels(df, clip_rul=False)

    sample_engines = [1, 2, 3, 4, 5]
    key_sensors = ["s2", "s3", "s4", "s7", "s11", "s12"]
    sensor_names = {
        "s2": "s2: Total Temperature at LPC Outlet (R)",
        "s3": "s3: Total Temperature at HPC Outlet (R)",
        "s4": "s4: Total Temperature at LPT Outlet (R)",
        "s7": "s7: Total Pressure at HPC Outlet (psia)",
        "s11": "s11: Static Pressure at HPC Outlet (psia)",
        "s12": "s12: Ratio of Fuel Flow to Ps30 (pps/psia)",
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for idx, sensor in enumerate(key_sensors):
        ax = axes[idx]
        for engine_id in sample_engines:
            engine_df = df_rul[df_rul["engine_id"] == engine_id]
            ax.plot(engine_df["RUL"], engine_df[sensor], alpha=0.75, linewidth=1.5, label=f"Engine {engine_id}")

        ax.set_title(sensor_names.get(sensor, sensor), fontsize=11, fontweight="bold")
        ax.set_ylabel("Sensor Reading")
        ax.invert_xaxis()  # RUL decreasing from left to right (towards failure at 0)
        ax.grid(True, alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=8)

    axes[4].set_xlabel("Remaining Useful Life (Cycles to Failure ->)")
    axes[5].set_xlabel("Remaining Useful Life (Cycles to Failure ->)")

    fig.suptitle(f"Turbofan Degradation Trajectories Across Sample Engines ({subset})\n(Run-to-Failure Evolution Toward 0 RUL)", y=0.99)
    out_path = FIGURES_DIR / f"degradation_profiles_{subset.lower()}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)
    return out_path


def plot_sensor_correlation_matrix(df: pd.DataFrame, subset: str = "FD001") -> Path:
    """Plot correlation heatmap for non-constant sensors."""
    logger.info("Generating sensor correlation matrix for %s...", subset)
    active_sensors = [s for s in SENSOR_COLUMNS if s not in FD001_LOW_VARIANCE_SENSORS]
    corr = df[active_sensors].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"Inter-Sensor Telemetry Correlation Matrix ({subset})", pad=15)

    out_path = FIGURES_DIR / f"sensor_correlation_{subset.lower()}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)
    return out_path


def plot_operational_clusters_fd002() -> Path:
    """Plot 6 operating condition clusters in 3D operational setting space for FD002."""
    logger.info("Generating operational clusters plot for FD002...")
    df_fd002 = load_dataset("FD002", "train")
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_fd002[op_cols])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        df_fd002["op_setting_1"],
        df_fd002["op_setting_2"],
        df_fd002["op_setting_3"],
        c=clusters,
        cmap="tab10",
        s=4,
        alpha=0.4,
    )

    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        kmeans.cluster_centers_[:, 2],
        c="black",
        s=120,
        marker="X",
        edgecolor="white",
        label="Cluster Centroids",
    )

    ax.set_xlabel("Altitude / Mach (Op Setting 1)")
    ax.set_ylabel("TRA (Op Setting 2)")
    ax.set_zlabel("Sea Level Flag (Op Setting 3)")
    ax.set_title("Operational Regime Clustering (FD002: 6 Distinct Flight Envelopes)", pad=20)
    ax.legend(loc="upper right")

    out_path = FIGURES_DIR / "op_condition_clusters_fd002.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info("Saved %s", out_path)
    return out_path


def run_all_eda():
    """Execute complete EDA pipeline and generate all artifacts."""
    logger.info("=== Starting Comprehensive C-MAPSS EDA ===")
    df_fd001 = load_dataset("FD001", "train")

    plot_sensor_variances(df_fd001, "FD001")
    plot_engine_lifecycle_distribution(df_fd001, "FD001")
    plot_degradation_profiles(df_fd001, "FD001")
    plot_sensor_correlation_matrix(df_fd001, "FD001")
    plot_operational_clusters_fd002()

    logger.info("=== EDA generation complete. All figures saved to %s ===", FIGURES_DIR)


if __name__ == "__main__":
    run_all_eda()
