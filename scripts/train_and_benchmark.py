"""
End-to-End Training, Benchmarking, and Explainability Pipeline for NASA C-MAPSS.

1. Ingests raw FD001 telemetry and creates run-to-failure RUL labels clipped at 125 cycles.
2. Engineers rolling window statistics, cumulative wear proxies, and cycle normalization.
3. Serializes processed data to Parquet.
4. Performs leak-free 5-fold GroupKFold cross-validation across Ridge, Random Forest,
   XGBoost, and LightGBM.
5. Optimizes LightGBM hyperparameters with Optuna.
6. Trains the champion model on the full training set and saves artifacts to models/.
7. Evaluates on the unseen test set (100 engines at final cycle) using RMSE, MAE,
   and NASA Asymmetric Scoring Function.
8. Computes SHAP attributions, saving summary beeswarm and local waterfall plots.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_loader import SENSOR_COLUMNS, load_dataset, load_rul
from src.evaluator import evaluate_predictions, nasa_score
from src.feature_engineer import build_feature_pipeline
from src.preprocessor import (
    FD001_LOW_VARIANCE_SENSORS,
    RUL_CLIP,
    add_rul_labels,
    fit_scaler,
    transform_features,
)
from src.trainer import cross_validate_model, get_model, tune_lightgbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_benchmark")

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for directory in [DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("=" * 70)
    logger.info("PHASE 1 & 3: INGESTION & FEATURE ENGINEERING (FD001)")
    logger.info("=" * 70)

    # 1. Ingest raw training telemetry
    raw_train = load_dataset(subset="FD001", split="train")

    # 2. Add clipped RUL labels (RUL_CLIP = 125)
    train_labeled = add_rul_labels(raw_train, clip_rul=True, rul_clip_limit=RUL_CLIP)

    # 3. Retain high-variance sensors (drop s1, s5, s6, s10, s16, s18, s19)
    retained_sensors = [s for s in SENSOR_COLUMNS if s not in FD001_LOW_VARIANCE_SENSORS]
    logger.info("Retained %d sensor channels: %s", len(retained_sensors), retained_sensors)

    # 4. Feature engineering: rolling stats (w=10), cumulative wear, normalized cycle
    fleet_max_cycle = float(raw_train["cycle"].max())
    logger.info("Fleet maximum operational cycle: %.1f", fleet_max_cycle)

    df_feat, _ = build_feature_pipeline(
        train_labeled,
        sensor_cols=retained_sensors,
        window=10,
        include_op_clusters=False,
        fleet_max_cycle=fleet_max_cycle
    )

    # Define feature set: drop metadata/target columns
    exclude_cols = ["engine_id", "cycle", "RUL", "op_setting_1", "op_setting_2", "op_setting_3"]
    # also exclude zero-variance raw sensors
    exclude_cols.extend(FD001_LOW_VARIANCE_SENSORS)
    feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
    logger.info("Constructed %d total features", len(feature_cols))

    # Save processed training dataset to Parquet
    parquet_path = DATA_PROCESSED_DIR / "train_FD001_processed.parquet"
    df_feat.to_parquet(parquet_path, index=False)
    logger.info("Saved processed dataset to %s", parquet_path)

    # 5. Scaling (Fit MinMaxScaler strictly on training features)
    scaler = fit_scaler(df_feat, feature_cols)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")
    logger.info("Serialized scaler and feature definitions to models/")

    X_train_scaled = transform_features(df_feat, feature_cols, scaler)[feature_cols]
    y_train = df_feat["RUL"]
    groups_train = df_feat["engine_id"]

    logger.info("=" * 70)
    logger.info("PHASE 4: MODEL BENCHMARKING (5-FOLD GROUPKFOLD CV)")
    logger.info("=" * 70)

    models_to_benchmark = [
        ("Ridge Regression", "ridge", {}),
        ("Random Forest", "random_forest", {"n_estimators": 100, "max_depth": 15, "n_jobs": -1}),
        ("XGBoost", "xgboost", {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05, "n_jobs": -1}),
        ("LightGBM", "lightgbm", {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05, "n_jobs": -1}),
    ]

    benchmark_records = []

    for name, mtype, params in models_to_benchmark:
        logger.info("--- Cross-validating %s ---", name)
        t0 = time.time()
        model_instance = get_model(mtype, params)
        cv_metrics, _ = cross_validate_model(
            model_instance,
            X_train_scaled,
            y_train,
            groups=groups_train,
            n_splits=5
        )
        elapsed = round(time.time() - t0, 2)

        benchmark_records.append({
            "Model": name,
            "CV_RMSE": cv_metrics["rmse"],
            "CV_MAE": cv_metrics["mae"],
            "CV_NASA_Score": cv_metrics["nasa_score"],
            "Training_Time_s": elapsed,
        })
        logger.info(
            "%s => RMSE: %.3f | MAE: %.3f | NASA Score: %.1f | Time: %.2fs",
            name, cv_metrics["rmse"], cv_metrics["mae"], cv_metrics["nasa_score"], elapsed
        )

    # 6. Optuna hyperparameter optimization for LightGBM
    logger.info("=" * 70)
    logger.info("HYPERPARAMETER OPTIMIZATION (OPTUNA - 30 TRIALS)")
    logger.info("=" * 70)
    best_lgbm_params = tune_lightgbm(
        X_train_scaled,
        y_train,
        groups=groups_train,
        n_trials=30
    )
    logger.info("Evaluating Tuned LightGBM under 5-Fold GroupKFold CV...")
    t0 = time.time()
    tuned_lgbm = get_model("lightgbm", best_lgbm_params)
    tuned_metrics, _ = cross_validate_model(
        tuned_lgbm,
        X_train_scaled,
        y_train,
        groups=groups_train,
        n_splits=5
    )
    elapsed_tuned = round(time.time() - t0, 2)

    benchmark_records.append({
        "Model": "LightGBM (Optuna Tuned)",
        "CV_RMSE": tuned_metrics["rmse"],
        "CV_MAE": tuned_metrics["mae"],
        "CV_NASA_Score": tuned_metrics["nasa_score"],
        "Training_Time_s": elapsed_tuned,
    })

    # Save benchmark table
    benchmark_df = pd.DataFrame(benchmark_records).sort_values("CV_NASA_Score")
    csv_path = REPORTS_DIR / "model_comparison.csv"
    benchmark_df.to_csv(csv_path, index=False)
    logger.info("\n%s", benchmark_df.to_string(index=False))
    logger.info("Saved benchmark results to %s", csv_path)

    # 7. Train Champion Model on Full Dataset
    champion_params = best_lgbm_params.copy()
    champion_model = get_model("lightgbm", champion_params)
    logger.info("Training champion model on full training set (20,631 cycles)...")
    champion_model.fit(X_train_scaled, y_train)

    best_model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(champion_model, best_model_path)
    logger.info("Champion model persisted to %s", best_model_path)

    # 8. Unseen Test Set Evaluation (100 engines at final cycle)
    logger.info("=" * 70)
    logger.info("UNSEEN TEST SET EVALUATION (FD001 — 100 ENGINES)")
    logger.info("=" * 70)
    raw_test = load_dataset("FD001", "test")
    test_rul_true = load_rul("FD001")

    test_feat, _ = build_feature_pipeline(
        raw_test,
        sensor_cols=retained_sensors,
        window=10,
        include_op_clusters=False,
        fleet_max_cycle=fleet_max_cycle
    )

    # Extract the last observed cycle for each engine
    test_last_cycles = test_feat.groupby("engine_id").last().reset_index()
    X_test_last = test_last_cycles[feature_cols]

    # Scale test features using fitted scaler (NEVER refit on test)
    X_test_scaled = transform_features(X_test_last, feature_cols, scaler)[feature_cols]

    # Predict RUL
    test_preds = champion_model.predict(X_test_scaled)
    # Clip predictions at 0 (RUL cannot be negative)
    test_preds = np.clip(test_preds, 0, None)

    test_metrics = evaluate_predictions(test_rul_true.values, test_preds)
    logger.info(">>> TEST RESULTS (100 ENGINES) <<<")
    logger.info("Test RMSE       : %.3f cycles", test_metrics["rmse"])
    logger.info("Test MAE        : %.3f cycles", test_metrics["mae"])
    logger.info("Test MAPE       : %.2f%%", test_metrics["mape"])
    logger.info("Test NASA Score : %.1f", test_metrics["nasa_score"])

    # 9. Plot Model Benchmarking Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models = benchmark_df["Model"]
    palette = ["#2ecc71" if "Tuned" in m else "#3498db" for m in models]

    axes[0].barh(models, benchmark_df["CV_RMSE"], color=palette, edgecolor="black")
    axes[0].set_title("Cross-Validation RMSE (Cycles)", fontweight="bold")
    axes[0].set_xlabel("RMSE (Lower is Better)")

    axes[1].barh(models, benchmark_df["CV_MAE"], color=palette, edgecolor="black")
    axes[1].set_title("Cross-Validation MAE (Cycles)", fontweight="bold")
    axes[1].set_xlabel("MAE (Lower is Better)")

    axes[2].barh(models, benchmark_df["CV_NASA_Score"], color=palette, edgecolor="black")
    axes[2].set_title("NASA Asymmetric Score", fontweight="bold")
    axes[2].set_xlabel("NASA Penalty (Lower is Better)")

    plt.suptitle("NASA C-MAPSS Model Benchmarking (5-Fold GroupKFold CV)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_benchmark_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Test Actual vs Predicted RUL
    fig, ax = plt.subplots(figsize=(12, 6))
    engine_ids = np.arange(1, len(test_rul_true) + 1)
    ax.plot(engine_ids, test_rul_true.values, label="True RUL", color="#2c3e50", marker="o", markersize=4, linestyle="-")
    ax.plot(engine_ids, test_preds, label="Predicted RUL (LightGBM)", color="#e74c3c", marker="s", markersize=4, linestyle="--")
    ax.fill_between(engine_ids, test_rul_true.values, test_preds, color="#e74c3c", alpha=0.15, label="Prediction Error")

    ax.set_xlabel("Test Engine ID (1 - 100)")
    ax.set_ylabel("Remaining Useful Life (Cycles)")
    ax.set_title(f"C-MAPSS FD001: True vs Predicted RUL across 100 Test Engines\n(RMSE: {test_metrics['rmse']:.2f} | MAE: {test_metrics['mae']:.2f} | NASA Score: {test_metrics['nasa_score']:.1f})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_prediction_rul_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 10. PHASE 5: SHAP Explainability Generation
    logger.info("=" * 70)
    logger.info("PHASE 5: SHAP EXPLAINABILITY GENERATION")
    logger.info("=" * 70)
    explainer = shap.TreeExplainer(champion_model)

    # Sample 500 rows for summary plot
    sample_indices = np.random.RandomState(42).choice(len(X_train_scaled), size=500, replace=False)
    X_shap_sample = X_train_scaled.iloc[sample_indices]
    shap_values_sample = explainer.shap_values(X_shap_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_sample, X_shap_sample, show=False, max_display=15)
    plt.title("SHAP Feature Importance Beeswarm Plot (NASA C-MAPSS FD001)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP beeswarm plot to %s", FIGURES_DIR / "shap_beeswarm.png")

    # Local Waterfall explanation for an engine near critical failure (test engine 34)
    sample_engine_idx = 34
    explanation = explainer(X_test_scaled)
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation[sample_engine_idx], show=False, max_display=10)
    plt.title(f"Local SHAP Explanation: Test Engine {sample_engine_idx + 1} (True RUL={test_rul_true.iloc[sample_engine_idx]})", pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP waterfall plot to %s", FIGURES_DIR / "shap_waterfall.png")

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
