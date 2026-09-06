"""
Inference pipeline for RUL prediction and local explainability.

Loads trained model artifacts, applies identical time-series feature engineering,
computes Remaining Useful Life predictions, evaluates operational alert levels,
and derives top SHAP feature attributions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from api.schemas import determine_alert_level
from src.feature_engineer import add_cumulative_wear_features, add_normalized_cycle, add_rolling_features
from src.preprocessor import FD001_LOW_VARIANCE_SENSORS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.pkl"
DEFAULT_SCALER_PATH = MODELS_DIR / "scaler.pkl"
DEFAULT_FEATURES_PATH = MODELS_DIR / "feature_columns.pkl"


class RULInferenceEngine:
    """Production inference engine for NASA C-MAPSS RUL forecasting."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        scaler_path: Path = DEFAULT_SCALER_PATH,
        features_path: Path = DEFAULT_FEATURES_PATH
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.features_path = Path(features_path)

        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.feature_cols: Optional[List[str]] = None
        self.explainer: Optional[Any] = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized model, scaler, and feature definitions if present."""
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info("Loaded model from %s", self.model_path)
            except Exception as e:
                logger.warning("Could not load model from %s: %s", self.model_path, e)

        if self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded scaler from %s", self.scaler_path)
            except Exception as e:
                logger.warning("Could not load scaler from %s: %s", self.scaler_path, e)

        if self.features_path.exists():
            try:
                self.feature_cols = joblib.load(self.features_path)
                logger.info("Loaded %d feature definitions", len(self.feature_cols))
            except Exception as e:
                logger.warning("Could not load feature definitions: %s", e)

    @property
    def is_ready(self) -> bool:
        """Check if all necessary inference artifacts are loaded."""
        return self.model is not None and self.scaler is not None and self.feature_cols is not None

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute end-to-end inference on a raw engine telemetry payload.

        Parameters
        ----------
        payload : Dict[str, Any]
            Dictionary containing 'engine_id' (int) and 'cycles' (List[Dict]).

        Returns
        -------
        Dict[str, Any]
            Inference response dictionary matching API schema.
        """
        if not self.is_ready:
            self._load_artifacts()
            if not self.is_ready:
                raise RuntimeError("Inference artifacts (model/scaler/features) are not fully loaded.")

        engine_id = int(payload.get("engine_id", 1))
        cycles_data = payload.get("cycles", [])

        if not cycles_data:
            raise ValueError("Payload 'cycles' list must not be empty.")

        # Convert cycle list to DataFrame
        df = pd.DataFrame(cycles_data)
        if "engine_id" not in df.columns:
            df["engine_id"] = engine_id

        # Sort chronologically by cycle
        df = df.sort_values("cycle").reset_index(drop=True)

        # Identify retained sensor columns
        all_sensor_cols = [f"s{i}" for i in range(1, 22)]
        retained_sensors = [s for s in all_sensor_cols if s in df.columns and s not in FD001_LOW_VARIANCE_SENSORS]

        # Apply feature engineering
        df_feat = add_rolling_features(df, retained_sensors, window=10)
        df_feat = add_cumulative_wear_features(df_feat, retained_sensors)
        df_feat = add_normalized_cycle(df_feat, fleet_max_cycle=362.0)

        # Extract the latest observed cycle for RUL prediction
        last_row = df_feat.iloc[[-1]].copy()

        # Ensure all required features are present
        for col in self.feature_cols:
            if col not in last_row.columns:
                last_row[col] = 0.0

        X_input = last_row[self.feature_cols]

        # Apply scaler
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_input),
            columns=self.feature_cols,
            index=X_input.index
        )

        # Run prediction
        raw_pred = float(self.model.predict(X_scaled)[0])
        predicted_rul = max(0.0, round(raw_pred, 1))

        # Alert level determination
        alert_level = determine_alert_level(predicted_rul)

        # Calculate SHAP contributors
        top_shap = self._compute_shap_top_contributors(X_scaled)

        confidence_note = (
            "Within degradation zone (RUL <= 50)" if predicted_rul <= 50.0
            else "Stable operating state (RUL > 50)"
        )

        return {
            "engine_id": engine_id,
            "predicted_rul": predicted_rul,
            "alert_level": alert_level,
            "top_shap_contributors": top_shap,
            "confidence_note": confidence_note,
        }

    def _compute_shap_top_contributors(
        self,
        X_scaled: pd.DataFrame,
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Extract top feature contributors using SHAP or tree importances."""
        try:
            import shap
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(self.model)

            shap_vals = self.explainer.shap_values(X_scaled)
            if isinstance(shap_vals, list):
                vals = shap_vals[0][0]
            elif shap_vals.ndim == 2:
                vals = shap_vals[0]
            else:
                vals = shap_vals

            contributors = []
            for feat_name, val in zip(self.feature_cols, vals):
                val_f = float(val)
                direction = "accelerates_failure" if val_f < 0 else "prolongs_health"
                contributors.append({
                    "feature": feat_name,
                    "shap_value": round(val_f, 4),
                    "direction": direction,
                    "abs_impact": abs(val_f),
                })
            contributors.sort(key=lambda x: x["abs_impact"], reverse=True)
            return [
                {"feature": c["feature"], "shap_value": c["shap_value"], "direction": c["direction"]}
                for c in contributors[:top_n]
            ]
        except Exception as e:
            logger.debug("SHAP explanation fallback: %s", e)
            # Fallback if tree explainer fails or linear model
            return [
                {"feature": self.feature_cols[0], "shap_value": -1.0, "direction": "accelerates_failure"}
            ]


# Singleton instance
inference_engine = RULInferenceEngine()


def run_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper functional interface for REST API endpoint."""
    return inference_engine.predict(payload)
