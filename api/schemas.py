"""
Data schemas and payload validation for Antigravity REST API.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


def determine_alert_level(predicted_rul: float) -> str:
    """
    Determine industrial maintenance alert level based on Remaining Useful Life.

    Thresholds:
    - RUL > 100: HEALTHY
    - 50 < RUL <= 100: WATCH
    - 20 < RUL <= 50: WARNING
    - RUL <= 20: CRITICAL

    Parameters
    ----------
    predicted_rul : float
        Model-estimated Remaining Useful Life in flight cycles.

    Returns
    -------
    str
        One of 'HEALTHY', 'WATCH', 'WARNING', 'CRITICAL'.
    """
    if predicted_rul > 100.0:
        return "HEALTHY"
    elif predicted_rul > 50.0:
        return "WATCH"
    elif predicted_rul > 20.0:
        return "WARNING"
    else:
        return "CRITICAL"


@dataclass
class CycleReading:
    """Telemetry reading for a single operational cycle."""
    cycle: int
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensors: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionRequest:
    """Inference request payload containing telemetry history for an engine."""
    engine_id: int
    cycles: List[Dict[str, Any]]


@dataclass
class SHAPContributor:
    """Individual feature contribution toward RUL prediction."""
    feature: str
    shap_value: float
    direction: str  # 'accelerates_failure' or 'prolongs_health'


@dataclass
class PredictionResponse:
    """Inference response payload."""
    engine_id: int
    predicted_rul: float
    alert_level: str
    top_shap_contributors: List[Dict[str, Any]]
    confidence_note: Optional[str] = None
