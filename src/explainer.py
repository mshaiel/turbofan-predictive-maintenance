"""
Model explainability module using SHAP (SHapley Additive exPlanations).

Provides:
- TreeExplainer wrappers for gradient boosted trees / random forests
- Local explanation generation (top-N feature contributors with directional impact)
- Summary beeswarm and waterfall visualization exports for reporting
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_tree_explainer(model: Any) -> Any:
    """
    Initialize a SHAP TreeExplainer for tree-based ensemble models.

    Parameters
    ----------
    model : Any
        Trained tree-based model (RandomForest, XGBoost, LightGBM).

    Returns
    -------
    shap.TreeExplainer
        Configured TreeExplainer instance.
    """
    import shap
    explainer = shap.TreeExplainer(model)
    logger.info("Initialized SHAP TreeExplainer for model %s", type(model).__name__)
    return explainer


def get_top_shap_contributors(
    explainer: Any,
    feature_row: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Extract the top-N SHAP contributors for a single telemetry observation.

    Maps negative SHAP values to 'accelerates_failure' (reduces predicted RUL)
    and positive values to 'prolongs_health' (increases predicted RUL).

    Parameters
    ----------
    explainer : Any
        Fitted SHAP Explainer.
    feature_row : Union[pd.DataFrame, np.ndarray]
        Single row of feature values (shape: 1 x num_features).
    feature_names : List[str]
        Names of the features corresponding to the row columns.
    top_n : int, optional
        Number of top contributing features to return, by default 3.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with keys: 'feature', 'shap_value', 'direction'.
    """
    # Ensure 2D input
    if isinstance(feature_row, pd.DataFrame):
        arr = feature_row.values
    else:
        arr = np.asarray(feature_row)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

    raw_shap_values = explainer.shap_values(arr)

    # Handle shap output shapes (some models return list of arrays or 2D array)
    if isinstance(raw_shap_values, list):
        shap_vals = raw_shap_values[0][0]
    elif raw_shap_values.ndim == 2:
        shap_vals = raw_shap_values[0]
    else:
        shap_vals = raw_shap_values

    # Pair features with their shap values
    contributions = []
    for feat_name, val in zip(feature_names, shap_vals):
        val_float = float(val)
        direction = "accelerates_failure" if val_float < 0 else "prolongs_health"
        contributions.append({
            "feature": feat_name,
            "shap_value": round(val_float, 4),
            "direction": direction,
            "abs_impact": abs(val_float),
        })

    # Sort descending by absolute impact
    contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

    top_contributors = [
        {
            "feature": c["feature"],
            "shap_value": c["shap_value"],
            "direction": c["direction"]
        }
        for c in contributions[:top_n]
    ]

    return top_contributors


def save_shap_summary_plot(
    explainer: Any,
    X_sample: pd.DataFrame,
    output_path: Union[str, Path]
) -> Path:
    """
    Generate and save a SHAP beeswarm summary plot.

    Parameters
    ----------
    explainer : Any
        Fitted SHAP Explainer.
    X_sample : pd.DataFrame
        Representative sample of feature vectors.
    output_path : Union[str, Path]
        Target image file path (.png).

    Returns
    -------
    Path
        Path to saved figure.
    """
    import matplotlib.pyplot as plt
    import shap

    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", dest)
    return dest
