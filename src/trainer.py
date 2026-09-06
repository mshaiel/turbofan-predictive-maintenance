"""
Model training and cross-validation module for RUL forecasting.

Implements:
- Strict GroupKFold cross-validation (grouping on engine_id) to eliminate temporal leakage
- Benchmarking across Ridge, RandomForest, XGBoost, and LightGBM
- Optuna hyperparameter optimization for gradient boosting models
- Model serialization via joblib
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.evaluator import evaluate_predictions, nasa_score

logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE: int = 42
N_SPLITS: int = 5
OPTUNA_TRIALS: int = 50


def get_model(model_name: str, params: Optional[Dict[str, Any]] = None) -> BaseEstimator:
    """
    Instantiate a regression model with default or specified hyperparameters.

    Parameters
    ----------
    model_name : str
        One of 'ridge', 'random_forest', 'xgboost', 'lightgbm'.
    params : Optional[Dict[str, Any]], optional
        Custom hyperparameters, by default None.

    Returns
    -------
    BaseEstimator
        Configured scikit-learn compatible regression estimator.

    Raises
    ------
    ValueError
        If model_name is unrecognized.
    """
    p = params.copy() if params else {}
    name_lower = model_name.lower()

    if name_lower == "ridge":
        p.setdefault("alpha", 1.0)
        p.setdefault("random_state", RANDOM_STATE)
        return Ridge(**p)

    elif name_lower in {"rf", "random_forest"}:
        p.setdefault("n_estimators", 100)
        p.setdefault("max_depth", 15)
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("n_jobs", -1)
        return RandomForestRegressor(**p)

    elif name_lower == "xgboost":
        import xgboost as xgb
        p.setdefault("n_estimators", 150)
        p.setdefault("max_depth", 6)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("subsample", 0.8)
        p.setdefault("colsample_bytree", 0.8)
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("n_jobs", -1)
        return xgb.XGBRegressor(**p)

    elif name_lower == "lightgbm":
        import lightgbm as lgb
        p.setdefault("n_estimators", 150)
        p.setdefault("max_depth", 6)
        p.setdefault("num_leaves", 31)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("subsample", 0.8)
        p.setdefault("colsample_bytree", 0.8)
        p.setdefault("random_state", RANDOM_STATE)
        p.setdefault("n_jobs", -1)
        p.setdefault("verbose", -1)
        return lgb.LGBMRegressor(**p)

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from 'ridge', 'random_forest', 'xgboost', 'lightgbm'."
        )


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = N_SPLITS
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Perform leak-free cross-validation using GroupKFold grouped by engine_id.

    Parameters
    ----------
    model : BaseEstimator
        Instantiated regressor.
    X : pd.DataFrame
        Engineered feature matrix.
    y : pd.Series
        Target Remaining Useful Life (RUL).
    groups : pd.Series
        Engine IDs ensuring complete isolation between train and validation folds.
    n_splits : int, optional
        Number of cross-validation folds, by default 5.

    Returns
    -------
    Tuple[Dict[str, float], np.ndarray]
        Aggregated evaluation metrics (RMSE, MAE, NASA Score) and out-of-fold predictions.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_predictions = np.zeros(len(y))

    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        # Clone and train model on training fold only
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)

        preds = fold_model.predict(X_val_fold)
        oof_predictions[val_idx] = preds

        metrics = evaluate_predictions(y_val_fold, preds)
        fold_metrics.append(metrics)
        logger.info("Fold %d/%d - RMSE: %.3f | MAE: %.3f | NASA Score: %.1f",
                    fold, n_splits, metrics["rmse"], metrics["mae"], metrics["nasa_score"])

    # Overall OOF metrics across all engines
    overall_metrics = evaluate_predictions(y, oof_predictions)
    logger.info("Overall CV Metrics - RMSE: %.3f | MAE: %.3f | NASA Score: %.1f",
                overall_metrics["rmse"], overall_metrics["mae"], overall_metrics["nasa_score"])

    return overall_metrics, oof_predictions


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_trials: int = OPTUNA_TRIALS
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna and GroupKFold CV.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Target RUL.
    groups : pd.Series
        Engine IDs for GroupKFold.
    n_trials : int, optional
        Number of optimization trials, by default 50.

    Returns
    -------
    Dict[str, Any]
        Best parameter dictionary found by Optuna.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = get_model("xgboost", params)
        metrics, _ = cross_validate_model(model, X, y, groups, n_splits=5)
        return metrics["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best XGBoost Optuna RMSE: %.4f with params: %s", study.best_value, study.best_params)
    return study.best_params


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_trials: int = OPTUNA_TRIALS
) -> Dict[str, Any]:
    """
    Optimize LightGBM hyperparameters using Optuna and GroupKFold CV.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Target RUL.
    groups : pd.Series
        Engine IDs for GroupKFold.
    n_trials : int, optional
        Number of optimization trials, by default 50.

    Returns
    -------
    Dict[str, Any]
        Best parameter dictionary found by Optuna.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = get_model("lightgbm", params)
        metrics, _ = cross_validate_model(model, X, y, groups, n_splits=5)
        return metrics["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best LightGBM Optuna RMSE: %.4f with params: %s", study.best_value, study.best_params)
    return study.best_params


def save_model(model: Any, filepath: Union[str, Path]) -> Path:
    """
    Save trained model artifact using joblib.

    Parameters
    ----------
    model : Any
        Trained model instance.
    filepath : Union[str, Path]
        Target destination path (.pkl or .joblib).

    Returns
    -------
    Path
        Absolute path to saved model artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model artifact to %s", path)
    return path


def load_model(filepath: Union[str, Path]) -> Any:
    """
    Load serialized model artifact.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to model artifact.

    Returns
    -------
    Any
        Loaded model instance.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model
