"""Baseline and graph-enhanced ETA models with business-facing metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, median_absolute_error

from .graph_features import BASE_FEATURES, GRAPH_FEATURES


@dataclass
class ETAModel:
    name: str
    features: list[str]
    estimator: HistGradientBoostingRegressor

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.maximum(1.0, np.expm1(self.estimator.predict(frame[self.features])))


def _new_estimator(seed: int) -> HistGradientBoostingRegressor:
    # Histogram boosting is fast on this dataset, captures nonlinear interactions,
    # and serializes to a small artifact suitable for an operational dashboard.
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )


def fit_eta_model(
    training: pd.DataFrame,
    graph_enhanced: bool,
    seed: int = 42,
) -> ETAModel:
    """Fit on log minutes to reduce domination by a small number of long-haul trips."""
    features = GRAPH_FEATURES if graph_enhanced else BASE_FEATURES
    estimator = _new_estimator(seed)
    estimator.fit(training[features], np.log1p(training["actual_minutes"]))
    return ETAModel(
        name="graph_enhanced" if graph_enhanced else "baseline",
        features=list(features),
        estimator=estimator,
    )


def regression_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    error = predicted_array - actual_array
    relative_error = np.abs(error) / np.maximum(actual_array, 1.0)
    return {
        "mae_minutes": float(mean_absolute_error(actual_array, predicted_array)),
        "median_absolute_error_minutes": float(median_absolute_error(actual_array, predicted_array)),
        "within_15pct": float(np.mean(relative_error <= 0.15) * 100),
        "within_30pct": float(np.mean(relative_error <= 0.30) * 100),
        "mean_bias_minutes": float(error.mean()),
        "n_test_trips": int(len(actual_array)),
    }


def evaluate_eta_model(model: ETAModel, test: pd.DataFrame) -> tuple[dict[str, float | int], np.ndarray]:
    predicted = model.predict(test)
    return regression_metrics(test["actual_minutes"], predicted), predicted


def benchmark_models(
    training: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = 42,
) -> tuple[dict[str, ETAModel], dict[str, dict[str, float | int]], pd.DataFrame]:
    """Fit both models on an identical time window and score one untouched window."""
    models = {
        "baseline": fit_eta_model(training, graph_enhanced=False, seed=seed),
        "graph_enhanced": fit_eta_model(training, graph_enhanced=True, seed=seed),
    }
    metrics: dict[str, dict[str, float | int]] = {}
    scored = test[[
        "trip_uuid", "trip_creation_time", "route_type", "start_hub", "end_hub",
        "actual_minutes", "osrm_minutes", "osrm_distance_km", "structural_risk",
        "edge_seen_share",
    ]].copy()
    for name, model in models.items():
        metrics[name], predictions = evaluate_eta_model(model, test)
        scored[f"{name}_prediction"] = predictions
        scored[f"{name}_absolute_error"] = np.abs(predictions - scored["actual_minutes"])
    baseline_mae = float(metrics["baseline"]["mae_minutes"])
    graph_mae = float(metrics["graph_enhanced"]["mae_minutes"])
    baseline_accuracy = float(metrics["baseline"]["within_15pct"])
    graph_accuracy = float(metrics["graph_enhanced"]["within_15pct"])
    metrics["improvement"] = {
        "mae_reduction_minutes": baseline_mae - graph_mae,
        "mae_reduction_pct": (baseline_mae - graph_mae) / baseline_mae * 100,
        "within_15pct_lift_points": graph_accuracy - baseline_accuracy,
        "passes_both_required_metrics": bool(
            graph_mae < baseline_mae and graph_accuracy > baseline_accuracy
        ),
    }
    scored["graph_delay_risk"] = np.clip(
        (scored["graph_enhanced_prediction"] / scored["osrm_minutes"].clip(lower=1) - 1) / 1.5,
        0,
        1,
    )
    return models, metrics, scored


def model_feature_importance(
    model: ETAModel,
    test: pd.DataFrame,
    max_rows: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance in the model's log-target space."""
    sample = test.sample(min(max_rows, len(test)), random_state=seed)
    importance = permutation_importance(
        model.estimator,
        sample[model.features],
        np.log1p(sample["actual_minutes"]),
        scoring="neg_mean_absolute_error",
        n_repeats=4,
        random_state=seed,
        n_jobs=1,
    )
    return pd.DataFrame({
        "feature": model.features,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
