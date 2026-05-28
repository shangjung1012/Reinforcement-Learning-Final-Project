from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.feature_group_diagnostics import feature_group_slices
from selective_rag_rl.feature_reward_diagnostics import TARGET_COLUMNS, reward_targets_frame


def export_feature_reward_predictive_diagnostics(
    train_features: np.ndarray,
    train_evals: list[dict[str, object]],
    test_features: np.ndarray,
    test_evals: list[dict[str, object]],
    output_csv: Path,
    dataset: str,
    alpha: float = 1.0,
) -> Path:
    diagnostics = feature_reward_predictive_diagnostics_frame(
        train_features,
        train_evals,
        test_features,
        test_evals,
        dataset=dataset,
        alpha=alpha,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_csv, index=False)
    return output_csv


def feature_reward_predictive_diagnostics_frame(
    train_features: np.ndarray,
    train_evals: list[dict[str, object]],
    test_features: np.ndarray,
    test_evals: list[dict[str, object]],
    dataset: str,
    alpha: float = 1.0,
) -> pd.DataFrame:
    train_matrix = _as_matrix(train_features)
    test_matrix = _as_matrix(test_features)
    if train_matrix.shape[1] != test_matrix.shape[1]:
        raise ValueError("Train and test features must have the same width")

    train_targets = reward_targets_frame(train_evals)
    test_targets = reward_targets_frame(test_evals)
    if len(train_matrix) != len(train_targets):
        raise ValueError("Train feature rows and reward targets must have the same length")
    if len(test_matrix) != len(test_targets):
        raise ValueError("Test feature rows and reward targets must have the same length")

    rows = []
    for group, (start, end) in feature_group_slices(train_matrix.shape[1]).items():
        x_train = train_matrix[:, start:end]
        x_test = test_matrix[:, start:end]
        active_mask = np.any(np.abs(x_train) > 1e-12, axis=0) if x_train.size else np.asarray([], dtype=bool)
        for target in TARGET_COLUMNS:
            y_train = train_targets[target].to_numpy(dtype=float)
            y_test = test_targets[target].to_numpy(dtype=float)
            model = _fit_ridge(x_train, y_train, alpha=alpha)
            train_pred = _predict_ridge(x_train, model)
            test_pred = _predict_ridge(x_test, model)
            rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "target": target,
                    "start": start,
                    "end": end,
                    "width": end - start,
                    "active_columns": int(np.sum(active_mask)),
                    "alpha": float(alpha),
                    "train_rows": int(len(y_train)),
                    "test_rows": int(len(y_test)),
                    "train_r2": _r2(y_train, train_pred),
                    "test_r2": _r2(y_test, test_pred),
                    "test_corr": _safe_corr(test_pred, y_test),
                    "target_train_std": float(np.std(y_train)) if y_train.size else 0.0,
                    "target_test_std": float(np.std(y_test)) if y_test.size else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _as_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("Predictive diagnostics require a vector or matrix")
    return matrix


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    if x.shape[0] == 0:
        return np.zeros(x.shape[1], dtype=float), 0.0
    x_mean = np.mean(x, axis=0)
    y_mean = float(np.mean(y)) if y.size else 0.0
    x_centered = x - x_mean
    y_centered = y - y_mean
    regularizer = np.eye(x.shape[1], dtype=float) * float(alpha)
    lhs = x_centered.T @ x_centered + regularizer
    rhs = x_centered.T @ y_centered
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs
    intercept = y_mean - float(x_mean @ weights)
    return weights, intercept


def _predict_ridge(x: np.ndarray, model: tuple[np.ndarray, float]) -> np.ndarray:
    weights, intercept = model
    return x @ weights + intercept


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.size == 0:
        return 0.0
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    if total <= 1e-12:
        return 0.0
    residual = float(np.sum((actual - predicted) ** 2))
    return float(1.0 - residual / total)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])
