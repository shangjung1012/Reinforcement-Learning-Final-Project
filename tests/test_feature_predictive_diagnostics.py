from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.feature_predictive_diagnostics import (
    export_feature_reward_predictive_diagnostics,
    feature_reward_predictive_diagnostics_frame,
)


def test_feature_reward_predictive_diagnostics_scores_held_out_group_signal() -> None:
    train_features = _features_from_signal([0.1, 0.3, 0.6, 0.8, 1.0])
    test_features = _features_from_signal([0.2, 0.5, 0.9])
    train_evals = [_eval(bm25=0.2, dense=0.2 + value, hybrid=0.3) for value in [0.1, 0.3, 0.6, 0.8, 1.0]]
    test_evals = [_eval(bm25=0.2, dense=0.2 + value, hybrid=0.3) for value in [0.2, 0.5, 0.9]]

    diagnostics = feature_reward_predictive_diagnostics_frame(
        train_features,
        train_evals,
        test_features,
        test_evals,
        dataset="toy",
        alpha=1e-6,
    )
    contrast_dense = diagnostics[
        (diagnostics["group"] == "retrieval_contrast")
        & (diagnostics["target"] == "dense_advantage_vs_bm25")
    ].iloc[0]

    assert contrast_dense["width"] == 8
    assert contrast_dense["active_columns"] == 8
    assert contrast_dense["train_rows"] == 5
    assert contrast_dense["test_rows"] == 3
    assert contrast_dense["test_r2"] > 0.8
    assert contrast_dense["test_corr"] > 0.9


def test_export_feature_reward_predictive_diagnostics_writes_csv(tmp_path: Path) -> None:
    train_features = _features_from_signal([0.1, 0.4, 0.7])
    test_features = _features_from_signal([0.2, 0.8])
    train_evals = [_eval(bm25=0.2, dense=0.2 + value, hybrid=0.3) for value in [0.1, 0.4, 0.7]]
    test_evals = [_eval(bm25=0.2, dense=0.2 + value, hybrid=0.3) for value in [0.2, 0.8]]
    output_csv = tmp_path / "predictive.csv"

    exported = export_feature_reward_predictive_diagnostics(
        train_features,
        train_evals,
        test_features,
        test_evals,
        output_csv,
        dataset="toy",
        alpha=1e-6,
    )

    diagnostics = pd.read_csv(exported)
    assert output_csv.exists()
    assert {
        "dataset",
        "group",
        "target",
        "train_r2",
        "test_r2",
        "test_corr",
        "target_train_std",
        "target_test_std",
    }.issubset(diagnostics.columns)


def _features_from_signal(values: list[float]) -> np.ndarray:
    rows = []
    for value in values:
        rows.append(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                *([0.0] * 8),
                value,
                value * 0.5,
                value + 0.1,
                value - 0.1,
                value * value,
                value + 0.2,
                1.0 - value,
                value * 0.25,
            ]
        )
    return np.asarray(rows, dtype=float)


def _eval(bm25: float, dense: float, hybrid: float) -> dict[str, object]:
    return {
        "actions": {
            "bm25_keep": {"reward": bm25},
            "bm25_keyword": {"reward": bm25 - 0.1},
            "dense_keep": {"reward": dense},
            "dense_keyword": {"reward": dense - 0.1},
            "hybrid_keep": {"reward": hybrid},
            "hybrid_keyword": {"reward": hybrid - 0.1},
        }
    }
