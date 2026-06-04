from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.diagnostics.feature_reward_diagnostics import export_feature_reward_diagnostics, reward_targets_frame


def test_reward_targets_frame_computes_oracle_and_retriever_advantages() -> None:
    evals = [
        {
            "actions": {
                "bm25_keep": {"reward": 0.20},
                "bm25_keyword": {"reward": 0.10},
                "dense_keep": {"reward": 0.70},
                "dense_keyword": {"reward": 0.30},
                "hybrid_keep": {"reward": 0.60},
                "hybrid_keyword": {"reward": 0.40},
            }
        },
        {
            "actions": {
                "bm25_keep": {"reward": 0.80},
                "bm25_keyword": {"reward": 0.40},
                "dense_keep": {"reward": 0.20},
                "dense_keyword": {"reward": 0.10},
                "hybrid_keep": {"reward": 0.50},
                "hybrid_keyword": {"reward": 0.30},
            }
        },
    ]

    targets = reward_targets_frame(evals)

    assert list(targets["oracle_action"]) == ["dense_keep", "bm25_keep"]
    assert list(targets["oracle_margin"]) == [0.10, 0.30]
    assert list(targets["dense_advantage_vs_bm25"]) == [0.50, -0.60]
    assert list(targets["hybrid_advantage_vs_bm25"]) == [0.40, -0.30]


def test_export_feature_reward_diagnostics_writes_group_correlations(tmp_path: Path) -> None:
    features = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, *([0.0] * 8), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, *([0.0] * 8), 0.2, 0.1, 0.2, 0.1, 0.3, 0.0, 0.1, 0.2],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, *([0.0] * 8), 0.8, 0.8, 0.9, 0.7, 0.9, 0.4, 0.8, 0.9],
        ],
        dtype=float,
    )
    evals = [
        _eval(0.8, 0.1, 0.2),
        _eval(0.6, 0.3, 0.4),
        _eval(0.2, 0.9, 0.8),
    ]
    output_csv = tmp_path / "feature_reward.csv"

    exported = export_feature_reward_diagnostics(features, evals, output_csv, dataset="toy", split="train")
    diagnostics = pd.read_csv(exported)
    contrast_dense = diagnostics[
        (diagnostics["group"] == "retrieval_contrast")
        & (diagnostics["target"] == "dense_advantage_vs_bm25")
    ].iloc[0]

    assert "retrieval_contrast" in set(diagnostics["group"])
    assert contrast_dense["width"] == 8
    assert contrast_dense["active_columns"] == 8
    assert contrast_dense["max_abs_corr"] > 0.9


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
