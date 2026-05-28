from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.policy_diagnostics import export_policy_diagnostics
from selective_rag_rl.statistical_diagnostics import export_paired_bootstrap_diagnostics


def test_export_policy_diagnostics_writes_regret_and_action_match(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "diagnostics.csv"
    pd.DataFrame(
        [
            _row("q1", "Selective retrieval policy", "dense_keep", 1.0),
            _row("q1", "Train-best retrieval action", "bm25_keep", 0.8),
            _row("q1", "Oracle retrieval action", "dense_keep", 1.2),
            _row("q1", "Vanilla BM25", "bm25_keep", 0.8),
            _row("q1", "Dense original", "dense_keep", 1.2),
            _row("q1", "Hybrid original", "hybrid_keep", 0.7),
            _row("q2", "Selective retrieval policy", "bm25_keep", 0.4),
            _row("q2", "Train-best retrieval action", "bm25_keep", 0.4),
            _row("q2", "Oracle retrieval action", "hybrid_keep", 0.9),
            _row("q2", "Vanilla BM25", "bm25_keep", 0.4),
            _row("q2", "Dense original", "dense_keep", 0.9),
            _row("q2", "Hybrid original", "hybrid_keep", 0.9),
        ]
    ).to_csv(detailed_csv, index=False)

    exported = export_policy_diagnostics(detailed_csv, output_csv, dataset="toy")
    diagnostics = pd.read_csv(exported)

    assert list(diagnostics["qid"]) == ["q1", "q2"]
    assert list(diagnostics["policy_regret"]) == [0.2, 0.5]
    assert list(diagnostics["policy_reward_delta_vs_train_best"]) == [0.2, 0.0]
    assert list(diagnostics["oracle_margin_vs_train_best"]) == [0.4, 0.5]
    assert list(diagnostics["oracle_margin_vs_second_best"]) == [0.4, 0.0]
    assert list(diagnostics["oracle_tie_count"]) == [1, 2]
    assert list(diagnostics["beats_train_best"]) == [True, False]
    assert list(diagnostics["matches_oracle_action"]) == [True, False]


def test_export_paired_bootstrap_diagnostics_writes_confidence_intervals(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "bootstrap.csv"
    pd.DataFrame(
        [
            _row("q1", "Selective retrieval policy", "dense_keep", 1.0),
            _row("q1", "Train-best retrieval action", "bm25_keep", 0.8),
            _row("q2", "Selective retrieval policy", "bm25_keep", 0.4),
            _row("q2", "Train-best retrieval action", "bm25_keep", 0.2),
            _row("q3", "Selective retrieval policy", "bm25_keep", 0.5),
            _row("q3", "Train-best retrieval action", "bm25_keep", 0.6),
        ]
    ).to_csv(detailed_csv, index=False)

    exported = export_paired_bootstrap_diagnostics(
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        dataset="toy",
        baseline_method="Train-best retrieval action",
        n_bootstrap=200,
        seed=7,
    )
    diagnostics = pd.read_csv(exported)
    reward = diagnostics[diagnostics["metric"] == "reward"].iloc[0]

    assert set(diagnostics["metric"]) == {"reward", "recall_at_5", "mrr", "ndcg_at_5"}
    assert reward["comparison"] == "Selective retrieval policy - Train-best retrieval action"
    assert reward["n_queries"] == 3
    assert round(float(reward["mean_delta"]), 6) == round((0.2 + 0.2 - 0.1) / 3, 6)
    assert float(reward["ci_lower"]) <= float(reward["mean_delta"]) <= float(reward["ci_upper"])
    assert 0.0 <= float(reward["prob_delta_gt_0"]) <= 1.0


def _row(qid: str, method: str, action: str, reward: float) -> dict[str, object]:
    return {
        "split": "test",
        "qid": qid,
        "question": f"question {qid}",
        "method": method,
        "action": action,
        "reward": reward,
        "recall_at_5": reward,
        "mrr": reward,
        "ndcg_at_5": reward,
        "retrieval_calls": 1,
    }
