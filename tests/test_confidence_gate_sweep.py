from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from selective_rag_rl.diagnostics.confidence_gate_sweep import export_confidence_gate_sweep


def test_export_confidence_gate_sweep_recomputes_metrics_from_predicted_margins(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "sweep.csv"
    pd.DataFrame(
        [
            _row("test", "q1", "Train-best retrieval action", "bm25_keep", 0.40, 1.0, margin=None),
            _row("test", "q1", "Selective retrieval policy", "dense_keep", 0.80, 2.0, margin=0.01),
            _row("test", "q2", "Train-best retrieval action", "bm25_keep", 0.60, 1.0, margin=None),
            _row("test", "q2", "Selective retrieval policy", "hybrid_keep", 0.20, 2.0, margin=0.10),
        ]
    ).to_csv(detailed_csv, index=False)

    exported = export_confidence_gate_sweep(
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        dataset="toy",
        margins=[0.0, 0.05],
    )
    sweep = pd.read_csv(exported)
    no_gate = sweep[sweep["confidence_gate_margin"] == 0.0].iloc[0]
    gated = sweep[sweep["confidence_gate_margin"] == 0.05].iloc[0]

    assert no_gate["dataset"] == "toy"
    assert no_gate["split"] == "test"
    assert no_gate["n_queries"] == 2
    assert no_gate["fallback_rate"] == 0.0
    assert no_gate["gated_reward"] == 0.5
    assert no_gate["gated_retrieval_calls"] == 2.0
    assert no_gate["gated_reward_delta_vs_selective"] == 0.0
    assert no_gate["gated_reward_delta_vs_train_best"] == 0.0

    assert gated["fallback_rate"] == 0.5
    assert gated["gated_reward"] == 0.3
    assert gated["gated_retrieval_calls"] == 1.5
    assert gated["gated_reward_delta_vs_selective"] == -0.2
    assert gated["gated_reward_delta_vs_train_best"] == -0.2
    assert gated["gated_call_delta_vs_selective"] == -0.5


def test_export_confidence_gate_sweep_requires_predicted_margins(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "old_detailed.csv"
    output_csv = tmp_path / "sweep.csv"
    pd.DataFrame(
        [
            _row("test", "q1", "Train-best retrieval action", "bm25_keep", 0.40, 1.0, margin=None),
            _row("test", "q1", "Selective retrieval policy", "dense_keep", 0.80, 2.0, margin=None),
        ]
    ).drop(columns=["predicted_action_margin"]).to_csv(detailed_csv, index=False)

    with pytest.raises(ValueError, match="predicted_action_margin"):
        export_confidence_gate_sweep(detailed_csv=detailed_csv, output_csv=output_csv, dataset="toy", margins=[0.05])


def _row(
    split: str,
    qid: str,
    method: str,
    action: str,
    reward: float,
    retrieval_calls: float,
    margin: float | None,
) -> dict[str, object]:
    return {
        "split": split,
        "method": method,
        "action": action,
        "qid": qid,
        "question": f"question {qid}",
        "recall_at_5": reward,
        "mrr": reward,
        "ndcg_at_5": reward,
        "reward": reward,
        "rewrite_cost": 0.0,
        "retrieval_calls": retrieval_calls,
        "rewrite_tokens": 3,
        "queries": f"query {qid}",
        "top_docs": "d1 | d2",
        "gold_docs": "d1",
        "predicted_action_margin": margin,
    }
