from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.constrained_policy import export_constrained_policy_bootstrap, export_constrained_policy_sweep


def test_export_constrained_policy_sweep_trades_reward_for_calls(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "constrained.csv"
    _toy_detailed_frame().to_csv(detailed_csv, index=False)

    exported = export_constrained_policy_sweep(
        dataset="toy",
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        call_penalties=[0.0, 1.0],
    )

    rows = pd.read_csv(exported)
    low_penalty = rows[rows["call_penalty"] == 0.0].iloc[0]
    high_penalty = rows[rows["call_penalty"] == 1.0].iloc[0]

    assert low_penalty["policy_primary_action"] == "hybrid"
    assert low_penalty["policy_retrieval_calls"] == 2.0
    assert high_penalty["policy_primary_action"] == "bm25"
    assert high_penalty["policy_retrieval_calls"] == 1.0
    assert high_penalty["policy_utility"] > low_penalty["policy_utility"] - 1.0


def test_export_constrained_policy_bootstrap_reports_policy_gap(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "bootstrap.csv"
    _toy_detailed_frame().to_csv(detailed_csv, index=False)

    exported = export_constrained_policy_bootstrap(
        dataset="toy",
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        call_penalties=[0.0, 1.0],
        bootstrap_samples=100,
        seed=3,
    )

    rows = pd.read_csv(exported)
    assert list(rows["call_penalty"]) == [0.0, 1.0]
    assert {"utility_delta_mean", "utility_delta_ci_low", "utility_delta_ci_high"} <= set(rows.columns)
    assert rows["n_queries"].tolist() == [6, 6]
    assert rows["utility_delta_ci_low"].le(rows["utility_delta_ci_high"]).all()


def _toy_detailed_frame() -> pd.DataFrame:
    rows = []
    for split in ["train", "test"]:
        for idx in range(6):
            qid = f"{split}-{idx}"
            for action, calls, recall, mrr in [
                ("bm25", 1.0, 0.4, 0.4),
                ("hybrid", 2.0, 0.9, 0.9),
            ]:
                rows.append(
                    {
                        "split": split,
                        "method": f"{action} method",
                        "action": action,
                        "qid": qid,
                        "question": f"question {qid}",
                        "recall_at_5": recall,
                        "mrr": mrr,
                        "ndcg_at_5": recall,
                        "rewrite_cost": 0.0,
                        "retrieval_calls": calls,
                        "reward": recall + 0.5 * mrr,
                        "state_question_length": float(idx + 1),
                        "state_capitalized_spans": 0.0,
                        "state_bm25_top1": float(idx + 1),
                        "state_bm25_gap": 0.0,
                        "state_bm25_entropy": 0.0,
                    }
                )
    return pd.DataFrame(rows)
