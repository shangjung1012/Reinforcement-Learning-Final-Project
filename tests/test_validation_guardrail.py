from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.validation_guardrail import (
    aggregate_guardrail_rows,
    evaluate_summary_frame,
    recommend_guardrail,
    run_validation_guardrail,
)


def test_guardrail_recommends_fallback_when_candidate_is_dominated() -> None:
    result = recommend_guardrail(
        candidate_reward=0.9,
        candidate_calls=1.2,
        train_best_reward=1.0,
        train_best_calls=1.0,
        has_validation=True,
    )

    assert result["dominated_by_train_best"] is True
    assert result["recommendation"] == "fallback_to_train_best_fixed"
    assert result["reason"] == "dominated_by_train_best_on_validation"


def test_guardrail_reviews_higher_reward_with_higher_calls() -> None:
    result = recommend_guardrail(
        candidate_reward=1.1,
        candidate_calls=2.0,
        train_best_reward=1.0,
        train_best_calls=1.0,
        has_validation=True,
    )

    assert result["dominated_by_train_best"] is False
    assert result["recommendation"] == "review_cost_reward_tradeoff"
    assert result["reason"] == "higher_reward_with_higher_calls"


def test_guardrail_selects_equal_reward_lower_calls() -> None:
    result = recommend_guardrail(
        candidate_reward=1.0,
        candidate_calls=0.8,
        train_best_reward=1.0,
        train_best_calls=1.0,
        has_validation=True,
    )

    assert result["recommendation"] == "select_candidate"
    assert result["reason"] == "lower_calls_with_equal_reward"


def test_summary_without_validation_is_analysis_only() -> None:
    df = pd.DataFrame(
        [
            {"method": "Train-best retrieval action", "reward": 1.0, "retrieval_calls": 1.0},
            {"method": "Selective retrieval policy", "reward": 1.1, "retrieval_calls": 1.0},
        ]
    )

    row = evaluate_summary_frame(df, dataset="toy")

    assert row["dataset"] == "toy"
    assert row["selected_config"] == "Selective retrieval policy"
    assert row["heldout_best_config"] == "Selective retrieval policy"
    assert row["validation_reward"] is None
    assert row["heldout_reward"] == 1.1
    assert row["recommendation"] == "analysis_only_no_validation"
    assert row["reason"] == "no_validation_split_available"


def test_repeated_aggregation_reports_support_and_fallback_rates() -> None:
    rows = [
        {"dataset": "toy", "recommendation": "select_candidate", "reward_gap": 0.1, "call_gap": 0.0},
        {"dataset": "toy", "recommendation": "fallback_to_train_best_fixed", "reward_gap": -0.1, "call_gap": 0.2},
    ]

    aggregate = aggregate_guardrail_rows(rows)

    assert aggregate["dataset"] == "toy"
    assert aggregate["n_runs"] == 2
    assert aggregate["support_rate"] == 0.5
    assert aggregate["fallback_rate"] == 0.5
    assert aggregate["mean_call_gap"] == 0.1


def test_run_validation_guardrail_writes_csv_and_json(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    output_csv = tmp_path / "guardrail.csv"
    summary_csv.write_text(
        "\n".join(
            [
                "method,reward,retrieval_calls",
                "Train-best retrieval action,1.0,1.0",
                "Selective retrieval policy,0.9,1.2",
            ]
        ),
        encoding="utf-8",
    )

    result = run_validation_guardrail(dataset="toy", summary_csv=summary_csv, output_csv=output_csv)

    assert output_csv.exists()
    assert Path(result["outputs"]["summary_json"]).exists()
    rows = pd.read_csv(output_csv)
    assert rows.loc[0, "recommendation"] == "analysis_only_no_validation"
