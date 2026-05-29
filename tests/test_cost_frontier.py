from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from selective_rag_rl.cost_frontier import build_cost_frontier, run_cost_frontier_summary


def test_cost_frontier_excludes_oracle_by_default() -> None:
    df = _summary_frame(
        [
            ("Cheap", 1.0, 1.0),
            ("Oracle retrieval action", 9.0, 1.0),
        ]
    )

    result = build_cost_frontier(df, dataset="toy", budgets=[1.0])

    assert result["budget_rows"][0]["selected_method"] == "Cheap"
    assert {row["method"] for row in result["frontier_rows"]} == {"Cheap"}


def test_cost_frontier_selects_best_feasible_method_under_each_budget() -> None:
    df = _summary_frame(
        [
            ("Cheap", 1.0, 1.0),
            ("Middle", 1.3, 1.5),
            ("Expensive", 1.5, 2.0),
        ]
    )

    result = build_cost_frontier(df, dataset="toy", budgets=[1.0, 1.5, 2.0])

    assert [row["selected_method"] for row in result["budget_rows"]] == ["Cheap", "Middle", "Expensive"]


def test_cost_frontier_marks_dominated_methods() -> None:
    df = _summary_frame(
        [
            ("Better cheap", 1.2, 1.0),
            ("Worse expensive", 1.0, 2.0),
        ]
    )

    result = build_cost_frontier(df, dataset="toy", budgets=[2.0])
    frontier = {row["method"]: row for row in result["frontier_rows"]}

    assert frontier["Worse expensive"]["dominated"] is True
    assert frontier["Worse expensive"]["reason"] == "lower_or_equal_reward_and_higher_or_equal_calls"


def test_cost_frontier_missing_columns_get_clear_error() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        build_cost_frontier(pd.DataFrame({"method": ["A"]}), dataset="toy", budgets=[1.0])


def test_run_cost_frontier_summary_writes_budget_frontier_and_json(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    output_csv = tmp_path / "budget.csv"
    _summary_frame(
        [
            ("Cheap", 1.0, 1.0),
            ("Expensive", 1.5, 2.0),
        ]
    ).to_csv(summary_csv, index=False)

    result = run_cost_frontier_summary(
        dataset="toy",
        summary_csvs=[summary_csv],
        output_csv=output_csv,
        budgets=[1.0, 2.0],
    )

    assert output_csv.exists()
    assert Path(result["outputs"]["frontier_csv"]).exists()
    assert Path(result["outputs"]["summary_json"]).exists()
    rows = pd.read_csv(output_csv)
    assert list(rows["selected_method"]) == ["Cheap", "Expensive"]


def _summary_frame(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": method,
                "reward": reward,
                "retrieval_calls": calls,
                "recall_at_5": reward,
                "mrr": reward / 2,
                "rewrite_cost": 0.0,
            }
            for method, reward, calls in rows
        ]
    )
