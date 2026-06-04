from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.diagnostics.budget_curve import export_budget_curve


def test_export_budget_curve_selects_best_method_under_call_budget(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    output_csv = tmp_path / "budget.csv"
    pd.DataFrame(
        [
            {"method": "BM25", "reward": 0.50, "recall_at_5": 0.40, "retrieval_calls": 1.0},
            {"method": "Dense", "reward": 0.60, "recall_at_5": 0.45, "retrieval_calls": 1.0},
            {"method": "Hybrid", "reward": 0.70, "recall_at_5": 0.55, "retrieval_calls": 2.0},
            {"method": "Generated", "reward": 0.80, "recall_at_5": 0.60, "retrieval_calls": 3.0},
        ]
    ).to_csv(summary_csv, index=False)

    result = export_budget_curve(
        dataset="toy",
        summary_csv=summary_csv,
        output_csv=output_csv,
        budgets=[1.0, 2.0],
    )

    rows = pd.read_csv(result)
    assert list(rows["call_budget"]) == [1.0, 2.0]
    assert list(rows["selected_method"]) == ["Dense", "Hybrid"]
    assert list(rows["selected_reward"]) == [0.6, 0.7]
