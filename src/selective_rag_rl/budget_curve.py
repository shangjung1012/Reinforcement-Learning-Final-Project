from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_budget_curve(
    dataset: str,
    summary_csv: Path,
    output_csv: Path,
    budgets: list[float],
    exclude_methods: list[str] | None = None,
) -> Path:
    summary = pd.read_csv(summary_csv)
    required = {"method", "reward", "recall_at_5", "retrieval_calls"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary_csv is missing required column(s): {', '.join(sorted(missing))}")
    excluded = set(exclude_methods or ["Oracle retrieval action"])
    if excluded:
        summary = summary[~summary["method"].isin(excluded)].copy()
    rows = []
    for budget in budgets:
        feasible = summary[summary["retrieval_calls"] <= budget].copy()
        if feasible.empty:
            rows.append(
                {
                    "dataset": dataset,
                    "call_budget": float(budget),
                    "selected_method": "",
                    "selected_reward": 0.0,
                    "selected_recall_at_5": 0.0,
                    "selected_retrieval_calls": 0.0,
                    "n_feasible_methods": 0,
                }
            )
            continue
        feasible = feasible.sort_values(["reward", "recall_at_5", "method"], ascending=[False, False, True])
        best = feasible.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "call_budget": float(budget),
                "selected_method": str(best["method"]),
                "selected_reward": float(best["reward"]),
                "selected_recall_at_5": float(best["recall_at_5"]),
                "selected_retrieval_calls": float(best["retrieval_calls"]),
                "n_feasible_methods": int(len(feasible)),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv
