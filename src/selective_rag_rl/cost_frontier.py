from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"method", "reward", "recall_at_5", "mrr", "retrieval_calls", "rewrite_cost"}
DEFAULT_EXCLUDE_PATTERNS = ("oracle",)


def build_cost_frontier(
    df: pd.DataFrame,
    dataset: str,
    budgets: list[float],
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
) -> dict[str, list[dict[str, object]]]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    methods = _filter_methods(df, exclude_patterns).copy()
    methods = methods.sort_values(["reward", "retrieval_calls"], ascending=[False, True]).reset_index(drop=True)
    frontier_rows = [_frontier_row(methods, idx, dataset) for idx in range(len(methods))]
    budget_rows = [_budget_row(methods, budget, dataset, frontier_rows) for budget in budgets]
    return {"budget_rows": budget_rows, "frontier_rows": frontier_rows}


def run_cost_frontier_summary(
    dataset: str,
    summary_csvs: list[Path],
    output_csv: Path,
    budgets: list[float],
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
) -> dict[str, object]:
    frames = []
    for path in summary_csvs:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one summary CSV is required")
    result = build_cost_frontier(pd.concat(frames, ignore_index=True), dataset, budgets, exclude_patterns)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frontier_csv = output_csv.with_name(f"{output_csv.stem}_frontier.csv")
    summary_json = output_csv.with_suffix(".json")
    pd.DataFrame(result["budget_rows"]).to_csv(output_csv, index=False)
    pd.DataFrame(result["frontier_rows"]).to_csv(frontier_csv, index=False)
    summary = {
        "dataset": dataset,
        "budgets": budgets,
        "budget_rows": result["budget_rows"],
        "frontier_rows": result["frontier_rows"],
        "outputs": {
            "budget_csv": str(output_csv),
            "frontier_csv": str(frontier_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _filter_methods(df: pd.DataFrame, exclude_patterns: tuple[str, ...]) -> pd.DataFrame:
    if not exclude_patterns:
        return df
    mask = pd.Series([False] * len(df), index=df.index)
    for pattern in exclude_patterns:
        mask |= df["method"].str.contains(pattern, case=False, na=False)
    return df[~mask]


def _frontier_row(methods: pd.DataFrame, idx: int, dataset: str) -> dict[str, object]:
    row = methods.iloc[idx]
    dominated = False
    for other_idx, other in methods.iterrows():
        if other_idx == idx:
            continue
        reward_not_worse = float(other["reward"]) >= float(row["reward"])
        calls_not_worse = float(other["retrieval_calls"]) <= float(row["retrieval_calls"])
        at_least_one_strict = float(other["reward"]) > float(row["reward"]) or float(other["retrieval_calls"]) < float(row["retrieval_calls"])
        if reward_not_worse and calls_not_worse and at_least_one_strict:
            dominated = True
            break
    return {
        "dataset": dataset,
        "method": str(row["method"]),
        "reward": float(row["reward"]),
        "recall_at_5": float(row["recall_at_5"]),
        "mrr": float(row["mrr"]),
        "retrieval_calls": float(row["retrieval_calls"]),
        "rewrite_cost": float(row["rewrite_cost"]),
        "dominated": bool(dominated),
        "reason": "lower_or_equal_reward_and_higher_or_equal_calls" if dominated else "pareto_frontier",
    }


def _budget_row(
    methods: pd.DataFrame,
    budget: float,
    dataset: str,
    frontier_rows: list[dict[str, object]],
) -> dict[str, object]:
    feasible = methods[methods["retrieval_calls"] <= budget].copy()
    dominated_count = sum(1 for row in frontier_rows if row["dominated"])
    if feasible.empty:
        return {
            "dataset": dataset,
            "budget": float(budget),
            "selected_method": "",
            "reward": None,
            "recall_at_5": None,
            "mrr": None,
            "retrieval_calls": None,
            "rewrite_cost": None,
            "feasible_methods": 0,
            "dominated_methods_count": dominated_count,
        }
    selected = feasible.sort_values(["reward", "retrieval_calls"], ascending=[False, True]).iloc[0]
    return {
        "dataset": dataset,
        "budget": float(budget),
        "selected_method": str(selected["method"]),
        "reward": float(selected["reward"]),
        "recall_at_5": float(selected["recall_at_5"]),
        "mrr": float(selected["mrr"]),
        "retrieval_calls": float(selected["retrieval_calls"]),
        "rewrite_cost": float(selected["rewrite_cost"]),
        "feasible_methods": int(len(feasible)),
        "dominated_methods_count": dominated_count,
    }
