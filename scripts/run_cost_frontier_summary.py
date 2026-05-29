from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.cost_frontier import run_cost_frontier_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reward/cost frontiers from policy summary CSVs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--summary-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--budgets", default="1.0,1.25,1.5,2.0")
    parser.add_argument("--exclude-pattern", action="append", default=["oracle"])
    args = parser.parse_args()

    budgets = [float(value) for value in args.budgets.split(",") if value.strip()]
    summary = run_cost_frontier_summary(
        dataset=args.dataset,
        summary_csvs=args.summary_csv,
        output_csv=args.output_csv,
        budgets=budgets,
        exclude_patterns=tuple(args.exclude_pattern),
    )
    print(json.dumps({"dataset": summary["dataset"], "outputs": summary["outputs"]}, indent=2))
    print(pd.read_csv(summary["outputs"]["budget_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
