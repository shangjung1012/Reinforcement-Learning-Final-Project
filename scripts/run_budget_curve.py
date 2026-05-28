from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.budget_curve import export_budget_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--budgets", default="1.0,1.25,1.5,1.75,2.0,3.0")
    parser.add_argument("--include-oracle", action="store_true")
    args = parser.parse_args()

    budgets = [float(value) for value in args.budgets.split(",") if value.strip()]
    output_csv = args.output_csv or Path("outputs/results") / f"{args.dataset}_budget_curve.csv"
    csv_path = export_budget_curve(
        dataset=args.dataset,
        summary_csv=args.summary_csv,
        output_csv=output_csv,
        budgets=budgets,
        exclude_methods=[] if args.include_oracle else ["Oracle retrieval action"],
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
