from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.diagnostics.qualitative import export_qualitative_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detailed-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-examples-per-case", type=int, default=3)
    args = parser.parse_args()

    detailed_csv = args.detailed_csv or Path("outputs") / "results" / f"{args.dataset}_retrieval_policy_detailed.csv"
    output_csv = args.output_csv or Path("outputs") / "results" / f"{args.dataset}_qualitative_examples.csv"
    csv_path = export_qualitative_examples(
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        dataset=args.dataset,
        max_examples_per_case=args.max_examples_per_case,
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
