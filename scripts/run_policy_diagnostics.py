from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.diagnostics.policy_diagnostics import export_policy_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detailed-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    detailed_csv = args.detailed_csv or Path("outputs") / "results" / f"{args.dataset}_retrieval_policy_detailed.csv"
    output_csv = args.output_csv or Path("outputs") / "results" / f"{args.dataset}_policy_diagnostics.csv"
    csv_path = export_policy_diagnostics(detailed_csv, output_csv, dataset=args.dataset)
    diagnostics = pd.read_csv(csv_path)
    print(diagnostics.describe(include="all").to_string())


if __name__ == "__main__":
    main()
