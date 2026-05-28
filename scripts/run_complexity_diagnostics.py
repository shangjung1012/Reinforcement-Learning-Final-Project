from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.complexity_diagnostics import export_complexity_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detailed-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--action-distribution-csv", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    bucket_csv, action_csv = export_complexity_diagnostics(
        dataset=args.dataset,
        detailed_csv=args.detailed_csv,
        output_csv=args.output_csv,
        action_distribution_csv=args.action_distribution_csv,
        split=args.split,
    )
    print(pd.read_csv(bucket_csv).head().to_string(index=False))
    print(pd.read_csv(action_csv).head().to_string(index=False))


if __name__ == "__main__":
    main()
