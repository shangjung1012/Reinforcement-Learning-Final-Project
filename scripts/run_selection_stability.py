from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.selection_stability import export_selection_stability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--grid-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    output_csv = args.output_csv or Path("outputs") / "results" / "selection_stability.csv"
    csv_path = export_selection_stability(args.grid_csv, output_csv, dataset=args.dataset)
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
