from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.feature_effects import export_feature_effects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--grid-csv", action="append", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs") / "results" / "feature_effects.csv")
    parser.add_argument("--baseline-feature-set", default="no_semantic")
    args = parser.parse_args()

    csv_path = export_feature_effects(
        args.grid_csv,
        args.output_csv,
        dataset=args.dataset,
        baseline_feature_set=args.baseline_feature_set,
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
