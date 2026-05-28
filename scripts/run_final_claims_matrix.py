from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.final_claims import export_final_claims_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/results/final_claims_matrix.csv"))
    args = parser.parse_args()

    csv_path = export_final_claims_matrix(root=args.root.resolve(), output_csv=args.output_csv)
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
