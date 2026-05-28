from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.artifact_index import export_artifact_index, final_project_artifact_specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/results/final_artifact_index.csv"))
    args = parser.parse_args()

    root = args.root.resolve()
    specs = final_project_artifact_specs(root)
    csv_path = export_artifact_index(specs, output_csv=args.output_csv, root=root)
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
