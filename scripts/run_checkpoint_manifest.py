from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.checkpoint_manifest import (
    export_checkpoint_manifest,
    final_project_checkpoint_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/results/final_checkpoint_manifest.csv"))
    args = parser.parse_args()

    root = args.root.resolve()
    csv_path = export_checkpoint_manifest(
        final_project_checkpoint_paths(root),
        output_csv=args.output_csv,
        root=root,
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
