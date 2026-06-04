from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.reports.final_paper_assets import export_final_paper_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    outputs = export_final_paper_assets(
        root=args.root.resolve(),
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
    print(pd.read_csv(outputs["main_results_csv"]).to_string(index=False))
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
