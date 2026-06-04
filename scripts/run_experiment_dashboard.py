from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.reports.experiment_dashboard import export_experiment_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Export evidence levels for final, smoke, and API-pilot artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/results/experiment_dashboard.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/EXPERIMENT_DASHBOARD.md"))
    parser.add_argument("--claims-csv", type=Path, default=Path("outputs/results/final_claims_matrix.csv"))
    args = parser.parse_args()

    output_csv, output_md = export_experiment_dashboard(
        root=args.root.resolve(),
        output_csv=args.output_csv,
        output_md=args.output_md,
        claims_csv=args.claims_csv,
    )
    dashboard = pd.read_csv(output_csv)
    print(f"Wrote {output_csv} ({len(dashboard)} artifacts)")
    if output_md is not None:
        print(f"Wrote {output_md}")
    print(dashboard["evidence_level"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
