from __future__ import annotations

import argparse
import json
from pathlib import Path

from selective_rag_rl.experiment_dashboard import build_experiment_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evidence-level dashboard from experiment artifacts.")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Artifact file or directory to scan. May be repeated.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/results/experiment_dashboard.csv"),
        help="Destination dashboard CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/EXPERIMENT_DASHBOARD.md"),
        help="Destination Markdown dashboard.",
    )
    args = parser.parse_args()

    input_paths = args.input or [Path("outputs/results")]
    metadata = build_experiment_dashboard(input_paths, args.output_csv, output_md=args.output_md)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
