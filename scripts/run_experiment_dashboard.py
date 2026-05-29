from __future__ import annotations

import argparse
from pathlib import Path

from selective_rag_rl.experiment_dashboard import write_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evidence-level dashboard for generated artifacts.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/results/experiment_dashboard.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/EXPERIMENT_DASHBOARD.md"))
    args = parser.parse_args()

    outputs = write_dashboard(Path.cwd(), output_csv=args.output_csv, output_md=args.output_md)
    print(outputs)


if __name__ == "__main__":
    main()
