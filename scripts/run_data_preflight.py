from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.preflight.data_preflight import run_data_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Check expected raw-data paths without printing raw data.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_data_preflight"))
    args = parser.parse_args()

    summary = run_data_preflight(project_root=Path.cwd(), output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "all_required_available": summary["all_required_available"],
                "required_missing_count": summary["required_missing_count"],
                "datasets": summary["datasets"],
                "outputs": summary["outputs"],
            },
            indent=2,
        )
    )
    csv_path = Path(summary["outputs"]["summary_csv"])
    if csv_path.exists():
        print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
