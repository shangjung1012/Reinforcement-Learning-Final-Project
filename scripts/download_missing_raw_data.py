from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.preflight.raw_data_download import available_dataset_keys, download_missing_raw_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download selected missing raw datasets into ignored data/raw paths.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[*available_dataset_keys(), "all"],
        required=True,
        help="Dataset key to download. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_data_download"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prefer-hf",
        action="store_true",
        help="Use Hugging Face Hub mirror first when the selected dataset has one.",
    )
    args = parser.parse_args()

    summary = download_missing_raw_data(
        project_root=Path.cwd(),
        dataset_keys=args.dataset,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        prefer_hf=args.prefer_hf,
    )
    print(
        json.dumps(
            {
                "dry_run": summary["dry_run"],
                "overwrite": summary["overwrite"],
                "prefer_hf": summary["prefer_hf"],
                "requested_dataset_keys": summary["requested_dataset_keys"],
                "downloaded_count": summary["downloaded_count"],
                "already_exists_count": summary["already_exists_count"],
                "would_download_count": summary["would_download_count"],
                "outputs": summary["outputs"],
            },
            indent=2,
        )
    )
    print(pd.read_csv(summary["outputs"]["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
