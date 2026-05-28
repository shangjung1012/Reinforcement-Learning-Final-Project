from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.nq_experiment import run_nq_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/natural-questions/default/validation-00000-of-00007.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    metadata = run_nq_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        seed=args.seed,
        pool_size=args.pool_size,
        k=args.k,
    )
    print(pd.read_csv(metadata["outputs"]["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
