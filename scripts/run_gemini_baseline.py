from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.gemini_baseline import evaluate_gemini_rewrites


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/HotpotQA/hotpot_dev_distractor_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-examples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cache-path", type=Path, default=Path("outputs/cache/gemini_rewrites.jsonl"))
    args = parser.parse_args()

    metadata = evaluate_gemini_rewrites(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        seed=args.seed,
        k=args.k,
        cache_path=args.cache_path,
    )
    print(pd.read_csv(metadata["outputs"]["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
