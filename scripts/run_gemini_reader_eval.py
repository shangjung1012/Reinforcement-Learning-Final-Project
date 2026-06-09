from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from selective_rag_rl.experiments.gemini_reader import evaluate_gemini_reader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded Gemini answer-reader EM/F1 evaluation against deterministic reader baselines.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", choices=["toy", "hotpot", "nq"], default="toy")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_gemini_reader"))
    parser.add_argument("--num-examples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cache-path", type=Path, default=Path("outputs/cache/codex_gemini_reader_answers.jsonl"))
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allow_api = args.allow_api or os.environ.get("CODEX_ALLOW_API_CALLS") == "1"
    metadata = evaluate_gemini_reader(
        dataset=args.dataset,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        seed=args.seed,
        k=args.k,
        cache_path=args.cache_path,
        allow_api=allow_api,
        max_new_calls=args.max_new_calls,
        dry_run=args.dry_run,
    )
    print(json.dumps(metadata, indent=2))
    summary_csv = metadata["outputs"].get("summary_csv")
    if summary_csv:
        print(pd.read_csv(summary_csv).to_string(index=False))


if __name__ == "__main__":
    main()
