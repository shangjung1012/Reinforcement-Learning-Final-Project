from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from selective_rag_rl.experiments.policy_gemini_reader_comparison import (
    run_policy_gemini_reader_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Gemini answer reader on BM25, train-best fixed, and learned-policy retrieval outputs.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", choices=["hotpot", "nq"], required=True)
    parser.add_argument("--detailed-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_policy_gemini_reader"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--source-num-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--retrieval-methods",
        default="Vanilla BM25,Train-best retrieval action,Selective retrieval policy",
    )
    parser.add_argument("--cache-path", type=Path, default=Path("outputs/cache/codex_policy_gemini_reader.jsonl"))
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish-results", action="store_true")
    args = parser.parse_args()

    retrieval_methods = [method.strip() for method in args.retrieval_methods.split(",") if method.strip()]
    allow_api = args.allow_api or os.environ.get("CODEX_ALLOW_API_CALLS") == "1"
    metadata = run_policy_gemini_reader_comparison(
        dataset=args.dataset,
        detailed_csv=args.detailed_csv,
        output_dir=args.output_dir,
        retrieval_methods=retrieval_methods,
        split=args.split,
        num_examples=args.num_examples,
        seed=args.seed,
        k=args.k,
        data_path=args.data_path,
        source_num_examples=args.source_num_examples,
        cache_path=args.cache_path,
        allow_api=allow_api,
        max_new_calls=args.max_new_calls,
        dry_run=args.dry_run,
    )
    if args.publish_results and "summary_csv" in metadata["outputs"]:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{args.dataset}_policy_gemini_reader_comparison"
        summary = pd.read_csv(metadata["outputs"]["summary_csv"])
        detailed = pd.read_csv(metadata["outputs"]["detailed_csv"])
        summary.to_csv(args.results_dir / f"{prefix}_summary.csv", index=False)
        detailed.to_csv(args.results_dir / f"{prefix}_detailed.csv", index=False)
        metadata["published_outputs"] = {
            "summary_csv": str(args.results_dir / f"{prefix}_summary.csv"),
            "detailed_csv": str(args.results_dir / f"{prefix}_detailed.csv"),
        }
    print(json.dumps(metadata, indent=2))
    summary_csv = metadata["outputs"].get("summary_csv")
    if summary_csv:
        print(pd.read_csv(summary_csv).to_string(index=False))


if __name__ == "__main__":
    main()
