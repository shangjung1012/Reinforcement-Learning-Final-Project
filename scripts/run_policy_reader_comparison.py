from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from selective_rag_rl.experiments.policy_reader_comparison import run_policy_reader_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare deterministic readers on BM25, train-best fixed, and learned-policy retrieval outputs.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", choices=["hotpot", "nq"], required=True)
    parser.add_argument("--detailed-csv", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_policy_reader"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--publish-results", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--source-num-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--readers", default="lexical,span,answer_type")
    parser.add_argument(
        "--retrieval-methods",
        default="Vanilla BM25,Train-best retrieval action,Selective retrieval policy",
    )
    args = parser.parse_args()

    readers = [reader.strip() for reader in args.readers.split(",") if reader.strip()]
    retrieval_methods = [method.strip() for method in args.retrieval_methods.split(",") if method.strip()]
    metadata = run_policy_reader_comparison(
        dataset=args.dataset,
        detailed_csv=args.detailed_csv,
        data_path=args.data_path,
        output_dir=args.output_dir,
        readers=readers,
        retrieval_methods=retrieval_methods,
        split=args.split,
        num_examples=args.num_examples,
        source_num_examples=args.source_num_examples,
        seed=args.seed,
        k=args.k,
    )
    if args.publish_results:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{args.dataset}_policy_reader_comparison"
        summary_dst = args.results_dir / f"{prefix}_summary.csv"
        detailed_dst = args.results_dir / f"{prefix}_detailed.csv"
        shutil.copyfile(metadata["outputs"]["summary_csv"], summary_dst)
        shutil.copyfile(metadata["outputs"]["detailed_csv"], detailed_dst)
        metadata["published_outputs"] = {
            "summary_csv": str(summary_dst),
            "detailed_csv": str(detailed_dst),
        }
    print(json.dumps(metadata, indent=2))
    print(pd.read_csv(metadata["outputs"]["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
