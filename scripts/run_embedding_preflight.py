from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.core.data import load_beir_dataset
from selective_rag_rl.preflight.embedding_preflight import estimate_split_embedding_workloads, write_embedding_workload_report
from selective_rag_rl.experiments.retrieval_policy_experiment import SEMANTIC_DEPTH_DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scifact", "nfcorpus"], default="nfcorpus")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=Path("outputs/cache/vertex_embeddings.jsonl"))
    parser.add_argument("--num-train-examples", type=int, default=300)
    parser.add_argument("--num-test-examples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--semantic-depth", type=int, default=SEMANTIC_DEPTH_DEFAULT)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--full-corpus", action="store_true")
    args = parser.parse_args()

    default_paths = {
        "scifact": Path("data/raw/scifact"),
        "nfcorpus": Path("data/raw/nfcorpus"),
    }
    data_path = args.data_path or default_paths[args.dataset]
    train_examples = load_beir_dataset(
        data_path,
        num_examples=args.num_train_examples,
        seed=args.seed,
        split="train",
        pool_size=args.pool_size,
        full_corpus=args.full_corpus,
        qtype=f"beir-{args.dataset}",
    )
    test_examples = load_beir_dataset(
        data_path,
        num_examples=args.num_test_examples,
        seed=args.seed,
        split="test",
        pool_size=args.pool_size,
        full_corpus=args.full_corpus,
        qtype=f"beir-{args.dataset}",
    )
    train_report, test_report, combined_report = estimate_split_embedding_workloads(
        train_examples,
        test_examples,
        cache_path=args.cache_path,
        k=args.semantic_depth,
        dataset=args.dataset,
    )
    output_csv = args.output_csv or Path("outputs/results") / f"{args.dataset}_embedding_preflight.csv"
    csv_path = write_embedding_workload_report(output_csv, train_report, test_report, combined_report)
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
