from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.bandit_baselines import run_beir_linucb_baseline
from selective_rag_rl.retrieval_policy_experiment import FEATURE_SET_CHOICES, SEMANTIC_DEPTH_DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scifact", "nfcorpus"], required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-train-examples", type=int, default=300)
    parser.add_argument("--num-test-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--retrieval-call-cost", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--posterior-scale", type=float, default=1.0)
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="full")
    parser.add_argument("--semantic-features", choices=["none", "vertex"], default="none")
    parser.add_argument("--semantic-cache-path", type=Path, default=None)
    parser.add_argument("--semantic-depth", type=int, default=SEMANTIC_DEPTH_DEFAULT)
    args = parser.parse_args()

    data_path = args.data_path or Path("data/raw") / args.dataset
    summary_csv = run_beir_linucb_baseline(
        dataset=args.dataset,
        data_path=data_path,
        output_dir=args.output_dir,
        num_train_examples=args.num_train_examples,
        num_test_examples=args.num_test_examples,
        seed=args.seed,
        full_corpus=args.full_corpus,
        k=args.k,
        pool_size=args.pool_size,
        embedder_name=args.embedder,
        dense_weight=args.dense_weight,
        retrieval_call_cost=args.retrieval_call_cost,
        alpha=args.alpha,
        l2=args.l2,
        epsilon=args.epsilon,
        posterior_scale=args.posterior_scale,
        feature_set=args.feature_set,
        semantic_features=args.semantic_features,
        semantic_cache_path=args.semantic_cache_path,
        semantic_depth=args.semantic_depth,
    )
    print(pd.read_csv(summary_csv).to_string(index=False))


if __name__ == "__main__":
    main()
