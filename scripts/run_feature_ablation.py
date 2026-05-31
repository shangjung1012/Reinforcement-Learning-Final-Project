from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.feature_ablation import run_feature_ablation
from selective_rag_rl.retrieval_policy_experiment import FEATURE_SET_CHOICES, SEMANTIC_DEPTH_DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scifact", "nfcorpus"], default="nfcorpus")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SET_CHOICES))
    parser.add_argument("--num-train-examples", type=int, default=300)
    parser.add_argument("--num-test-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--retrieval-call-cost", type=float, default=0.03)
    parser.add_argument("--semantic-features", choices=["none", "vertex"], default="none")
    parser.add_argument("--semantic-cache-path", type=Path, default=None)
    parser.add_argument("--semantic-allow-api", action="store_true")
    parser.add_argument("--semantic-max-new-texts", type=int, default=0)
    parser.add_argument("--semantic-dry-run", action="store_true")
    parser.add_argument("--semantic-depth", type=int, default=SEMANTIC_DEPTH_DEFAULT)
    parser.add_argument("--policy-model", choices=["knn", "ridge", "ridge_sweep", "margin_ridge", "extra_trees", "random_forest", "mlp", "auto"], default="ridge")
    parser.add_argument("--knn-k-candidates", default="1,3,5,7,9,11,15,21")
    parser.add_argument("--tuning-folds", type=int, default=5)
    parser.add_argument("--retrieval-contrast-features", action="store_true")
    args = parser.parse_args()

    default_paths = {
        "scifact": Path("data/raw/scifact"),
        "nfcorpus": Path("data/raw/nfcorpus"),
    }
    feature_sets = [value.strip() for value in args.feature_sets.split(",") if value.strip()]
    knn_k_candidates = [int(value) for value in args.knn_k_candidates.split(",") if value.strip()]
    csv_path = run_feature_ablation(
        dataset=args.dataset,
        data_path=args.data_path or default_paths[args.dataset],
        output_dir=args.output_dir,
        feature_sets=feature_sets,
        num_train_examples=args.num_train_examples,
        num_test_examples=args.num_test_examples,
        seed=args.seed,
        k=args.k,
        pool_size=args.pool_size,
        full_corpus=args.full_corpus,
        embedder_name=args.embedder,
        dense_weight=args.dense_weight,
        retrieval_call_cost=args.retrieval_call_cost,
        semantic_features=args.semantic_features,
        semantic_cache_path=args.semantic_cache_path,
        semantic_allow_api=args.semantic_allow_api,
        semantic_max_new_texts=args.semantic_max_new_texts,
        semantic_dry_run=args.semantic_dry_run,
        semantic_depth=args.semantic_depth,
        policy_model=args.policy_model,
        knn_k_candidates=knn_k_candidates,
        tuning_folds=args.tuning_folds,
        retrieval_contrast_features=args.retrieval_contrast_features,
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
