from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.retrieval_policy_experiment import FEATURE_SET_CHOICES, SEMANTIC_DEPTH_DEFAULT, run_nq_retrieval_policy_experiment


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
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--retrieval-call-cost", type=float, default=0.03)
    parser.add_argument("--semantic-features", choices=["none", "vertex"], default="none")
    parser.add_argument("--semantic-cache-path", type=Path, default=None)
    parser.add_argument("--semantic-allow-api", action="store_true")
    parser.add_argument("--semantic-max-new-texts", type=int, default=0)
    parser.add_argument("--semantic-dry-run", action="store_true")
    parser.add_argument("--semantic-depth", type=int, default=SEMANTIC_DEPTH_DEFAULT)
    parser.add_argument("--policy-model", choices=["knn", "ridge", "ridge_sweep", "margin_ridge", "extra_trees", "random_forest", "mlp", "auto"], default="knn")
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="full")
    parser.add_argument("--knn-k-candidates", default="1,3,5,7,9,11,15,21")
    parser.add_argument("--tuning-folds", type=int, default=5)
    parser.add_argument("--confidence-gate-margin", type=float, default=None)
    args = parser.parse_args()
    knn_k_candidates = [int(value) for value in args.knn_k_candidates.split(",") if value.strip()]

    metadata = run_nq_retrieval_policy_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        seed=args.seed,
        k=args.k,
        pool_size=args.pool_size,
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
        feature_set=args.feature_set,
        knn_k_candidates=knn_k_candidates,
        tuning_folds=args.tuning_folds,
        confidence_gate_margin=args.confidence_gate_margin,
    )
    print(pd.read_csv(metadata["outputs"]["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
