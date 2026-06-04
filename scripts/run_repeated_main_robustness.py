from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.policies.bandit_baselines import run_beir_linucb_baseline
from selective_rag_rl.policies.constrained_policy import export_constrained_policy_sweep
from selective_rag_rl.experiments.repeated_main_robustness import export_repeated_main_robustness
from selective_rag_rl.experiments.retrieval_policy_experiment import (
    FEATURE_SET_CHOICES,
    SEMANTIC_DEPTH_DEFAULT,
    run_nfcorpus_retrieval_policy_experiment,
    run_scifact_retrieval_policy_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="scifact,nfcorpus")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "repeated_main_runs")
    parser.add_argument("--num-train-examples", type=int, default=600)
    parser.add_argument("--num-test-examples", type=int, default=300)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--retrieval-call-cost", type=float, default=0.03)
    parser.add_argument("--policy-model", choices=["knn", "ridge", "ridge_sweep", "margin_ridge", "extra_trees", "random_forest", "mlp", "auto"], default="auto")
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="full")
    parser.add_argument("--knn-k-candidates", default="1,3,5,7,9,11,15,21")
    parser.add_argument("--tuning-folds", type=int, default=5)
    parser.add_argument("--semantic-depth", type=int, default=SEMANTIC_DEPTH_DEFAULT)
    parser.add_argument("--linucb-alpha", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--posterior-scale", type=float, default=1.0)
    parser.add_argument("--call-penalty", type=float, default=0.03)
    args = parser.parse_args()

    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    knn_k_candidates = [int(value) for value in args.knn_k_candidates.split(",") if value.strip()]
    manifest_rows = []
    for dataset in datasets:
        for seed in seeds:
            run_dir = args.output_dir / f"{dataset}_seed_{seed}"
            policy_metadata = _run_policy(
                dataset=dataset,
                output_dir=run_dir,
                seed=seed,
                num_train_examples=args.num_train_examples,
                num_test_examples=args.num_test_examples,
                k=args.k,
                pool_size=args.pool_size,
                full_corpus=args.full_corpus,
                embedder_name=args.embedder,
                dense_weight=args.dense_weight,
                retrieval_call_cost=args.retrieval_call_cost,
                policy_model=args.policy_model,
                feature_set=args.feature_set,
                knn_k_candidates=knn_k_candidates,
                tuning_folds=args.tuning_folds,
                semantic_depth=args.semantic_depth,
            )
            bandit_csv = run_beir_linucb_baseline(
                dataset=dataset,
                data_path=_data_path(dataset),
                output_dir=run_dir,
                num_train_examples=args.num_train_examples,
                num_test_examples=args.num_test_examples,
                seed=seed,
                full_corpus=args.full_corpus,
                k=args.k,
                pool_size=args.pool_size,
                embedder_name=args.embedder,
                dense_weight=args.dense_weight,
                retrieval_call_cost=args.retrieval_call_cost,
                alpha=args.linucb_alpha,
                epsilon=args.epsilon,
                posterior_scale=args.posterior_scale,
                feature_set=args.feature_set,
                semantic_depth=args.semantic_depth,
            )
            detailed_csv = Path(policy_metadata["outputs"]["detailed_csv"])
            constrained_csv = run_dir / "results" / f"{dataset}_constrained_policy_sweep.csv"
            export_constrained_policy_sweep(
                dataset=dataset,
                detailed_csv=detailed_csv,
                output_csv=constrained_csv,
                call_penalties=[args.call_penalty],
            )
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "policy_summary_csv": policy_metadata["outputs"]["summary_csv"],
                    "bandit_summary_csv": str(bandit_csv),
                    "constrained_sweep_csv": str(constrained_csv),
                }
            )

    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = results_dir / "repeated_main_robustness_manifest.csv"
    per_seed_csv = results_dir / "repeated_main_robustness_per_seed.csv"
    aggregate_csv = results_dir / "repeated_main_robustness_aggregate.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    export_repeated_main_robustness(
        manifest_csv=manifest_csv,
        per_seed_csv=per_seed_csv,
        aggregate_csv=aggregate_csv,
        call_penalty=args.call_penalty,
    )
    print(pd.read_csv(aggregate_csv).to_string(index=False))


def _run_policy(
    *,
    dataset: str,
    output_dir: Path,
    seed: int,
    num_train_examples: int,
    num_test_examples: int,
    k: int,
    pool_size: int,
    full_corpus: bool,
    embedder_name: str,
    dense_weight: float,
    retrieval_call_cost: float,
    policy_model: str,
    feature_set: str,
    knn_k_candidates: list[int],
    tuning_folds: int,
    semantic_depth: int,
) -> dict[str, object]:
    common = {
        "data_path": _data_path(dataset),
        "output_dir": output_dir,
        "num_train_examples": num_train_examples,
        "num_test_examples": num_test_examples,
        "seed": seed,
        "k": k,
        "pool_size": pool_size,
        "full_corpus": full_corpus,
        "embedder_name": embedder_name,
        "dense_weight": dense_weight,
        "retrieval_call_cost": retrieval_call_cost,
        "policy_model": policy_model,
        "feature_set": feature_set,
        "knn_k_candidates": knn_k_candidates,
        "tuning_folds": tuning_folds,
        "semantic_depth": semantic_depth,
    }
    if dataset == "scifact":
        return run_scifact_retrieval_policy_experiment(**common)
    if dataset == "nfcorpus":
        return run_nfcorpus_retrieval_policy_experiment(**common)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _data_path(dataset: str) -> Path:
    paths = {"scifact": Path("data/raw/scifact"), "nfcorpus": Path("data/raw/nfcorpus")}
    if dataset not in paths:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return paths[dataset]


if __name__ == "__main__":
    main()
