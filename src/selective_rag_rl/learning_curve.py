from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.retrieval_policy_experiment import (
    SEMANTIC_DEPTH_DEFAULT,
    run_nfcorpus_retrieval_policy_experiment,
    run_scifact_retrieval_policy_experiment,
)


def run_policy_learning_curve(
    dataset: str,
    data_path: Path,
    output_dir: Path,
    train_sizes: list[int],
    num_test_examples: int,
    seed: int = 42,
    full_corpus: bool = False,
    k: int = 5,
    pool_size: int = 100,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    policy_model: str = "auto",
    feature_set: str = "full",
    knn_k_candidates: list[int] | None = None,
    tuning_folds: int = 5,
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
) -> Path:
    if dataset not in {"scifact", "nfcorpus"}:
        raise ValueError(f"Unsupported learning-curve dataset: {dataset}")

    rows = []
    for train_size in train_sizes:
        run_dir = output_dir / "learning_curve_runs" / f"{dataset}_train_{train_size}"
        if dataset == "scifact":
            metadata = run_scifact_retrieval_policy_experiment(
                data_path=data_path,
                output_dir=run_dir,
                num_train_examples=train_size,
                num_test_examples=num_test_examples,
                seed=seed,
                k=k,
                pool_size=pool_size,
                full_corpus=full_corpus,
                embedder_name=embedder_name,
                dense_weight=dense_weight,
                retrieval_call_cost=retrieval_call_cost,
                semantic_features=semantic_features,
                semantic_cache_path=semantic_cache_path,
                semantic_depth=semantic_depth,
                policy_model=policy_model,
                feature_set=feature_set,
                knn_k_candidates=knn_k_candidates,
                tuning_folds=tuning_folds,
            )
        else:
            metadata = run_nfcorpus_retrieval_policy_experiment(
                data_path=data_path,
                output_dir=run_dir,
                num_train_examples=train_size,
                num_test_examples=num_test_examples,
                seed=seed,
                k=k,
                pool_size=pool_size,
                full_corpus=full_corpus,
                embedder_name=embedder_name,
                dense_weight=dense_weight,
                retrieval_call_cost=retrieval_call_cost,
                semantic_features=semantic_features,
                semantic_cache_path=semantic_cache_path,
                semantic_depth=semantic_depth,
                policy_model=policy_model,
                feature_set=feature_set,
                knn_k_candidates=knn_k_candidates,
                tuning_folds=tuning_folds,
            )
        summary = pd.read_csv(metadata["outputs"]["summary_csv"])
        rows.append(
            {
                "dataset": dataset,
                "train_size": train_size,
                "test_size": metadata["test_examples"],
                "selected_policy_model": metadata["selected_policy_model"],
                "semantic_depth": int(semantic_depth),
                "selective_reward": _metric(summary, "Selective retrieval policy", "reward"),
                "best_fixed_reward": _metric(summary, "Train-best retrieval action", "reward"),
                "oracle_reward": _metric(summary, "Oracle retrieval action", "reward"),
                "selective_recall_at_5": _metric(summary, "Selective retrieval policy", "recall_at_5"),
                "best_fixed_recall_at_5": _metric(summary, "Train-best retrieval action", "recall_at_5"),
                "selective_retrieval_calls": _metric(summary, "Selective retrieval policy", "retrieval_calls"),
            }
        )

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{dataset}_policy_learning_curve.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _metric(summary: pd.DataFrame, method: str, column: str) -> float:
    value = summary.loc[summary["method"] == method, column]
    return float(value.iloc[0]) if len(value) else 0.0
