from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.core.data import load_beir_dataset
from selective_rag_rl.experiments.dense_experiment import FakeDenseEmbedder
from selective_rag_rl.core.dense_retriever import load_sentence_transformer
from selective_rag_rl.diagnostics.feature_group_diagnostics import feature_group_diagnostics_frame
from selective_rag_rl.diagnostics.feature_predictive_diagnostics import feature_reward_predictive_diagnostics_frame
from selective_rag_rl.diagnostics.feature_reward_diagnostics import feature_reward_diagnostics_frame
from selective_rag_rl.experiments.retrieval_policy_experiment import (
    AUTO_POLICY_MODEL_CHOICES,
    BASE_RETRIEVAL_ACTIONS,
    FEATURE_SET_CHOICES,
    SEMANTIC_DEPTH_DEFAULT,
    evaluate_retrieval_actions,
    fit_policy_feature_transform,
    select_policy_model,
    _validate_semantic_depth,
    _load_semantic_embedder,
    _prewarm_semantic_embeddings,
)

POLICY_MODEL_CHOICES = ["knn", "ridge", "ridge_sweep", "margin_ridge", "extra_trees", "random_forest", "mlp", "auto"]


def run_policy_model_sweep(
    dataset: str,
    data_path: Path,
    output_dir: Path,
    policy_models: list[str] | None = None,
    feature_set: str = "full",
    feature_sets: list[str] | None = None,
    num_train_examples: int = 300,
    num_test_examples: int = 300,
    seed: int = 42,
    full_corpus: bool = False,
    k: int = 5,
    pool_size: int = 100,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    semantic_allow_api: bool = False,
    semantic_max_new_texts: int = 0,
    semantic_dry_run: bool = False,
    knn_k_candidates: list[int] | None = None,
    tuning_folds: int = 5,
    auto_candidate_models: list[str] | None = None,
    retrieval_contrast_features: bool = False,
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
) -> Path:
    if dataset not in {"scifact", "nfcorpus"}:
        raise ValueError(f"Unsupported policy-model-sweep dataset: {dataset}")
    selected_feature_sets = feature_sets or [feature_set]
    unknown_feature_sets = sorted(set(selected_feature_sets) - set(FEATURE_SET_CHOICES))
    if unknown_feature_sets:
        raise ValueError(f"Unknown feature set(s): {', '.join(unknown_feature_sets)}")

    selected_policy_models = policy_models or POLICY_MODEL_CHOICES
    unknown = sorted(set(selected_policy_models) - set(POLICY_MODEL_CHOICES))
    if unknown:
        raise ValueError(f"Unknown policy model(s): {', '.join(unknown)}")
    selected_auto_candidates = auto_candidate_models or AUTO_POLICY_MODEL_CHOICES
    unknown_auto = sorted(set(selected_auto_candidates) - set(AUTO_POLICY_MODEL_CHOICES))
    if unknown_auto:
        raise ValueError(f"Unknown auto candidate model(s): {', '.join(unknown_auto)}")
    semantic_depth = _validate_semantic_depth(semantic_depth)

    train_examples = load_beir_dataset(
        data_path,
        num_examples=num_train_examples,
        seed=seed,
        split="train",
        pool_size=pool_size,
        full_corpus=full_corpus,
        qtype=f"beir-{dataset}",
    )
    test_examples = load_beir_dataset(
        data_path,
        num_examples=num_test_examples,
        seed=seed,
        split="test",
        pool_size=pool_size,
        full_corpus=full_corpus,
        qtype=f"beir-{dataset}",
    )
    embedder = FakeDenseEmbedder() if embedder_name == "fake" else load_sentence_transformer(embedder_name)
    semantic_embedder = _load_semantic_embedder(
        semantic_features,
        output_dir,
        semantic_cache_path,
        semantic_allow_api=semantic_allow_api,
        semantic_max_new_texts=semantic_max_new_texts,
        semantic_dry_run=semantic_dry_run,
    )
    if semantic_embedder is not None:
        _prewarm_semantic_embeddings([*train_examples, *test_examples], semantic_embedder, semantic_depth)

    retriever_cache = {}
    train_evals = [
        evaluate_retrieval_actions(
            ex,
            embedder,
            k,
            dense_weight,
            retrieval_call_cost,
            semantic_embedder,
            actions=BASE_RETRIEVAL_ACTIONS,
            retrieval_contrast_features=retrieval_contrast_features,
            retriever_cache=retriever_cache,
            semantic_depth=semantic_depth,
        )
        for ex in tqdm(train_examples, desc="model sweep train retrieval actions")
    ]
    test_evals = [
        evaluate_retrieval_actions(
            ex,
            embedder,
            k,
            dense_weight,
            retrieval_call_cost,
            semantic_embedder,
            actions=BASE_RETRIEVAL_ACTIONS,
            retrieval_contrast_features=retrieval_contrast_features,
            retriever_cache=retriever_cache,
            semantic_depth=semantic_depth,
        )
        for ex in tqdm(test_examples, desc="model sweep test retrieval actions")
    ]

    raw_features = np.vstack([row["features"] for row in train_evals])
    test_raw_features = np.vstack([row["features"] for row in test_evals])
    rewards = {action: [row["actions"][action]["reward"] for row in train_evals] for action in BASE_RETRIEVAL_ACTIONS}
    best_fixed_action = max(
        BASE_RETRIEVAL_ACTIONS,
        key=lambda action: (float(np.mean(rewards[action])), -BASE_RETRIEVAL_ACTIONS.index(action)),
    )
    best_fixed_validation_reward = float(np.mean(rewards[best_fixed_action]))
    oracle_actions = [
        max(BASE_RETRIEVAL_ACTIONS, key=lambda action: action_eval["actions"][action]["reward"])
        for action_eval in test_evals
    ]

    rows = []
    for feature_set_name in selected_feature_sets:
        feature_transform = fit_policy_feature_transform(raw_features, feature_set_name)
        features = feature_transform.transform(raw_features)
        for policy_model in selected_policy_models:
            policy, selected_policy_model, validation_scores = select_policy_model(
                features=features,
                rewards=rewards,
                policy_model=policy_model,
                knn_k_candidates=knn_k_candidates or [1, 3, 5, 7, 9, 11, 15, 21],
                folds=tuning_folds,
                seed=seed,
                actions=BASE_RETRIEVAL_ACTIONS,
                auto_candidate_models=selected_auto_candidates,
            )
            policy.fit(features, rewards)
            selected_actions = [policy.predict(feature_transform.transform(action_eval["features"])) for action_eval in test_evals]
            rows.append(
                {
                    "dataset": dataset,
                    "feature_set": feature_set_name,
                    "policy_model": policy_model,
                    "auto_candidate_models": ",".join(selected_auto_candidates) if policy_model == "auto" else "",
                    "retrieval_contrast_features": retrieval_contrast_features,
                    "semantic_depth": semantic_depth,
                    "selected_policy_model": selected_policy_model,
                    "train_size": len(train_examples),
                    "test_size": len(test_examples),
                    "feature_width": int(features.shape[1]),
                    "validation_reward": float(validation_scores[selected_policy_model]),
                    "best_fixed_validation_reward": best_fixed_validation_reward,
                    "validation_reward_gap_vs_best_fixed": round(
                        float(validation_scores[selected_policy_model] - best_fixed_validation_reward),
                        12,
                    ),
                    "selective_reward": _action_metric(test_evals, selected_actions, "reward"),
                    "best_fixed_reward": _action_metric(test_evals, [best_fixed_action] * len(test_evals), "reward"),
                    "oracle_reward": _action_metric(test_evals, oracle_actions, "reward"),
                    "selective_recall_at_5": _action_metric(test_evals, selected_actions, "recall_at_5"),
                    "best_fixed_recall_at_5": _action_metric(test_evals, [best_fixed_action] * len(test_evals), "recall_at_5"),
                    "best_fixed_retrieval_calls": _action_metric(
                        test_evals,
                        [best_fixed_action] * len(test_evals),
                        "retrieval_calls",
                    ),
                    "selective_retrieval_calls": _action_metric(test_evals, selected_actions, "retrieval_calls"),
                }
            )

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{dataset}_policy_model_sweep.csv"
    feature_diagnostics_csv = results_dir / f"{dataset}_policy_feature_diagnostics.csv"
    feature_reward_diagnostics_csv = results_dir / f"{dataset}_policy_feature_reward_diagnostics.csv"
    feature_predictive_diagnostics_csv = results_dir / f"{dataset}_policy_feature_predictive_diagnostics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    feature_diagnostics = pd.concat(
        [
            feature_group_diagnostics_frame(raw_features, dataset=dataset, split="train"),
            feature_group_diagnostics_frame(test_raw_features, dataset=dataset, split="test"),
        ],
        ignore_index=True,
    )
    feature_diagnostics.to_csv(feature_diagnostics_csv, index=False)
    feature_reward_diagnostics = pd.concat(
        [
            feature_reward_diagnostics_frame(raw_features, train_evals, dataset=dataset, split="train"),
            feature_reward_diagnostics_frame(test_raw_features, test_evals, dataset=dataset, split="test"),
        ],
        ignore_index=True,
    )
    feature_reward_diagnostics.to_csv(feature_reward_diagnostics_csv, index=False)
    feature_predictive_diagnostics = feature_reward_predictive_diagnostics_frame(
        raw_features,
        train_evals,
        test_raw_features,
        test_evals,
        dataset=dataset,
    )
    feature_predictive_diagnostics.to_csv(feature_predictive_diagnostics_csv, index=False)
    return csv_path


def _action_metric(evals: list[dict[str, object]], actions: list[str], metric: str) -> float:
    values = [float(action_eval["actions"][action][metric]) for action_eval, action in zip(evals, actions)]
    return float(np.mean(values)) if values else 0.0
