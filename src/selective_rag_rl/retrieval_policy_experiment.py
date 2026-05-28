from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.bandit import (
    DirectMethodBandit,
    KnnDirectMethodBandit,
    MarginWeightedDirectMethodBandit,
    NeuralRegressorFactory,
    SklearnRegressorBandit,
    TreeRegressorFactory,
    state_features,
)
from selective_rag_rl.data import QAExample, load_beir_dataset, load_hotpotqa, load_natural_questions, split_examples
from selective_rag_rl.dense_experiment import FakeDenseEmbedder
from selective_rag_rl.dense_retriever import DenseRetriever, hybrid_merge, load_sentence_transformer
from selective_rag_rl.experiment import _latex_table, _merge_results
from selective_rag_rl.gemini_baseline import GeminiCache, RewriteProvider, VertexGeminiRewriter
from selective_rag_rl.heuristic_policy import heuristic_retrieval_action
from selective_rag_rl.metrics import mrr, ndcg_at_k, recall_at_k
from selective_rag_rl.policy_confidence import confidence_gated_action, prediction_margin
from selective_rag_rl.policy_io import save_checkpoint
from selective_rag_rl.retriever import BM25Retriever, RetrievalResult
from selective_rag_rl.rewrites import rewrite, rewrite_cost
from selective_rag_rl.text import content_tokens
from selective_rag_rl.vertex_embeddings import SemanticEmbedder, VertexTextEmbeddingProvider

BASE_RETRIEVAL_ACTIONS = [
    "bm25_keep",
    "bm25_keyword",
    "dense_keep",
    "dense_keyword",
    "hybrid_keep",
    "hybrid_keyword",
]
LLM_RETRIEVAL_ACTIONS = [
    "bm25_llm_rewrite",
    "bm25_llm_decompose",
    "hybrid_llm_decompose",
    "bm25_hyde",
    "dense_hyde",
    "hybrid_hyde",
    "bm25_multi_query",
    "hybrid_multi_query",
]
RETRIEVAL_ACTIONS = BASE_RETRIEVAL_ACTIONS
RETRIEVAL_ACTIONS_WITH_LLM = [*BASE_RETRIEVAL_ACTIONS, *LLM_RETRIEVAL_ACTIONS]
RIDGE_L2_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]
AUTO_POLICY_MODEL_CHOICES = ["knn", "ridge", "extra_trees", "random_forest", "mlp"]

BASE_METHOD_ORDER = [
    "Vanilla BM25",
    "BM25 keyword",
    "Dense original",
    "Dense keyword",
    "Hybrid original",
    "Hybrid keyword",
    "Train-best retrieval action",
    "Heuristic retrieval router",
    "Selective retrieval policy",
    "Oracle retrieval action",
]
CONFIDENCE_GATED_METHOD = "Confidence-gated retrieval policy"
LLM_METHOD_ORDER = [
    "Vanilla BM25",
    "BM25 keyword",
    "Dense original",
    "Dense keyword",
    "Hybrid original",
    "Hybrid keyword",
    "Gemini rewrite action",
    "Gemini decompose action",
    "Gemini decompose hybrid action",
    "Gemini HyDE BM25 action",
    "Gemini HyDE dense action",
    "Gemini HyDE hybrid action",
    "Gemini multi-query BM25 action",
    "Gemini multi-query hybrid action",
    "Train-best retrieval action",
    "Heuristic retrieval router",
    "Selective retrieval policy",
    "Oracle retrieval action",
]
METHOD_ORDER = BASE_METHOD_ORDER

BASE_ACTION_TO_METHOD = {
    "bm25_keep": "Vanilla BM25",
    "bm25_keyword": "BM25 keyword",
    "dense_keep": "Dense original",
    "dense_keyword": "Dense keyword",
    "hybrid_keep": "Hybrid original",
    "hybrid_keyword": "Hybrid keyword",
}
LLM_ACTION_TO_METHOD = {
    **BASE_ACTION_TO_METHOD,
    "bm25_llm_rewrite": "Gemini rewrite action",
    "bm25_llm_decompose": "Gemini decompose action",
    "hybrid_llm_decompose": "Gemini decompose hybrid action",
    "bm25_hyde": "Gemini HyDE BM25 action",
    "dense_hyde": "Gemini HyDE dense action",
    "hybrid_hyde": "Gemini HyDE hybrid action",
    "bm25_multi_query": "Gemini multi-query BM25 action",
    "hybrid_multi_query": "Gemini multi-query hybrid action",
}
ACTION_TO_METHOD = BASE_ACTION_TO_METHOD
FEATURE_SET_CHOICES = [
    "full",
    "no_query",
    "no_retrieval",
    "no_wh",
    "retrieval_only",
    "no_semantic",
    "semantic_only",
    "no_profile",
    "profile_only",
    "no_score_shape",
    "score_shape_only",
    "no_rank_agreement",
    "rank_agreement_only",
    "no_projection",
    "projection_only",
    "no_interactions",
    "interactions_only",
    "semantic_zscore",
    "no_contrast",
    "contrast_only",
]
QUERY_FEATURE_IDXS = [1, 2]
RETRIEVAL_FEATURE_IDXS = [3, 4, 5]
WH_FEATURE_START = 6
WH_FEATURE_END = 14
SEMANTIC_DEPTH_DEFAULT = 5
SEMANTIC_DEPTH_MIN = 3
SEMANTIC_DEPTH_MAX = 8
SEMANTIC_FEATURE_START = 14
SEMANTIC_FEATURE_END = 23
SEMANTIC_PROFILE_START = 23
SEMANTIC_PROFILE_END = 32
SEMANTIC_SCORE_SHAPE_START = 32
SEMANTIC_SCORE_SHAPE_END = 36
SEMANTIC_RANK_AGREEMENT_START = 36
SEMANTIC_RANK_AGREEMENT_END = 42
PROJECTION_FEATURE_START = 42
PROJECTION_FEATURE_END = 54
INTERACTION_FEATURE_START = 54
RETRIEVAL_CONTRAST_FEATURE_WIDTH = 8
SEMANTIC_SUMMARY_WIDTH = 9
SEMANTIC_SCORE_SHAPE_WIDTH = 4
SEMANTIC_RANK_AGREEMENT_WIDTH = 6
EMBEDDING_PROJECTION_WIDTH = 12
LEXICAL_SEMANTIC_INTERACTION_WIDTH = 6


@dataclass(frozen=True)
class PolicyFeatureTransform:
    feature_set: str
    semantic_mean: np.ndarray | None = None
    semantic_std: np.ndarray | None = None

    def transform(self, features: np.ndarray) -> np.ndarray:
        return _transform_policy_features(
            features,
            self.feature_set,
            semantic_mean=self.semantic_mean,
            semantic_std=self.semantic_std,
        )


def fit_policy_feature_transform(features: np.ndarray, feature_set: str) -> PolicyFeatureTransform:
    if feature_set != "semantic_zscore":
        return PolicyFeatureTransform(feature_set=feature_set)
    transformed = transform_policy_features(features, "full")
    if transformed.ndim == 1:
        transformed = transformed.reshape(1, -1)
    semantic = transformed[:, SEMANTIC_FEATURE_START : _contrast_feature_start(transformed.shape[1])]
    mean = semantic.mean(axis=0) if semantic.size else np.asarray([], dtype=float)
    std = semantic.std(axis=0) if semantic.size else np.asarray([], dtype=float)
    std = np.where(std <= 1e-12, 1.0, std)
    return PolicyFeatureTransform(feature_set=feature_set, semantic_mean=mean, semantic_std=std)


def semantic_feature_group_slices(width: int) -> dict[str, tuple[int, int]]:
    semantic_end = _contrast_feature_start(width)
    return _semantic_group_slices_for_end(semantic_end)


def _semantic_group_slices_for_end(semantic_end: int) -> dict[str, tuple[int, int]]:
    if semantic_end <= SEMANTIC_FEATURE_START:
        return {}
    semantic_block_width = semantic_end - SEMANTIC_FEATURE_START
    with_score_shape = _semantic_group_slices_from_core_width(
        semantic_end,
        semantic_block_width,
        include_interactions=False,
        include_score_shape=True,
    )
    with_score_shape_and_interactions = _semantic_group_slices_from_core_width(
        semantic_end,
        semantic_block_width - LEXICAL_SEMANTIC_INTERACTION_WIDTH,
        include_interactions=True,
        include_score_shape=True,
    )
    legacy_without_interactions = _semantic_group_slices_from_core_width(semantic_end, semantic_block_width, False)
    with_interactions = _semantic_group_slices_from_core_width(
        semantic_end,
        semantic_block_width - LEXICAL_SEMANTIC_INTERACTION_WIDTH,
        True,
    )
    default_semantic_core_width_with_shape = (
        SEMANTIC_FEATURE_END
        - SEMANTIC_FEATURE_START
        + (SEMANTIC_PROFILE_END - SEMANTIC_PROFILE_START)
        + SEMANTIC_SCORE_SHAPE_WIDTH
        + SEMANTIC_RANK_AGREEMENT_WIDTH
        + EMBEDDING_PROJECTION_WIDTH
    )
    default_semantic_core_width = (
        SEMANTIC_FEATURE_END
        - SEMANTIC_FEATURE_START
        + (SEMANTIC_PROFILE_END - SEMANTIC_PROFILE_START)
        + SEMANTIC_RANK_AGREEMENT_WIDTH
        + EMBEDDING_PROJECTION_WIDTH
    )
    if (
        with_score_shape_and_interactions
        and semantic_block_width >= default_semantic_core_width_with_shape + LEXICAL_SEMANTIC_INTERACTION_WIDTH
    ):
        return with_score_shape_and_interactions
    if with_score_shape:
        return with_score_shape
    if with_interactions and semantic_block_width != default_semantic_core_width:
        return with_interactions
    return legacy_without_interactions or with_interactions


def _semantic_group_slices_from_core_width(
    semantic_end: int,
    core_width: int,
    include_interactions: bool,
    include_score_shape: bool = False,
) -> dict[str, tuple[int, int]]:
    score_shape_width = SEMANTIC_SCORE_SHAPE_WIDTH if include_score_shape else 0
    profile_width = (
        core_width
        - SEMANTIC_SUMMARY_WIDTH
        - score_shape_width
        - SEMANTIC_RANK_AGREEMENT_WIDTH
        - EMBEDDING_PROJECTION_WIDTH
    )
    if profile_width < 1 or profile_width % 2 == 0:
        return {}

    summary_start = SEMANTIC_FEATURE_START
    profile_start = summary_start + SEMANTIC_SUMMARY_WIDTH
    score_shape_start = profile_start + profile_width
    rank_start = score_shape_start + score_shape_width
    projection_start = rank_start + SEMANTIC_RANK_AGREEMENT_WIDTH
    interaction_start = projection_start + EMBEDDING_PROJECTION_WIDTH
    expected_end = interaction_start + (LEXICAL_SEMANTIC_INTERACTION_WIDTH if include_interactions else 0)
    if expected_end != semantic_end:
        return {}
    groups = {
        "semantic_summary": (summary_start, profile_start),
        "semantic_rank_profile": (profile_start, score_shape_start),
        "semantic_rank_agreement": (rank_start, projection_start),
        "embedding_projection": (projection_start, interaction_start),
    }
    if include_score_shape:
        groups["semantic_score_shape"] = (score_shape_start, rank_start)
    if include_interactions:
        groups["lexical_semantic_interactions"] = (interaction_start, semantic_end)
    return groups


def transform_policy_features(features: np.ndarray, feature_set: str) -> np.ndarray:
    return _transform_policy_features(features, feature_set)


def _transform_policy_features(
    features: np.ndarray,
    feature_set: str,
    semantic_mean: np.ndarray | None = None,
    semantic_std: np.ndarray | None = None,
) -> np.ndarray:
    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unknown feature set: {feature_set}")

    transformed = np.asarray(features, dtype=float).copy()
    is_vector = transformed.ndim == 1
    if is_vector:
        transformed = transformed.reshape(1, -1)
    if transformed.ndim != 2:
        raise ValueError("Policy features must be a vector or matrix")

    if feature_set == "full":
        return transformed[0] if is_vector else transformed
    semantic_groups = semantic_feature_group_slices(transformed.shape[1])
    semantic_end = _contrast_feature_start(transformed.shape[1])
    if feature_set == "no_query":
        _zero_columns(transformed, QUERY_FEATURE_IDXS)
        _zero_range(transformed, WH_FEATURE_START, WH_FEATURE_END)
    elif feature_set == "no_retrieval":
        _zero_columns(transformed, RETRIEVAL_FEATURE_IDXS)
    elif feature_set == "no_wh":
        _zero_range(transformed, WH_FEATURE_START, WH_FEATURE_END)
    elif feature_set == "retrieval_only":
        _zero_columns(transformed, QUERY_FEATURE_IDXS)
        _zero_range(transformed, WH_FEATURE_START, transformed.shape[1])
    elif feature_set == "no_semantic":
        _zero_range(transformed, SEMANTIC_FEATURE_START, semantic_end)
    elif feature_set == "semantic_only":
        _zero_range(transformed, 0, SEMANTIC_FEATURE_START)
        _zero_range(transformed, semantic_groups.get("embedding_projection", (semantic_end, semantic_end))[0], transformed.shape[1])
    elif feature_set == "no_profile":
        _zero_named_group(transformed, semantic_groups, "semantic_rank_profile")
    elif feature_set == "profile_only":
        _keep_named_group_only(transformed, semantic_groups, "semantic_rank_profile")
    elif feature_set == "no_score_shape":
        _zero_named_group(transformed, semantic_groups, "semantic_score_shape")
    elif feature_set == "score_shape_only":
        _keep_named_group_only(transformed, semantic_groups, "semantic_score_shape")
    elif feature_set == "no_rank_agreement":
        _zero_named_group(transformed, semantic_groups, "semantic_rank_agreement")
    elif feature_set == "rank_agreement_only":
        _keep_named_group_only(transformed, semantic_groups, "semantic_rank_agreement")
    elif feature_set == "no_projection":
        _zero_named_group(transformed, semantic_groups, "embedding_projection")
    elif feature_set == "projection_only":
        _keep_named_group_only(transformed, semantic_groups, "embedding_projection")
    elif feature_set == "no_interactions":
        _zero_named_group(transformed, semantic_groups, "lexical_semantic_interactions")
    elif feature_set == "interactions_only":
        _keep_named_group_only(transformed, semantic_groups, "lexical_semantic_interactions")
    elif feature_set == "semantic_zscore":
        _zscore_range(
            transformed,
            SEMANTIC_FEATURE_START,
            semantic_end,
            semantic_mean,
            semantic_std,
        )
    elif feature_set == "no_contrast":
        _zero_range(transformed, _contrast_feature_start(transformed.shape[1]), transformed.shape[1])
    elif feature_set == "contrast_only":
        _zero_range(transformed, 0, _contrast_feature_start(transformed.shape[1]))
    return transformed[0] if is_vector else transformed


def _zero_columns(features: np.ndarray, indices: list[int]) -> None:
    for idx in indices:
        if idx < features.shape[1]:
            features[:, idx] = 0.0


def _zero_range(features: np.ndarray, start: int, end: int) -> None:
    if start < features.shape[1]:
        features[:, start : min(end, features.shape[1])] = 0.0


def _zero_named_group(features: np.ndarray, groups: dict[str, tuple[int, int]], name: str) -> None:
    start, end = groups.get(name, (features.shape[1], features.shape[1]))
    _zero_range(features, start, end)


def _keep_named_group_only(features: np.ndarray, groups: dict[str, tuple[int, int]], name: str) -> None:
    start, end = groups.get(name, (features.shape[1], features.shape[1]))
    _zero_range(features, 0, start)
    _zero_range(features, end, features.shape[1])


def _contrast_feature_start(width: int) -> int:
    if width >= INTERACTION_FEATURE_START + 6 + RETRIEVAL_CONTRAST_FEATURE_WIDTH:
        return width - RETRIEVAL_CONTRAST_FEATURE_WIDTH
    if width == SEMANTIC_FEATURE_START + RETRIEVAL_CONTRAST_FEATURE_WIDTH:
        return SEMANTIC_FEATURE_START
    return width


def _zscore_range(
    features: np.ndarray,
    start: int,
    end: int,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> None:
    if start >= features.shape[1]:
        return
    end = min(end, features.shape[1])
    values = features[:, start:end]
    fitted_mean = values.mean(axis=0) if mean is None else np.asarray(mean, dtype=float)[: values.shape[1]]
    fitted_std = values.std(axis=0) if std is None else np.asarray(std, dtype=float)[: values.shape[1]]
    fitted_std = np.where(fitted_std <= 1e-12, 1.0, fitted_std)
    features[:, start:end] = (values - fitted_mean) / fitted_std


def run_retrieval_policy_experiment(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 300,
    seed: int = 42,
    k: int = 5,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    knn_k_candidates: list[int] | None = None,
    tuning_folds: int = 5,
    policy_model: str = "knn",
    feature_set: str = "full",
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
    confidence_gate_margin: float | None = None,
) -> dict[str, object]:
    examples = load_hotpotqa(data_path, num_examples=num_examples, seed=seed)
    return run_retrieval_policy_on_examples(
        examples=examples,
        output_dir=output_dir,
        dataset_name="HotpotQA retrieval-action policy",
        output_prefix="retrieval_policy",
        checkpoint_name="hotpot_retrieval_policy.pkl",
        seed=seed,
        k=k,
        embedder_name=embedder_name,
        dense_weight=dense_weight,
        retrieval_call_cost=retrieval_call_cost,
        semantic_features=semantic_features,
        semantic_cache_path=semantic_cache_path,
        knn_k_candidates=knn_k_candidates,
        tuning_folds=tuning_folds,
        policy_model=policy_model,
        feature_set=feature_set,
        semantic_depth=semantic_depth,
        confidence_gate_margin=confidence_gate_margin,
    )


def run_llm_retrieval_policy_experiment(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 40,
    seed: int = 42,
    k: int = 5,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    cache_path: Path | None = None,
    rewrite_provider: RewriteProvider | None = None,
    llm_base_cost: float = 1.0,
    llm_token_cost: float = 0.01,
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    knn_k_candidates: list[int] | None = None,
    tuning_folds: int = 5,
    policy_model: str = "auto",
    feature_set: str = "full",
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
    confidence_gate_margin: float | None = None,
) -> dict[str, object]:
    examples = load_hotpotqa(data_path, num_examples=num_examples, seed=seed)
    return run_retrieval_policy_on_examples(
        examples=examples,
        output_dir=output_dir,
        dataset_name="HotpotQA LLM retrieval-action policy",
        output_prefix="llm_retrieval_policy",
        checkpoint_name="hotpot_llm_retrieval_policy.pkl",
        seed=seed,
        k=k,
        embedder_name=embedder_name,
        dense_weight=dense_weight,
        retrieval_call_cost=retrieval_call_cost,
        semantic_features=semantic_features,
        semantic_cache_path=semantic_cache_path,
        knn_k_candidates=knn_k_candidates,
        tuning_folds=tuning_folds,
        policy_model=policy_model,
        feature_set=feature_set,
        actions=RETRIEVAL_ACTIONS_WITH_LLM,
        action_to_method=LLM_ACTION_TO_METHOD,
        method_order=LLM_METHOD_ORDER,
        llm_cache_path=cache_path,
        rewrite_provider=rewrite_provider,
        llm_base_cost=llm_base_cost,
        llm_token_cost=llm_token_cost,
        semantic_depth=semantic_depth,
        confidence_gate_margin=confidence_gate_margin,
    )


def run_scifact_retrieval_policy_experiment(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 300,
    num_train_examples: int | None = None,
    num_test_examples: int | None = None,
    seed: int = 42,
    k: int = 5,
    pool_size: int = 100,
    full_corpus: bool = False,
    train_split: str = "train",
    test_split: str = "test",
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    knn_k_candidates: list[int] | None = None,
    tuning_folds: int = 5,
    policy_model: str = "knn",
    feature_set: str = "full",
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
    confidence_gate_margin: float | None = None,
    generated_actions: bool = False,
    llm_cache_path: Path | None = None,
    rewrite_provider: RewriteProvider | None = None,
    llm_base_cost: float = 1.0,
    llm_token_cost: float = 0.01,
) -> dict[str, object]:
    return run_beir_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        dataset_key="scifact",
        dataset_name="BEIR SciFact retrieval-action policy",
        output_prefix="scifact_retrieval_policy",
        checkpoint_name="scifact_retrieval_policy.pkl",
        num_examples=num_examples,
        num_train_examples=num_train_examples,
        num_test_examples=num_test_examples,
        seed=seed,
        k=k,
        pool_size=pool_size,
        full_corpus=full_corpus,
        train_split=train_split,
        test_split=test_split,
        embedder_name=embedder_name,
        dense_weight=dense_weight,
        retrieval_call_cost=retrieval_call_cost,
        semantic_features=semantic_features,
        semantic_cache_path=semantic_cache_path,
        knn_k_candidates=knn_k_candidates,
        tuning_folds=tuning_folds,
        policy_model=policy_model,
        feature_set=feature_set,
        semantic_depth=semantic_depth,
        confidence_gate_margin=confidence_gate_margin,
        generated_actions=generated_actions,
        llm_cache_path=llm_cache_path,
        rewrite_provider=rewrite_provider,
        llm_base_cost=llm_base_cost,
        llm_token_cost=llm_token_cost,
    )


def run_nfcorpus_retrieval_policy_experiment(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 300,
    num_train_examples: int | None = None,
    num_test_examples: int | None = None,
    seed: int = 42,
    k: int = 5,
    pool_size: int = 100,
    full_corpus: bool = False,
    train_split: str = "train",
    test_split: str = "test",
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    knn_k_candidates: list[int] | None = None,
    tuning_folds: int = 5,
    policy_model: str = "knn",
    feature_set: str = "full",
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
    confidence_gate_margin: float | None = None,
    generated_actions: bool = False,
    llm_cache_path: Path | None = None,
    rewrite_provider: RewriteProvider | None = None,
    llm_base_cost: float = 1.0,
    llm_token_cost: float = 0.01,
) -> dict[str, object]:
    return run_beir_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        dataset_key="nfcorpus",
        dataset_name="BEIR NFCorpus retrieval-action policy",
        output_prefix="nfcorpus_retrieval_policy",
        checkpoint_name="nfcorpus_retrieval_policy.pkl",
        num_examples=num_examples,
        num_train_examples=num_train_examples,
        num_test_examples=num_test_examples,
        seed=seed,
        k=k,
        pool_size=pool_size,
        full_corpus=full_corpus,
        train_split=train_split,
        test_split=test_split,
        embedder_name=embedder_name,
        dense_weight=dense_weight,
        retrieval_call_cost=retrieval_call_cost,
        semantic_features=semantic_features,
        semantic_cache_path=semantic_cache_path,
        knn_k_candidates=knn_k_candidates,
        tuning_folds=tuning_folds,
        policy_model=policy_model,
        feature_set=feature_set,
        semantic_depth=semantic_depth,
        confidence_gate_margin=confidence_gate_margin,
        generated_actions=generated_actions,
        llm_cache_path=llm_cache_path,
        rewrite_provider=rewrite_provider,
        llm_base_cost=llm_base_cost,
        llm_token_cost=llm_token_cost,
    )


def run_beir_retrieval_policy_experiment(
    data_path: Path,
    output_dir: Path,
    dataset_key: str,
    dataset_name: str,
    output_prefix: str,
    checkpoint_name: str,
    num_examples: int = 300,
    num_train_examples: int | None = None,
    num_test_examples: int | None = None,
    seed: int = 42,
    k: int = 5,
    pool_size: int = 100,
    full_corpus: bool = False,
    train_split: str = "train",
    test_split: str = "test",
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    knn_k_candidates: list[int] | None = None,
    tuning_folds: int = 5,
    policy_model: str = "knn",
    feature_set: str = "full",
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
    confidence_gate_margin: float | None = None,
    generated_actions: bool = False,
    llm_cache_path: Path | None = None,
    rewrite_provider: RewriteProvider | None = None,
    llm_base_cost: float = 1.0,
    llm_token_cost: float = 0.01,
) -> dict[str, object]:
    train_count = num_train_examples if num_train_examples is not None else num_examples
    test_count = num_test_examples if num_test_examples is not None else num_examples
    train_examples = load_beir_dataset(
        data_path,
        num_examples=train_count,
        seed=seed,
        split=train_split,
        pool_size=pool_size,
        full_corpus=full_corpus,
        qtype=f"beir-{dataset_key}",
    )
    test_examples = load_beir_dataset(
        data_path,
        num_examples=test_count,
        seed=seed,
        split=test_split,
        pool_size=pool_size,
        full_corpus=full_corpus,
        qtype=f"beir-{dataset_key}",
    )
    actions = RETRIEVAL_ACTIONS_WITH_LLM if generated_actions else RETRIEVAL_ACTIONS
    action_to_method = LLM_ACTION_TO_METHOD if generated_actions else ACTION_TO_METHOD
    method_order = LLM_METHOD_ORDER if generated_actions else METHOD_ORDER
    if generated_actions:
        dataset_name = f"{dataset_name} with generated query actions"
        output_prefix = f"{output_prefix}_generated"
        checkpoint_name = checkpoint_name.replace(".pkl", "_generated.pkl")
    metadata = run_retrieval_policy_on_examples(
        examples=train_examples,
        test_examples=test_examples,
        output_dir=output_dir,
        dataset_name=dataset_name,
        output_prefix=output_prefix,
        checkpoint_name=checkpoint_name,
        seed=seed,
        k=k,
        embedder_name=embedder_name,
        dense_weight=dense_weight,
        retrieval_call_cost=retrieval_call_cost,
        semantic_features=semantic_features,
        semantic_cache_path=semantic_cache_path,
        knn_k_candidates=knn_k_candidates,
        tuning_folds=tuning_folds,
        policy_model=policy_model,
        feature_set=feature_set,
        actions=actions,
        action_to_method=action_to_method,
        method_order=method_order,
        llm_cache_path=llm_cache_path,
        rewrite_provider=rewrite_provider,
        llm_base_cost=llm_base_cost,
        llm_token_cost=llm_token_cost,
        semantic_depth=semantic_depth,
        confidence_gate_margin=confidence_gate_margin,
        extra_metadata={
            "dataset_key": dataset_key,
            "pool_size": pool_size,
            "full_corpus": full_corpus,
            "corpus_size": len(train_examples[0].passages) if full_corpus and train_examples else None,
            "train_split": train_split,
            "test_split": test_split,
            "generated_actions": generated_actions,
        },
    )
    metadata["dataset_key"] = dataset_key
    metadata["pool_size"] = pool_size
    metadata["full_corpus"] = full_corpus
    metadata["corpus_size"] = len(train_examples[0].passages) if full_corpus and train_examples else None
    metadata["train_split"] = train_split
    metadata["test_split"] = test_split
    metadata["generated_actions"] = generated_actions
    Path(metadata["outputs"]["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_nq_retrieval_policy_experiment(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 500,
    seed: int = 42,
    k: int = 5,
    pool_size: int = 50,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    knn_k_candidates: list[int] | None = None,
    tuning_folds: int = 5,
    policy_model: str = "knn",
    feature_set: str = "full",
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
    confidence_gate_margin: float | None = None,
) -> dict[str, object]:
    examples = load_natural_questions(data_path, num_examples=num_examples, seed=seed, pool_size=pool_size)
    metadata = run_retrieval_policy_on_examples(
        examples=examples,
        output_dir=output_dir,
        dataset_name="Natural Questions retrieval-action policy",
        output_prefix="nq_retrieval_policy",
        checkpoint_name="nq_retrieval_policy.pkl",
        seed=seed,
        k=k,
        embedder_name=embedder_name,
        dense_weight=dense_weight,
        retrieval_call_cost=retrieval_call_cost,
        semantic_features=semantic_features,
        semantic_cache_path=semantic_cache_path,
        knn_k_candidates=knn_k_candidates,
        tuning_folds=tuning_folds,
        policy_model=policy_model,
        feature_set=feature_set,
        semantic_depth=semantic_depth,
        confidence_gate_margin=confidence_gate_margin,
    )
    metadata["pool_size"] = pool_size
    Path(metadata["outputs"]["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_retrieval_policy_on_examples(
    examples: list[QAExample],
    output_dir: Path,
    dataset_name: str,
    output_prefix: str,
    checkpoint_name: str,
    test_examples: list[QAExample] | None = None,
    seed: int = 42,
    k: int = 5,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    semantic_embedder: SemanticEmbedder | None = None,
    knn_k_candidates: list[int] | None = None,
    tuning_folds: int = 5,
    policy_model: str = "knn",
    feature_set: str = "full",
    actions: list[str] | None = None,
    action_to_method: dict[str, str] | None = None,
    method_order: list[str] | None = None,
    extra_metadata: dict[str, object] | None = None,
    llm_cache_path: Path | None = None,
    rewrite_provider: RewriteProvider | None = None,
    llm_base_cost: float = 1.0,
    llm_token_cost: float = 0.01,
    retrieval_contrast_features: bool = False,
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
    confidence_gate_margin: float | None = None,
) -> dict[str, object]:
    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if confidence_gate_margin is not None and confidence_gate_margin < 0:
        raise ValueError("confidence_gate_margin must be non-negative")
    semantic_depth = _validate_semantic_depth(semantic_depth)
    train, test = (examples, test_examples) if test_examples is not None else split_examples(examples)
    actions = list(actions or RETRIEVAL_ACTIONS)
    if action_to_method is None:
        action_to_method = {action: ACTION_TO_METHOD[action] for action in actions if action in ACTION_TO_METHOD}
    else:
        unknown_mapped_actions = [action for action in action_to_method if action not in actions]
        if unknown_mapped_actions:
            raise ValueError(
                "action_to_method contains action(s) not present in actions: "
                + ", ".join(sorted(unknown_mapped_actions))
            )
        action_to_method = {action: method for action, method in action_to_method.items() if action in actions}
    method_order = list(method_order or METHOD_ORDER)
    if confidence_gate_margin is not None and CONFIDENCE_GATED_METHOD not in method_order:
        oracle_index = method_order.index("Oracle retrieval action") if "Oracle retrieval action" in method_order else len(method_order)
        method_order.insert(oracle_index, CONFIDENCE_GATED_METHOD)
    embedder = FakeDenseEmbedder() if embedder_name == "fake" else load_sentence_transformer(embedder_name)
    semantic_embedder = semantic_embedder or _load_semantic_embedder(semantic_features, output_dir, semantic_cache_path)
    if semantic_embedder is not None:
        _prewarm_semantic_embeddings([*train, *test], semantic_embedder, semantic_depth)
    knn_k_candidates = knn_k_candidates or [1, 3, 5, 7, 9, 11, 15, 21]
    llm_cache, rewrite_provider = _llm_dependencies(actions, output_dir, llm_cache_path, rewrite_provider)
    extra_metadata = extra_metadata or {}
    retriever_cache: dict[tuple[str, ...], tuple[BM25Retriever, DenseRetriever]] = {}

    train_evals = [
        evaluate_retrieval_actions(
            ex,
            embedder,
            k,
            dense_weight,
            retrieval_call_cost,
            semantic_embedder,
            actions=actions,
            llm_cache=llm_cache,
            rewrite_provider=rewrite_provider,
            llm_base_cost=llm_base_cost,
            llm_token_cost=llm_token_cost,
            retrieval_contrast_features=retrieval_contrast_features,
            retriever_cache=retriever_cache,
            semantic_depth=semantic_depth,
        )
        for ex in tqdm(train, desc="train retrieval actions")
    ]
    raw_features = np.vstack([row["features"] for row in train_evals])
    feature_transform = fit_policy_feature_transform(raw_features, feature_set)
    features = feature_transform.transform(raw_features)
    feature_width = int(features.shape[1])
    rewards = {action: [row["actions"][action]["reward"] for row in train_evals] for action in actions}
    policy, selected_policy_model, validation_policy_scores = select_policy_model(
        features=features,
        rewards=rewards,
        policy_model=policy_model,
        knn_k_candidates=knn_k_candidates,
        folds=tuning_folds,
        seed=seed,
        actions=actions,
    )
    best_fixed_action = max(
        actions,
        key=lambda action: (float(np.mean(rewards[action])), -actions.index(action)),
    )
    policy.fit(features, rewards)

    rows: list[dict[str, object]] = []
    for split_name, split_examples_ in [("train", train), ("test", test)]:
        for ex in tqdm(split_examples_, desc=f"{split_name} retrieval policy evaluation"):
            action_eval = evaluate_retrieval_actions(
                ex,
                embedder,
                k,
                dense_weight,
                retrieval_call_cost,
                semantic_embedder,
                actions=actions,
                llm_cache=llm_cache,
                rewrite_provider=rewrite_provider,
                llm_base_cost=llm_base_cost,
                llm_token_cost=llm_token_cost,
                retrieval_contrast_features=retrieval_contrast_features,
                retriever_cache=retriever_cache,
                semantic_depth=semantic_depth,
            )
            transformed_feature = feature_transform.transform(action_eval["features"])
            difficulty_features = _difficulty_feature_row(action_eval, actions, k)
            policy_scores = policy.predict_scores(transformed_feature)
            selected, predicted_margin = prediction_margin(policy_scores, actions)
            for action, method_name in action_to_method.items():
                rows.append(_method_row(split_name, method_name, action, ex, action_eval, difficulty_features))
            rows.append(
                _method_row(split_name, "Train-best retrieval action", best_fixed_action, ex, action_eval, difficulty_features)
            )
            heuristic_action = heuristic_retrieval_action(action_eval["features"], actions)
            rows.append(
                _method_row(split_name, "Heuristic retrieval router", heuristic_action, ex, action_eval, difficulty_features)
            )
            rows.append(
                _method_row(
                    split_name,
                    "Selective retrieval policy",
                    selected,
                    ex,
                    action_eval,
                    difficulty_features,
                    extra={
                        "policy_action": selected,
                        "fallback_action": None,
                        "fallback_used": False,
                        "predicted_action_margin": predicted_margin,
                        "policy_action_score": policy_scores[selected],
                        "runner_up_action_score": _runner_up_score(policy_scores, actions),
                    },
                )
            )
            if confidence_gate_margin is not None:
                gated = confidence_gated_action(policy_scores, actions, best_fixed_action, confidence_gate_margin)
                rows.append(
                    _method_row(
                        split_name,
                        CONFIDENCE_GATED_METHOD,
                        gated.action,
                        ex,
                        action_eval,
                        difficulty_features,
                        extra={
                            "policy_action": gated.policy_action,
                            "fallback_action": gated.fallback_action,
                            "fallback_used": gated.fallback_used,
                            "predicted_action_margin": gated.predicted_margin,
                            "policy_action_score": gated.policy_score,
                            "runner_up_action_score": gated.runner_up_score,
                        },
                    )
                )
            oracle_action = max(actions, key=lambda action: action_eval["actions"][action]["reward"])
            rows.append(
                _method_row(split_name, "Oracle retrieval action", oracle_action, ex, action_eval, difficulty_features)
            )

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    checkpoints_dir = output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    detailed_csv = results_dir / f"{output_prefix}_detailed.csv"
    summary_csv = results_dir / f"{output_prefix}_summary.csv"
    summary_json = results_dir / f"{output_prefix}_summary.json"
    table_tex = results_dir / f"{output_prefix}_table.tex"
    metadata_json = results_dir / f"{output_prefix}_metadata.json"
    checkpoint_path = checkpoints_dir / checkpoint_name
    action_distribution_csv = results_dir / f"{output_prefix}_action_distribution.csv"

    df.to_csv(detailed_csv, index=False)
    summary = summarize_retrieval_policy(df, method_order=method_order)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    table_tex.write_text(_latex_table(summary), encoding="utf-8")
    policy_action_distribution = _policy_action_distribution(df)
    _policy_action_distribution_frame(policy_action_distribution).to_csv(action_distribution_csv, index=False)
    save_checkpoint(
        checkpoint_path,
        policy,
        {
            "dataset": dataset_name,
            "num_examples": len(examples),
            "train_examples": len(train),
            "test_examples": len(test),
            "seed": seed,
            "k": k,
            "actions": actions,
            "best_fixed_action": best_fixed_action,
            "policy_model": policy_model,
            "feature_set": feature_set,
            "feature_width": feature_width,
            "selected_policy_model": selected_policy_model,
            "selected_knn_k": _selected_knn_k(selected_policy_model),
            "validation_policy_scores": validation_policy_scores,
            "tuning_folds": tuning_folds,
            "embedder": embedder_name,
            "semantic_features": semantic_features,
            "semantic_depth": semantic_depth,
            "semantic_cache_path": str(semantic_cache_path or output_dir / "cache" / "vertex_embeddings.jsonl")
            if semantic_embedder is not None
            else None,
            "dense_weight": dense_weight,
            "retrieval_call_cost": retrieval_call_cost,
            "llm_cache_path": str(llm_cache.path) if llm_cache is not None else None,
            "llm_base_cost": llm_base_cost if llm_cache is not None else None,
            "llm_token_cost": llm_token_cost if llm_cache is not None else None,
            "retrieval_contrast_features": retrieval_contrast_features,
            "confidence_gate_margin": confidence_gate_margin,
            "policy_action_distribution": policy_action_distribution,
            **extra_metadata,
        },
    )

    metadata = {
        "num_examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "seed": seed,
        "k": k,
        "best_fixed_action": best_fixed_action,
        "policy_model": policy_model,
        "feature_set": feature_set,
        "feature_width": feature_width,
        "selected_policy_model": selected_policy_model,
        "selected_knn_k": _selected_knn_k(selected_policy_model),
        "validation_policy_scores": validation_policy_scores,
        "tuning_folds": tuning_folds,
        "embedder": embedder_name,
        "semantic_features": semantic_features,
        "semantic_depth": semantic_depth,
        "semantic_cache_path": str(semantic_cache_path or output_dir / "cache" / "vertex_embeddings.jsonl")
        if semantic_embedder is not None
        else None,
        "dense_weight": dense_weight,
        "retrieval_call_cost": retrieval_call_cost,
        "llm_cache_path": str(llm_cache.path) if llm_cache is not None else None,
        "llm_base_cost": llm_base_cost if llm_cache is not None else None,
        "llm_token_cost": llm_token_cost if llm_cache is not None else None,
        "retrieval_contrast_features": retrieval_contrast_features,
        "confidence_gate_margin": confidence_gate_margin,
        "policy_action_distribution": policy_action_distribution,
        **extra_metadata,
        "outputs": {
            "detailed_csv": str(detailed_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "table_tex": str(table_tex),
            "checkpoint": str(checkpoint_path),
            "metadata_json": str(metadata_json),
            "action_distribution_csv": str(action_distribution_csv),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def select_policy_model(
    features: np.ndarray,
    rewards: dict[str, list[float]],
    policy_model: str,
    knn_k_candidates: list[int],
    folds: int,
    seed: int,
    actions: list[str] | None = None,
    auto_candidate_models: list[str] | None = None,
) -> tuple[object, str, dict[str, float]]:
    return _select_policy_model(
        features,
        rewards,
        policy_model,
        knn_k_candidates,
        folds,
        seed,
        actions or RETRIEVAL_ACTIONS,
        auto_candidate_models,
    )


def _select_policy_model(
    features: np.ndarray,
    rewards: dict[str, list[float]],
    policy_model: str,
    knn_k_candidates: list[int],
    folds: int,
    seed: int,
    actions: list[str],
    auto_candidate_models: list[str] | None = None,
) -> tuple[object, str, dict[str, float]]:
    n = features.shape[0]
    folds = max(2, min(folds, n))
    fold_ids = np.arange(n) % folds
    scores: dict[str, float] = {}
    candidates = _policy_candidates(policy_model, knn_k_candidates, seed, actions, auto_candidate_models)
    for name, factory in candidates:
        selected_rewards: list[float] = []
        for fold in range(folds):
            train_mask = fold_ids != fold
            valid_idx = np.where(fold_ids == fold)[0]
            policy = factory()
            policy.fit(
                features[train_mask],
                {action: list(np.asarray(rewards[action], dtype=float)[train_mask]) for action in actions},
            )
            for idx in valid_idx:
                action = policy.predict(features[idx])
                selected_rewards.append(float(rewards[action][idx]))
        scores[name] = float(np.mean(selected_rewards))
    selected_name, selected_factory = max(candidates, key=lambda item: (scores[item[0]], -candidates.index(item)))
    return selected_factory(), selected_name, scores


def _policy_candidates(
    policy_model: str,
    knn_k_candidates: list[int],
    seed: int,
    actions: list[str],
    auto_candidate_models: list[str] | None = None,
) -> list[tuple[str, object]]:
    candidates: list[tuple[str, object]] = []
    auto_families = _normalize_auto_candidate_models(auto_candidate_models)
    if policy_model == "knn" or (policy_model == "auto" and "knn" in auto_families):
        candidates.extend((f"knn_k={k}", lambda k=k: KnnDirectMethodBandit(actions=actions, k=k)) for k in knn_k_candidates)
    if policy_model == "ridge" or (policy_model == "auto" and "ridge" in auto_families):
        candidates.append(("ridge_l2=1.0", lambda: DirectMethodBandit(actions=actions, l2=1.0)))
    if policy_model == "margin_ridge":
        candidates.append(("margin_ridge_l2=1.0", lambda: MarginWeightedDirectMethodBandit(actions=actions, l2=1.0)))
    if policy_model == "ridge_sweep":
        candidates.extend((f"ridge_l2={l2}", lambda l2=l2: DirectMethodBandit(actions=actions, l2=l2)) for l2 in RIDGE_L2_CANDIDATES)
    if policy_model == "extra_trees" or (policy_model == "auto" and "extra_trees" in auto_families):
        candidates.append(("extra_trees", lambda: _tree_bandit("extra_trees", seed, actions)))
    if policy_model == "random_forest" or (policy_model == "auto" and "random_forest" in auto_families):
        candidates.append(("random_forest", lambda: _tree_bandit("random_forest", seed, actions)))
    if policy_model == "mlp" or (policy_model == "auto" and "mlp" in auto_families):
        candidates.append(("mlp", lambda: _neural_bandit(seed, actions)))
    if not candidates:
        raise ValueError(f"Unknown policy model: {policy_model}")
    return candidates


def _normalize_auto_candidate_models(auto_candidate_models: list[str] | None = None) -> list[str]:
    candidates = auto_candidate_models or AUTO_POLICY_MODEL_CHOICES
    unknown = sorted(set(candidates) - set(AUTO_POLICY_MODEL_CHOICES))
    if unknown:
        raise ValueError(f"Unknown auto candidate model(s): {', '.join(unknown)}")
    return list(dict.fromkeys(candidates))


def _tree_bandit(model_type: str, seed: int, actions: list[str]) -> SklearnRegressorBandit:
    if model_type in {"extra_trees", "random_forest"}:
        return SklearnRegressorBandit(
            actions=actions,
            estimator_factory=TreeRegressorFactory(model_type=model_type, seed=seed),
        )
    raise ValueError(f"Unknown tree model: {model_type}")


def _neural_bandit(seed: int, actions: list[str]) -> SklearnRegressorBandit:
    return SklearnRegressorBandit(
        actions=actions,
        estimator_factory=NeuralRegressorFactory(seed=seed),
    )


def _selected_knn_k(selected_policy_model: str) -> int | None:
    if not selected_policy_model.startswith("knn_k="):
        return None
    return int(selected_policy_model.split("=", 1)[1])


def _policy_action_distribution(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    policy_rows = df[df["method"] == "Selective retrieval policy"]
    distribution: dict[str, dict[str, object]] = {}
    for split_name, split_rows in policy_rows.groupby("split", sort=True):
        counts = split_rows["action"].value_counts().sort_index()
        total = int(counts.sum())
        distribution[str(split_name)] = {
            "total": total,
            "actions": {str(action): int(count) for action, count in counts.items()},
            "proportions": {str(action): float(count / total) if total else 0.0 for action, count in counts.items()},
        }
    return distribution


def _policy_action_distribution_frame(distribution: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for split_name, split_distribution in distribution.items():
        total = int(split_distribution["total"])
        actions = split_distribution["actions"]
        for action, count in actions.items():
            rows.append(
                {
                    "split": split_name,
                    "action": action,
                    "count": int(count),
                    "proportion": float(count / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=["split", "action", "count", "proportion"])


def _runner_up_score(scores: dict[str, float], actions: list[str]) -> float:
    ordered = sorted(
        actions,
        key=lambda action: (scores[action], -actions.index(action)),
        reverse=True,
    )
    runner_up = ordered[1] if len(ordered) > 1 else ordered[0]
    return float(scores[runner_up])


def evaluate_retrieval_actions(
    ex: QAExample,
    embedder: object,
    k: int,
    dense_weight: float,
    retrieval_call_cost: float,
    semantic_embedder: SemanticEmbedder | None = None,
    actions: list[str] | None = None,
    llm_cache: GeminiCache | None = None,
    rewrite_provider: RewriteProvider | None = None,
    llm_base_cost: float = 1.0,
    llm_token_cost: float = 0.01,
    retrieval_contrast_features: bool = False,
    retriever_cache: dict[tuple[str, ...], tuple[BM25Retriever, DenseRetriever]] | None = None,
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
) -> dict[str, object]:
    semantic_depth = _validate_semantic_depth(semantic_depth)
    actions = actions or RETRIEVAL_ACTIONS
    bm25, dense = _retrievers_for_example(ex, embedder, retriever_cache)
    initial_depth = max(k, semantic_depth) if semantic_embedder is not None else k
    initial = bm25.search(ex.question, k=initial_depth)
    keep = rewrite(ex.question, "keep")
    keyword = rewrite(ex.question, "keyword_compress")
    output = {}
    if "bm25_keep" in actions:
        output["bm25_keep"] = _evaluate_action(
            ex, "bm25_keep", keep.joined_query, bm25.search(keep.joined_query, k=k), keep, 1, k, 0.0
        )
    if "bm25_keyword" in actions:
        output["bm25_keyword"] = _evaluate_action(
            ex,
            "bm25_keyword",
            keyword.joined_query,
            bm25.search(keyword.joined_query, k=k),
            keyword,
            1,
            k,
            0.0,
        )
    if "dense_keep" in actions:
        output["dense_keep"] = _evaluate_action(
            ex,
            "dense_keep",
            keep.joined_query,
            dense.search(keep.joined_query, k=k),
            keep,
            1,
            k,
            0.0,
        )
    if "dense_keyword" in actions:
        output["dense_keyword"] = _evaluate_action(
            ex,
            "dense_keyword",
            keyword.joined_query,
            dense.search(keyword.joined_query, k=k),
            keyword,
            1,
            k,
            0.0,
        )
    if "hybrid_keep" in actions:
        output["hybrid_keep"] = _evaluate_action(
            ex,
            "hybrid_keep",
            keep.joined_query,
            hybrid_merge(bm25.search(keep.joined_query, k=k), dense.search(keep.joined_query, k=k), k, dense_weight),
            keep,
            2,
            k,
            retrieval_call_cost,
        )
    if "hybrid_keyword" in actions:
        output["hybrid_keyword"] = _evaluate_action(
            ex,
            "hybrid_keyword",
            keyword.joined_query,
            hybrid_merge(bm25.search(keyword.joined_query, k=k), dense.search(keyword.joined_query, k=k), k, dense_weight),
            keyword,
            2,
            k,
            retrieval_call_cost,
        )
    if "bm25_llm_rewrite" in actions:
        output["bm25_llm_rewrite"] = _evaluate_generated_action(
            ex,
            "bm25_llm_rewrite",
            "rewrite",
            bm25,
            dense,
            "bm25",
            k,
            dense_weight,
            retrieval_call_cost,
            llm_cache,
            rewrite_provider,
            llm_base_cost,
            llm_token_cost,
        )
    if "bm25_llm_decompose" in actions:
        output["bm25_llm_decompose"] = _evaluate_generated_action(
            ex,
            "bm25_llm_decompose",
            "decompose",
            bm25,
            dense,
            "bm25",
            k,
            dense_weight,
            retrieval_call_cost,
            llm_cache,
            rewrite_provider,
            llm_base_cost,
            llm_token_cost,
        )
    if "hybrid_llm_decompose" in actions:
        output["hybrid_llm_decompose"] = _evaluate_generated_action(
            ex,
            "hybrid_llm_decompose",
            "decompose",
            bm25,
            dense,
            "hybrid",
            k,
            dense_weight,
            retrieval_call_cost,
            llm_cache,
            rewrite_provider,
            llm_base_cost,
            llm_token_cost,
        )
    if "bm25_hyde" in actions:
        output["bm25_hyde"] = _evaluate_generated_action(
            ex,
            "bm25_hyde",
            "hyde",
            bm25,
            dense,
            "bm25",
            k,
            dense_weight,
            retrieval_call_cost,
            llm_cache,
            rewrite_provider,
            llm_base_cost,
            llm_token_cost,
        )
    if "dense_hyde" in actions:
        output["dense_hyde"] = _evaluate_generated_action(
            ex,
            "dense_hyde",
            "hyde",
            bm25,
            dense,
            "dense",
            k,
            dense_weight,
            retrieval_call_cost,
            llm_cache,
            rewrite_provider,
            llm_base_cost,
            llm_token_cost,
        )
    if "hybrid_hyde" in actions:
        output["hybrid_hyde"] = _evaluate_generated_action(
            ex,
            "hybrid_hyde",
            "hyde",
            bm25,
            dense,
            "hybrid",
            k,
            dense_weight,
            retrieval_call_cost,
            llm_cache,
            rewrite_provider,
            llm_base_cost,
            llm_token_cost,
        )
    if "bm25_multi_query" in actions:
        output["bm25_multi_query"] = _evaluate_generated_action(
            ex,
            "bm25_multi_query",
            "multi_query",
            bm25,
            dense,
            "bm25",
            k,
            dense_weight,
            retrieval_call_cost,
            llm_cache,
            rewrite_provider,
            llm_base_cost,
            llm_token_cost,
        )
    if "hybrid_multi_query" in actions:
        output["hybrid_multi_query"] = _evaluate_generated_action(
            ex,
            "hybrid_multi_query",
            "multi_query",
            bm25,
            dense,
            "hybrid",
            k,
            dense_weight,
            retrieval_call_cost,
            llm_cache,
            rewrite_provider,
            llm_base_cost,
            llm_token_cost,
        )
    semantic = (
        _semantic_state_features(ex, initial, semantic_embedder, semantic_depth=semantic_depth)
        if semantic_embedder is not None
        else None
    )
    features = state_features(ex.question, initial, semantic)
    if retrieval_contrast_features:
        features = np.concatenate([features, np.asarray(_retrieval_contrast_features(output, k), dtype=float)])
    return {"features": features, "actions": output}


def _retrieval_contrast_features(action_outputs: dict[str, dict[str, object]], k: int) -> list[float]:
    bm25_keep = _top_doc_ids(action_outputs, "bm25_keep")
    dense_keep = _top_doc_ids(action_outputs, "dense_keep")
    hybrid_keep = _top_doc_ids(action_outputs, "hybrid_keep")
    bm25_keyword = _top_doc_ids(action_outputs, "bm25_keyword")
    dense_keyword = _top_doc_ids(action_outputs, "dense_keyword")
    hybrid_keyword = _top_doc_ids(action_outputs, "hybrid_keyword")
    return [
        _overlap_rate(bm25_keep, dense_keep, k),
        _overlap_rate(bm25_keep, hybrid_keep, k),
        _overlap_rate(dense_keep, hybrid_keep, k),
        _overlap_rate(bm25_keyword, dense_keyword, k),
        _overlap_rate(bm25_keyword, hybrid_keyword, k),
        _overlap_rate(dense_keyword, hybrid_keyword, k),
        _new_doc_rate(dense_keep, bm25_keep, k),
        _new_doc_rate(hybrid_keep, bm25_keep, k),
    ]


def _top_doc_ids(action_outputs: dict[str, dict[str, object]], action: str) -> list[str]:
    top_docs = str(action_outputs.get(action, {}).get("top_docs", ""))
    return [doc_id.strip() for doc_id in top_docs.split("|") if doc_id.strip()]


def _overlap_rate(left: list[str], right: list[str], k: int) -> float:
    return len(set(left) & set(right)) / k if k > 0 else 0.0


def _new_doc_rate(candidate: list[str], baseline: list[str], k: int) -> float:
    return len(set(candidate) - set(baseline)) / k if k > 0 else 0.0


def _retrievers_for_example(
    ex: QAExample,
    embedder: object,
    retriever_cache: dict[tuple[str, ...], tuple[BM25Retriever, DenseRetriever]] | None,
) -> tuple[BM25Retriever, DenseRetriever]:
    if retriever_cache is None:
        return BM25Retriever(ex.passages), DenseRetriever(ex.passages, embedder)
    key = tuple(p.doc_id for p in ex.passages)
    if key not in retriever_cache:
        retriever_cache[key] = (BM25Retriever(ex.passages), DenseRetriever(ex.passages, embedder))
    return retriever_cache[key]


def _llm_dependencies(
    actions: list[str],
    output_dir: Path,
    cache_path: Path | None,
    rewrite_provider: RewriteProvider | None,
) -> tuple[GeminiCache | None, RewriteProvider | None]:
    if not any(action in LLM_RETRIEVAL_ACTIONS for action in actions):
        return None, None
    cache = GeminiCache(cache_path or output_dir / "cache" / "gemini_rewrites.jsonl")
    return cache, rewrite_provider or VertexGeminiRewriter(Path.cwd()).rewrite


def _evaluate_generated_action(
    ex: QAExample,
    action: str,
    mode: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    retriever_kind: str,
    k: int,
    dense_weight: float,
    retrieval_call_cost: float,
    llm_cache: GeminiCache | None,
    rewrite_provider: RewriteProvider | None,
    llm_base_cost: float,
    llm_token_cost: float,
) -> dict[str, object]:
    if llm_cache is None or rewrite_provider is None:
        raise ValueError("Generated retrieval actions require a Gemini cache and rewrite provider")
    queries = _cached_llm_queries(ex, mode, llm_cache, rewrite_provider)
    merged, retrieval_calls = _generated_retrieval_results(queries, bm25, dense, retriever_kind, k, dense_weight)
    return _evaluate_action(
        ex,
        action,
        " || ".join(queries),
        merged,
        None,
        retrieval_calls,
        k,
        retrieval_call_cost,
        fixed_rewrite_cost=_llm_rewrite_cost(queries, llm_base_cost, llm_token_cost),
    )


def _generated_retrieval_results(
    queries: list[str],
    bm25: BM25Retriever,
    dense: DenseRetriever,
    retriever_kind: str,
    k: int,
    dense_weight: float,
) -> tuple[list[RetrievalResult], int]:
    if retriever_kind == "bm25":
        return _merge_results([bm25.search(query, k=k) for query in queries], k=k), len(queries)
    if retriever_kind == "dense":
        return _merge_results([dense.search(query, k=k) for query in queries], k=k), len(queries)
    if retriever_kind == "hybrid":
        per_query = [
            hybrid_merge(bm25.search(query, k=k), dense.search(query, k=k), k, dense_weight)
            for query in queries
        ]
        return _merge_results(per_query, k=k), 2 * len(queries)
    raise ValueError(f"Unknown generated retriever kind: {retriever_kind}")


def _cached_llm_queries(
    ex: QAExample,
    mode: str,
    cache: GeminiCache,
    rewrite_provider: RewriteProvider,
) -> list[str]:
    cached = cache.get(ex.qid, mode)
    if cached is not None:
        return _clean_llm_queries(cached, fallback=ex.question, mode=mode)
    queries = _clean_llm_queries(rewrite_provider(ex.question, mode), fallback=ex.question, mode=mode)
    cache.set(ex.qid, mode, queries)
    return queries


def _clean_llm_queries(queries: list[str], fallback: str, mode: str) -> list[str]:
    cleaned = [query.strip() for query in queries if query.strip()]
    if not cleaned:
        return [fallback.strip()]
    if mode in {"rewrite", "hyde"}:
        return cleaned[:1]
    if mode == "multi_query":
        return cleaned[:3]
    return cleaned[:2]


def _llm_rewrite_cost(queries: list[str], base_cost: float, token_cost: float) -> float:
    return base_cost + token_cost * sum(len(content_tokens(query)) for query in queries)


def _load_semantic_embedder(
    semantic_features: str,
    output_dir: Path,
    semantic_cache_path: Path | None,
) -> SemanticEmbedder | None:
    if semantic_features == "none":
        return None
    if semantic_features == "vertex":
        return VertexTextEmbeddingProvider(
            project_root=Path.cwd(),
            cache_path=semantic_cache_path or output_dir / "cache" / "vertex_embeddings.jsonl",
        )
    raise ValueError(f"Unknown semantic feature provider: {semantic_features}")


def _prewarm_semantic_embeddings(examples: list[QAExample], embedder: SemanticEmbedder, k: int) -> None:
    query_texts: list[str] = []
    passage_texts: list[str] = []
    for ex in tqdm(examples, desc="prewarm semantic embeddings"):
        query_texts.append(ex.question)
        bm25 = BM25Retriever(ex.passages)
        top_docs = {result.doc_id for result in bm25.search(ex.question, k=k)}
        passage_texts.extend(p.text for p in ex.passages if p.doc_id in top_docs)
    embedder.embed_texts(query_texts, task_type="RETRIEVAL_QUERY")
    embedder.embed_texts(passage_texts, task_type="RETRIEVAL_DOCUMENT")


def _semantic_state_features(
    ex: QAExample,
    initial_results: list[RetrievalResult],
    embedder: SemanticEmbedder,
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
) -> list[float]:
    semantic_depth = _validate_semantic_depth(semantic_depth)
    passage_by_id = {passage.doc_id: passage for passage in ex.passages}
    top_passages = [passage_by_id[result.doc_id] for result in initial_results[:semantic_depth] if result.doc_id in passage_by_id]
    if not top_passages:
        return [0.0] * _semantic_feature_width(semantic_depth)

    query_embedding = embedder.embed_text(ex.question, task_type="RETRIEVAL_QUERY")
    passage_embeddings = embedder.embed_texts([p.text for p in top_passages], task_type="RETRIEVAL_DOCUMENT")
    similarities = np.asarray([float(query_embedding @ passage_embedding) for passage_embedding in passage_embeddings])
    ranked = np.sort(similarities)[::-1]
    top1 = float(ranked[0])
    top2 = float(ranked[1]) if len(ranked) > 1 else top1
    passage_centroid = np.mean(np.vstack(passage_embeddings), axis=0)
    return [
        top1,
        float(np.mean(similarities)),
        float(np.max(similarities)),
        top1 - top2,
        float(np.mean(similarities > 0.0)),
        float(np.std(similarities)),
        float(np.min(similarities)),
        float(np.median(similarities)),
        float(ranked[0] - ranked[-1]),
        *_semantic_rank_profile(similarities, k=semantic_depth),
        *_semantic_score_shape(similarities),
        *_semantic_rank_agreement(similarities, k=semantic_depth),
        *_embedding_projection_features(query_embedding, passage_centroid),
    ]


def _validate_semantic_depth(semantic_depth: int) -> int:
    depth = int(semantic_depth)
    if depth < SEMANTIC_DEPTH_MIN or depth > SEMANTIC_DEPTH_MAX:
        raise ValueError(f"semantic_depth must be between {SEMANTIC_DEPTH_MIN} and {SEMANTIC_DEPTH_MAX}")
    return depth


def _semantic_feature_width(semantic_depth: int) -> int:
    depth = _validate_semantic_depth(semantic_depth)
    return (
        SEMANTIC_SUMMARY_WIDTH
        + (2 * depth - 1)
        + SEMANTIC_SCORE_SHAPE_WIDTH
        + SEMANTIC_RANK_AGREEMENT_WIDTH
        + EMBEDDING_PROJECTION_WIDTH
    )


def _semantic_rank_profile(similarities: np.ndarray, k: int) -> list[float]:
    values = [float(value) for value in np.asarray(similarities, dtype=float)[:k]]
    padded = [*values, *([0.0] * max(k - len(values), 0))]
    adjacent_deltas = [
        values[idx] - values[idx + 1] if idx + 1 < len(values) else 0.0
        for idx in range(k - 1)
    ]
    return [*padded[:k], *adjacent_deltas]


def _semantic_score_shape(similarities: np.ndarray) -> list[float]:
    values = np.asarray(similarities, dtype=float).reshape(-1)
    if values.size == 0:
        return [0.0] * SEMANTIC_SCORE_SHAPE_WIDTH
    ranked = np.sort(values)[::-1]
    head = ranked[: min(3, ranked.size)]
    tail = ranked[-min(2, ranked.size) :]
    non_top = ranked[1:]
    head_mean = float(np.mean(head)) if head.size else 0.0
    tail_mean = float(np.mean(tail)) if tail.size else head_mean
    non_top_positive_rate = float(np.mean(non_top > 0.0)) if non_top.size else 0.0
    return [
        head_mean,
        tail_mean,
        head_mean - tail_mean,
        non_top_positive_rate,
    ]


def _semantic_rank_agreement(similarities: np.ndarray, k: int) -> list[float]:
    values = np.asarray(similarities, dtype=float).reshape(-1)[:k]
    if values.size == 0:
        return [0.0] * 6
    semantic_order = sorted(range(len(values)), key=lambda idx: (-values[idx], idx))
    semantic_ranks = np.empty(len(values), dtype=int)
    for rank, idx in enumerate(semantic_order, start=1):
        semantic_ranks[idx] = rank

    best_semantic_bm25_rank = int(semantic_order[0]) + 1
    top_bm25_semantic_rank = int(semantic_ranks[0])
    bm25_scores = pd.Series(np.arange(len(values), 0, -1, dtype=float))
    semantic_scores = pd.Series(values)
    spearman = bm25_scores.corr(semantic_scores, method="spearman")
    if pd.isna(spearman):
        spearman = 0.0

    top_n = min(2, len(values))
    bm25_top = set(range(top_n))
    semantic_top = set(semantic_order[:top_n])
    overlap_at_2 = len(bm25_top & semantic_top) / top_n
    semantic_top_mean_bm25_rr = float(np.mean([1.0 / (idx + 1) for idx in semantic_order[:top_n]]))
    return [
        best_semantic_bm25_rank / len(values),
        1.0 / best_semantic_bm25_rank,
        top_bm25_semantic_rank / len(values),
        float(spearman),
        float(overlap_at_2),
        semantic_top_mean_bm25_rr,
    ]


def _embedding_projection_features(query_embedding: np.ndarray, passage_centroid: np.ndarray) -> list[float]:
    query = np.asarray(query_embedding, dtype=float)
    centroid = np.asarray(passage_centroid, dtype=float)
    delta = query - centroid
    return [
        *_fixed_random_projection(query, dims=4, seed=11),
        *_fixed_random_projection(centroid, dims=4, seed=17),
        *_fixed_random_projection(delta, dims=4, seed=23),
    ]


def _fixed_random_projection(vector: np.ndarray, dims: int, seed: int) -> list[float]:
    values = np.asarray(vector, dtype=float).reshape(-1)
    if values.size == 0:
        return [0.0] * dims
    rng = np.random.default_rng(seed)
    projection = rng.normal(loc=0.0, scale=1.0 / np.sqrt(values.size), size=(dims, values.size))
    return [float(value) for value in projection @ values]


def _summarize_retrieval_policy(df: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    test = df[df["split"] == "test"]
    metrics = ["recall_at_5", "mrr", "ndcg_at_5", "reward", "rewrite_cost", "retrieval_calls"]
    rows = []
    for method in method_order:
        part = test[test["method"] == method]
        row = {"method": method}
        for metric in metrics:
            row[metric] = float(part[metric].mean()) if len(part) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_retrieval_policy(df: pd.DataFrame, method_order: list[str] | None = None) -> pd.DataFrame:
    return _summarize_retrieval_policy(df, method_order or METHOD_ORDER)


def _evaluate_action(
    ex: QAExample,
    action: str,
    query: str,
    results: list[RetrievalResult],
    rewritten: object,
    retrieval_calls: int,
    k: int,
    retrieval_call_cost: float,
    fixed_rewrite_cost: float | None = None,
) -> dict[str, object]:
    rec = recall_at_k(results, ex.gold_doc_ids, k)
    rr = mrr(results, ex.gold_doc_ids)
    action_cost = fixed_rewrite_cost if fixed_rewrite_cost is not None else rewrite_cost(ex.question, rewritten)
    cost = action_cost + retrieval_call_cost * max(retrieval_calls - 1, 0)
    return {
        "recall_at_5": rec,
        "mrr": rr,
        "ndcg_at_5": ndcg_at_k(results, ex.gold_doc_ids, k),
        "rewrite_cost": cost,
        "retrieval_calls": retrieval_calls,
        "rewrite_tokens": len(query.split()),
        "reward": rec + 0.5 * rr - cost,
        "queries": query,
        "top_docs": " | ".join(r.doc_id for r in results),
        "gold_docs": " | ".join(sorted(ex.gold_doc_ids)),
    }


def _difficulty_feature_row(action_eval: dict[str, object], actions: list[str], k: int) -> dict[str, object]:
    features = np.asarray(action_eval["features"], dtype=float)
    action_outputs = action_eval["actions"]
    rewards = np.asarray([float(action_outputs[action]["reward"]) for action in actions], dtype=float)
    sorted_rewards = np.sort(rewards)
    top_reward = float(sorted_rewards[-1]) if sorted_rewards.size else 0.0
    runner_up_reward = float(sorted_rewards[-2]) if sorted_rewards.size > 1 else top_reward
    bm25_keep = _top_doc_ids(action_outputs, "bm25_keep")
    dense_keep = _top_doc_ids(action_outputs, "dense_keep")
    hybrid_keep = _top_doc_ids(action_outputs, "hybrid_keep")
    return {
        "state_question_length": float(features[1]) if features.size > 1 else 0.0,
        "state_capitalized_spans": float(features[2]) if features.size > 2 else 0.0,
        "state_bm25_top1": float(features[3]) if features.size > 3 else 0.0,
        "state_bm25_gap": float(features[4]) if features.size > 4 else 0.0,
        "state_bm25_entropy": float(features[5]) if features.size > 5 else 0.0,
        "oracle_reward_margin": float(top_reward - runner_up_reward),
        "oracle_tie_count": int(np.count_nonzero(np.isclose(rewards, top_reward))) if rewards.size else 0,
        "action_reward_std": float(np.std(rewards)) if rewards.size else 0.0,
        "bm25_dense_doc_overlap": _overlap_rate(bm25_keep, dense_keep, k),
        "dense_new_doc_rate": _new_doc_rate(dense_keep, bm25_keep, k),
        "hybrid_new_doc_rate": _new_doc_rate(hybrid_keep, bm25_keep, k),
    }


def _method_row(
    split_name: str,
    method_name: str,
    action: str,
    ex: QAExample,
    action_eval: dict[str, object],
    difficulty_features: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "split": split_name,
        "method": method_name,
        "action": action,
        "qid": ex.qid,
        "question": ex.question,
        **action_eval["actions"][action],
    }
    if difficulty_features:
        row.update(difficulty_features)
    if extra:
        row.update(extra)
    return row
