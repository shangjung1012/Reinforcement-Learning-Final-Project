from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import selective_rag_rl.experiments.retrieval_policy_experiment as rpe
from selective_rag_rl.core.data import Passage, QAExample, load_hotpotqa
from selective_rag_rl.experiments.dense_experiment import FakeDenseEmbedder
from selective_rag_rl.experiments.feature_ablation import run_feature_ablation
from selective_rag_rl.experiments.gemini_baseline import GeminiCache
from selective_rag_rl.experiments.learning_curve import run_policy_learning_curve
from selective_rag_rl.experiments.policy_model_sweep import run_policy_model_sweep
from selective_rag_rl.core.policy_io import load_checkpoint
from selective_rag_rl.diagnostics.qualitative import export_qualitative_examples
from selective_rag_rl.experiments.retrieval_policy_experiment import (
    _retrieval_contrast_features,
    _semantic_state_features,
    evaluate_retrieval_actions,
    fit_policy_feature_transform,
    run_llm_retrieval_policy_experiment,
    run_nfcorpus_retrieval_policy_experiment,
    run_retrieval_policy_experiment,
    run_retrieval_policy_on_examples,
    semantic_feature_group_slices,
    run_scifact_retrieval_policy_experiment,
    select_policy_model,
    transform_policy_features,
)
from selective_rag_rl.core.retriever import RetrievalResult


def test_run_retrieval_policy_experiment_writes_summary_and_checkpoint(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(10)]), encoding="utf-8")

    metadata = run_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=10,
        seed=7,
        embedder_name="fake",
        retrieval_call_cost=0.03,
        knn_k_candidates=[1, 3],
        tuning_folds=3,
    )

    summary = Path(metadata["outputs"]["summary_csv"]).read_text(encoding="utf-8")
    assert "Vanilla BM25" in summary
    assert "Dense original" in summary
    assert "Hybrid keyword" in summary
    assert "Heuristic retrieval router" in summary
    assert "Selective retrieval policy" in summary
    assert "Oracle retrieval action" in summary

    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert checkpoint["metadata"]["actions"] == [
        "bm25_keep",
        "bm25_keyword",
        "dense_keep",
        "dense_keyword",
        "hybrid_keep",
        "hybrid_keyword",
    ]
    assert checkpoint["metadata"]["retrieval_call_cost"] == 0.03
    assert checkpoint["metadata"]["policy_model"] == "knn"
    assert checkpoint["metadata"]["selected_policy_model"] in {"knn_k=1", "knn_k=3"}
    assert checkpoint["metadata"]["selected_knn_k"] in [1, 3]
    assert set(checkpoint["metadata"]["validation_policy_scores"]) == {"knn_k=1", "knn_k=3"}
    assert checkpoint["metadata"]["tuning_folds"] == 3
    assert metadata["selected_knn_k"] in [1, 3]
    assert metadata["policy_model"] == "knn"
    assert metadata["selected_policy_model"] in {"knn_k=1", "knn_k=3"}
    assert set(metadata["validation_policy_scores"]) == {"knn_k=1", "knn_k=3"}
    assert metadata["tuning_folds"] == 3
    assert Path(metadata["outputs"]["action_distribution_csv"]).exists()
    assert metadata["policy_action_distribution"]["test"]["total"] == 4
    assert sum(metadata["policy_action_distribution"]["test"]["actions"].values()) == 4

    detailed = pd.read_csv(metadata["outputs"]["detailed_csv"])
    assert {
        "state_question_length",
        "state_capitalized_spans",
        "state_bm25_top1",
        "state_bm25_gap",
        "state_bm25_entropy",
        "oracle_reward_margin",
        "oracle_tie_count",
        "action_reward_std",
        "bm25_dense_doc_overlap",
        "dense_new_doc_rate",
        "hybrid_new_doc_rate",
    } <= set(detailed.columns)
    selective_rows = detailed[detailed["method"] == "Selective retrieval policy"]
    assert selective_rows["state_bm25_top1"].notna().all()
    assert (selective_rows["oracle_tie_count"] >= 1).all()


def test_run_retrieval_policy_can_report_confidence_gated_policy(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(10)]), encoding="utf-8")

    metadata = run_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=10,
        seed=7,
        embedder_name="fake",
        retrieval_call_cost=0.03,
        knn_k_candidates=[1],
        tuning_folds=2,
        confidence_gate_margin=999.0,
    )

    summary = pd.read_csv(metadata["outputs"]["summary_csv"])
    detailed = pd.read_csv(metadata["outputs"]["detailed_csv"])
    gated = detailed[detailed["method"] == "Confidence-gated retrieval policy"]

    assert "Confidence-gated retrieval policy" in set(summary["method"])
    assert len(gated[gated["split"] == "test"]) == metadata["test_examples"]
    assert {
        "policy_action",
        "fallback_action",
        "fallback_used",
        "predicted_action_margin",
        "policy_action_score",
        "runner_up_action_score",
    } <= set(detailed.columns)
    assert metadata["confidence_gate_margin"] == 999.0
    assert (gated["fallback_action"] == metadata["best_fixed_action"]).all()
    assert gated["fallback_used"].astype(bool).all()


def test_run_retrieval_policy_can_select_tree_policy_model(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(12)]), encoding="utf-8")

    metadata = run_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=12,
        seed=7,
        embedder_name="fake",
        policy_model="auto",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    assert metadata["policy_model"] == "auto"
    assert metadata["selected_policy_model"] in {
        "knn_k=1",
        "ridge_l2=1.0",
        "extra_trees",
        "random_forest",
        "mlp",
    }
    assert set(metadata["validation_policy_scores"]) == {
        "knn_k=1",
        "ridge_l2=1.0",
        "extra_trees",
        "random_forest",
        "mlp",
    }


def test_run_retrieval_policy_can_checkpoint_tree_policy_model(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(12)]), encoding="utf-8")

    metadata = run_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=12,
        seed=7,
        embedder_name="fake",
        policy_model="extra_trees",
        tuning_folds=2,
    )

    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert checkpoint["metadata"]["selected_policy_model"] == "extra_trees"
    assert checkpoint["model"].predict(np.ones(checkpoint["model"].models["bm25_keep"].n_features_in_))


def test_ridge_policy_model_cross_validates_regularization_strengths() -> None:
    features = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=float)
    rewards = {
        "left": [1.0, 0.8, 0.2, 0.0],
        "right": [0.0, 0.2, 0.8, 1.0],
    }

    _, selected_model, validation_scores = select_policy_model(
        features=features,
        rewards=rewards,
        policy_model="ridge_sweep",
        knn_k_candidates=[1],
        folds=2,
        seed=7,
        actions=["left", "right"],
    )

    assert selected_model.startswith("ridge_l2=")
    assert set(validation_scores) == {
        "ridge_l2=0.01",
        "ridge_l2=0.1",
        "ridge_l2=1.0",
        "ridge_l2=10.0",
        "ridge_l2=100.0",
    }


def test_margin_ridge_policy_model_uses_reward_margin_weighting() -> None:
    features = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=float)
    rewards = {
        "left": [1.0, 0.51, 0.50, 0.0],
        "right": [0.0, 0.50, 0.49, 1.0],
    }

    policy, selected_model, validation_scores = select_policy_model(
        features=features,
        rewards=rewards,
        policy_model="margin_ridge",
        knn_k_candidates=[1],
        folds=2,
        seed=7,
        actions=["left", "right"],
    )

    assert selected_model == "margin_ridge_l2=1.0"
    assert set(validation_scores) == {"margin_ridge_l2=1.0"}
    policy.fit(features, rewards)
    assert policy.sample_weights_[1] < policy.sample_weights_[0]


def test_auto_policy_model_can_limit_candidate_families() -> None:
    features = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=float)
    rewards = {
        "left": [1.0, 0.8, 0.2, 0.0],
        "right": [0.0, 0.2, 0.8, 1.0],
    }

    _, selected_model, validation_scores = select_policy_model(
        features=features,
        rewards=rewards,
        policy_model="auto",
        knn_k_candidates=[1],
        folds=2,
        seed=7,
        actions=["left", "right"],
        auto_candidate_models=["knn", "ridge"],
    )

    assert selected_model in validation_scores
    assert set(validation_scores) == {"knn_k=1", "ridge_l2=1.0"}


def test_run_retrieval_policy_can_checkpoint_mlp_policy_model(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(12)]), encoding="utf-8")

    metadata = run_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=12,
        seed=7,
        embedder_name="fake",
        policy_model="mlp",
        tuning_folds=2,
    )

    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert checkpoint["metadata"]["selected_policy_model"] == "mlp"
    assert checkpoint["model"].predict(np.ones(checkpoint["model"].models["bm25_keep"].n_features_in_))


def test_run_llm_retrieval_policy_experiment_writes_cached_actions(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    cache_path = tmp_path / "gemini_cache.jsonl"
    data_path.write_text(json.dumps([_example(i) for i in range(12)]), encoding="utf-8")

    metadata = run_llm_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=12,
        seed=7,
        embedder_name="fake",
        cache_path=cache_path,
        rewrite_provider=lambda question, mode: [question] if mode == "rewrite" else [question, question],
        policy_model="auto",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    summary = Path(metadata["outputs"]["summary_csv"]).read_text(encoding="utf-8")
    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert "Gemini rewrite action" in summary
    assert "Gemini decompose action" in summary
    assert "Gemini HyDE BM25 action" in summary
    assert "Gemini HyDE dense action" in summary
    assert "Gemini HyDE hybrid action" in summary
    assert "Gemini multi-query BM25 action" in summary
    assert "Gemini multi-query hybrid action" in summary
    assert "Gemini decompose hybrid action" in summary
    assert "Selective retrieval policy" in summary
    assert "Oracle retrieval action" in summary
    assert "bm25_llm_rewrite" in checkpoint["metadata"]["actions"]
    assert "bm25_llm_decompose" in checkpoint["metadata"]["actions"]
    assert "hybrid_llm_decompose" in checkpoint["metadata"]["actions"]
    assert "bm25_hyde" in checkpoint["metadata"]["actions"]
    assert "dense_hyde" in checkpoint["metadata"]["actions"]
    assert "hybrid_hyde" in checkpoint["metadata"]["actions"]
    assert "bm25_multi_query" in checkpoint["metadata"]["actions"]
    assert "hybrid_multi_query" in checkpoint["metadata"]["actions"]
    assert checkpoint["metadata"]["llm_cache_path"] == str(cache_path)
    assert checkpoint["metadata"]["llm_base_cost"] == 1.0
    assert cache_path.exists()


def test_run_retrieval_policy_can_use_semantic_features(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(10)]), encoding="utf-8")
    examples = load_hotpotqa(data_path, num_examples=10, seed=7)

    metadata = run_retrieval_policy_on_examples(
        examples=examples,
        output_dir=output_dir,
        dataset_name="test semantic retrieval policy",
        output_prefix="semantic_policy",
        checkpoint_name="semantic_policy.pkl",
        embedder_name="fake",
        semantic_features="fake",
        semantic_embedder=_FakeSemanticEmbedder(),
        knn_k_candidates=[1, 3],
        tuning_folds=2,
    )

    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert checkpoint["metadata"]["semantic_features"] == "fake"
    assert checkpoint["metadata"]["feature_width"] == 60
    assert checkpoint["model"].mean.shape[0] == 60


def test_load_semantic_embedder_passes_vertex_budget_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeVertexProvider:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(rpe, "VertexTextEmbeddingProvider", FakeVertexProvider)

    embedder = rpe._load_semantic_embedder(
        "vertex",
        tmp_path / "outputs",
        tmp_path / "cache" / "vertex.jsonl",
        semantic_allow_api=True,
        semantic_max_new_texts=12,
        semantic_dry_run=True,
    )

    assert isinstance(embedder, FakeVertexProvider)
    assert captured["allow_api"] is True
    assert captured["max_new_texts"] == 12
    assert captured["dry_run"] is True
    assert captured["cache_path"] == tmp_path / "cache" / "vertex.jsonl"


def test_run_retrieval_policy_forwards_semantic_budget_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(10)]), encoding="utf-8")
    examples = load_hotpotqa(data_path, num_examples=10, seed=7)
    captured: dict[str, object] = {}

    def fake_load_semantic_embedder(
        semantic_features: str,
        output_dir: Path,
        semantic_cache_path: Path | None,
        *,
        semantic_allow_api: bool,
        semantic_max_new_texts: int,
        semantic_dry_run: bool,
    ) -> _FakeSemanticEmbedder:
        captured.update(
            {
                "semantic_features": semantic_features,
                "output_dir": output_dir,
                "semantic_cache_path": semantic_cache_path,
                "semantic_allow_api": semantic_allow_api,
                "semantic_max_new_texts": semantic_max_new_texts,
                "semantic_dry_run": semantic_dry_run,
            }
        )
        return _FakeSemanticEmbedder()

    monkeypatch.setattr(rpe, "_load_semantic_embedder", fake_load_semantic_embedder)

    metadata = rpe.run_retrieval_policy_on_examples(
        examples=examples,
        output_dir=output_dir,
        dataset_name="semantic budget retrieval policy",
        output_prefix="semantic_budget_policy",
        checkpoint_name="semantic_budget_policy.pkl",
        embedder_name="fake",
        semantic_features="vertex",
        semantic_cache_path=tmp_path / "cache" / "vertex.jsonl",
        semantic_allow_api=True,
        semantic_max_new_texts=7,
        semantic_dry_run=False,
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    assert captured["semantic_allow_api"] is True
    assert captured["semantic_max_new_texts"] == 7
    assert captured["semantic_dry_run"] is False
    assert metadata["semantic_allow_api"] is True
    assert metadata["semantic_max_new_texts"] == 7
    assert metadata["semantic_dry_run"] is False


def test_run_retrieval_policy_accepts_subset_action_map(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(8)]), encoding="utf-8")
    examples = load_hotpotqa(data_path, num_examples=8, seed=7)

    default_map_metadata = run_retrieval_policy_on_examples(
        examples=examples,
        output_dir=output_dir,
        dataset_name="subset retrieval policy default map",
        output_prefix="subset_policy_default",
        checkpoint_name="subset_policy_default.pkl",
        embedder_name="fake",
        actions=["bm25_keep", "bm25_keyword"],
        method_order=[
            "Vanilla BM25",
            "BM25 keyword",
            "Train-best retrieval action",
            "Heuristic retrieval router",
            "Selective retrieval policy",
            "Oracle retrieval action",
        ],
        knn_k_candidates=[1],
        tuning_folds=2,
    )
    metadata = run_retrieval_policy_on_examples(
        examples=examples,
        output_dir=output_dir,
        dataset_name="subset retrieval policy",
        output_prefix="subset_policy",
        checkpoint_name="subset_policy.pkl",
        embedder_name="fake",
        actions=["bm25_keep", "bm25_keyword"],
        action_to_method={"bm25_keep": "BM25 original", "bm25_keyword": "BM25 keyword"},
        method_order=[
            "BM25 original",
            "BM25 keyword",
            "Train-best retrieval action",
            "Heuristic retrieval router",
            "Selective retrieval policy",
            "Oracle retrieval action",
        ],
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    default_map_summary = pd.read_csv(default_map_metadata["outputs"]["summary_csv"])
    summary = pd.read_csv(metadata["outputs"]["summary_csv"])
    assert "Dense original" not in set(default_map_summary["method"])
    assert "Hybrid keyword" not in set(default_map_summary["method"])
    assert set(summary["method"]) >= {
        "BM25 original",
        "BM25 keyword",
        "Train-best retrieval action",
        "Heuristic retrieval router",
        "Selective retrieval policy",
        "Oracle retrieval action",
    }
    assert "Dense original" not in set(summary["method"])
    assert "Hybrid keyword" not in set(summary["method"])


def test_run_retrieval_policy_records_extended_semantic_feature_width(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(10)]), encoding="utf-8")
    examples = load_hotpotqa(data_path, num_examples=10, seed=7)

    metadata = run_retrieval_policy_on_examples(
        examples=examples,
        output_dir=output_dir,
        dataset_name="test semantic retrieval policy",
        output_prefix="semantic_policy",
        checkpoint_name="semantic_policy.pkl",
        embedder_name="fake",
        semantic_features="fake",
        semantic_embedder=_FakeSemanticEmbedder(),
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert metadata["feature_width"] >= 41
    assert checkpoint["metadata"]["feature_width"] == metadata["feature_width"]
    assert checkpoint["model"].mean.shape[0] == checkpoint["metadata"]["feature_width"]


def test_transform_policy_features_masks_expected_groups() -> None:
    features = np.arange(19, dtype=float)

    no_query = transform_policy_features(features, "no_query")
    assert no_query[1] == 0.0
    assert no_query[2] == 0.0
    assert np.all(no_query[6:14] == 0.0)
    assert list(no_query[[0, 3, 4, 5, 14]]) == [0.0, 3.0, 4.0, 5.0, 14.0]

    no_retrieval = transform_policy_features(features, "no_retrieval")
    assert np.all(no_retrieval[3:6] == 0.0)
    assert list(no_retrieval[[0, 1, 2, 6]]) == [0.0, 1.0, 2.0, 6.0]

    retrieval_only = transform_policy_features(features, "retrieval_only")
    assert np.all(retrieval_only[[1, 2]] == 0.0)
    assert np.all(retrieval_only[6:] == 0.0)
    assert list(retrieval_only[[0, 3, 4, 5]]) == [0.0, 3.0, 4.0, 5.0]


def test_transform_policy_features_can_isolate_semantic_group() -> None:
    features = np.arange(60, dtype=float)

    no_semantic = transform_policy_features(features, "no_semantic")
    assert np.all(no_semantic[14:] == 0.0)
    assert list(no_semantic[[0, 3, 13]]) == [0.0, 3.0, 13.0]

    semantic_only = transform_policy_features(features, "semantic_only")
    assert np.all(semantic_only[:14] == 0.0)
    assert np.all(semantic_only[42:] == 0.0)
    assert list(semantic_only[[14, 22, 23, 31, 32, 35, 36, 41]]) == [
        14.0,
        22.0,
        23.0,
        31.0,
        32.0,
        35.0,
        36.0,
        41.0,
    ]


def test_transform_policy_features_can_isolate_semantic_rank_profile() -> None:
    features = np.arange(54, dtype=float)

    no_profile = transform_policy_features(features, "no_profile")
    assert np.all(no_profile[23:32] == 0.0)
    assert list(no_profile[[14, 22, 32, 53]]) == [14.0, 22.0, 32.0, 53.0]

    profile_only = transform_policy_features(features, "profile_only")
    assert np.all(profile_only[:23] == 0.0)
    assert np.all(profile_only[32:] == 0.0)
    assert list(profile_only[[23, 27, 31]]) == [23.0, 27.0, 31.0]


def test_transform_policy_features_can_isolate_semantic_score_shape_group() -> None:
    features = np.arange(60, dtype=float)

    no_score_shape = transform_policy_features(features, "no_score_shape")
    assert np.all(no_score_shape[32:36] == 0.0)
    assert list(no_score_shape[[14, 31, 36, 59]]) == [14.0, 31.0, 36.0, 59.0]

    score_shape_only = transform_policy_features(features, "score_shape_only")
    assert np.all(score_shape_only[:32] == 0.0)
    assert np.all(score_shape_only[36:] == 0.0)
    assert list(score_shape_only[[32, 34, 35]]) == [32.0, 34.0, 35.0]


def test_transform_policy_features_can_isolate_embedding_projection_group() -> None:
    features = np.arange(60, dtype=float)

    no_projection = transform_policy_features(features, "no_projection")
    assert np.all(no_projection[42:54] == 0.0)
    assert list(no_projection[[14, 31, 32, 54, 59]]) == [14.0, 31.0, 32.0, 54.0, 59.0]

    projection_only = transform_policy_features(features, "projection_only")
    assert np.all(projection_only[:42] == 0.0)
    assert np.all(projection_only[54:] == 0.0)
    assert list(projection_only[[42, 48, 53]]) == [42.0, 48.0, 53.0]


def test_transform_policy_features_can_isolate_semantic_rank_agreement_group() -> None:
    features = np.arange(60, dtype=float)

    no_rank_agreement = transform_policy_features(features, "no_rank_agreement")
    assert np.all(no_rank_agreement[36:42] == 0.0)
    assert list(no_rank_agreement[[14, 31, 42, 59]]) == [14.0, 31.0, 42.0, 59.0]

    rank_agreement_only = transform_policy_features(features, "rank_agreement_only")
    assert np.all(rank_agreement_only[:36] == 0.0)
    assert np.all(rank_agreement_only[42:] == 0.0)
    assert list(rank_agreement_only[[36, 39, 41]]) == [36.0, 39.0, 41.0]


def test_transform_policy_features_can_isolate_interaction_group() -> None:
    features = np.arange(60, dtype=float)

    no_interactions = transform_policy_features(features, "no_interactions")
    assert np.all(no_interactions[54:] == 0.0)
    assert list(no_interactions[[0, 14, 31, 53]]) == [0.0, 14.0, 31.0, 53.0]

    interactions_only = transform_policy_features(features, "interactions_only")
    assert np.all(interactions_only[:54] == 0.0)
    assert list(interactions_only[[54, 56, 59]]) == [54.0, 56.0, 59.0]


def test_fit_policy_feature_transform_zscores_semantic_columns_with_train_stats() -> None:
    features = np.vstack(
        [
            np.arange(60, dtype=float),
            np.arange(60, dtype=float) + 10.0,
            np.arange(60, dtype=float) + 20.0,
        ]
    )

    transform = fit_policy_feature_transform(features, "semantic_zscore")
    transformed = transform.transform(features)
    test_vector = transform.transform(features[0])

    assert np.allclose(transformed[:, :14], features[:, :14])
    assert np.allclose(transformed[:, 14:].mean(axis=0), 0.0)
    assert np.allclose(transformed[:, 14:].std(axis=0), 1.0)
    assert np.allclose(test_vector[14:], transformed[0, 14:])


def test_transform_policy_features_can_apply_semantic_zscore_to_matrix() -> None:
    features = np.vstack([np.arange(60, dtype=float), np.arange(60, dtype=float) + 2.0])

    transformed = transform_policy_features(features, "semantic_zscore")

    assert np.allclose(transformed[:, :14], features[:, :14])
    assert np.allclose(transformed[:, 14:].mean(axis=0), 0.0)
    assert np.allclose(transformed[:, 14:].std(axis=0), 1.0)


def test_retrieval_contrast_features_measure_action_rank_overlap() -> None:
    action_outputs = {
        "bm25_keep": {"top_docs": "a | b | c"},
        "dense_keep": {"top_docs": "b | c | d"},
        "hybrid_keep": {"top_docs": "a | b | d"},
        "bm25_keyword": {"top_docs": "a | c"},
        "dense_keyword": {"top_docs": "c | d"},
        "hybrid_keyword": {"top_docs": "a | d"},
    }

    features = _retrieval_contrast_features(action_outputs, k=3)

    assert np.allclose(features, [2 / 3, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3])


def test_evaluate_retrieval_actions_supports_generated_action_family(tmp_path: Path) -> None:
    ex = QAExample(
        qid="q1",
        question="alpha beta evidence",
        answer="alpha",
        passages=[
            Passage("d1", "D1", "alpha beta evidence"),
            Passage("d2", "D2", "beta dense evidence"),
            Passage("d3", "D3", "gamma alternate evidence"),
            Passage("d4", "D4", "unrelated tokens"),
        ],
        gold_doc_ids={"d1"},
        level="easy",
        qtype="toy",
    )
    cache = GeminiCache(tmp_path / "generated.jsonl")

    def provider(question: str, mode: str) -> list[str]:
        if mode == "hyde":
            return ["alpha beta evidence supporting passage"]
        if mode == "multi_query":
            return ["alpha evidence", "beta evidence", "gamma evidence"]
        if mode == "decompose":
            return ["alpha evidence", "beta evidence"]
        return [question]

    evaluated = evaluate_retrieval_actions(
        ex,
        FakeDenseEmbedder(),
        k=3,
        dense_weight=0.5,
        retrieval_call_cost=0.03,
        actions=["bm25_hyde", "dense_hyde", "hybrid_hyde", "bm25_multi_query", "hybrid_multi_query", "hybrid_llm_decompose"],
        llm_cache=cache,
        rewrite_provider=provider,
        llm_base_cost=1.0,
        llm_token_cost=0.01,
    )

    actions = evaluated["actions"]
    assert set(actions) == {
        "bm25_hyde",
        "dense_hyde",
        "hybrid_hyde",
        "bm25_multi_query",
        "hybrid_multi_query",
        "hybrid_llm_decompose",
    }
    assert actions["bm25_hyde"]["retrieval_calls"] == 1
    assert actions["dense_hyde"]["retrieval_calls"] == 1
    assert actions["hybrid_hyde"]["retrieval_calls"] == 2
    assert actions["bm25_multi_query"]["retrieval_calls"] == 3
    assert actions["hybrid_multi_query"]["retrieval_calls"] == 6
    assert actions["hybrid_llm_decompose"]["retrieval_calls"] == 4
    assert cache.get("q1", "hyde") == ["alpha beta evidence supporting passage"]
    assert cache.get("q1", "multi_query") == ["alpha evidence", "beta evidence", "gamma evidence"]
    assert cache.get("q1", "decompose") == ["alpha evidence", "beta evidence"]


def test_evaluate_retrieval_actions_can_append_retrieval_contrast_features() -> None:
    ex = QAExample(
        qid="q1",
        question="alpha beta evidence",
        answer="alpha",
        passages=[
            Passage("d1", "D1", "alpha beta evidence"),
            Passage("d2", "D2", "beta dense evidence"),
            Passage("d3", "D3", "unrelated tokens"),
            Passage("d4", "D4", "more unrelated text"),
        ],
        gold_doc_ids={"d1"},
        level="easy",
        qtype="toy",
    )

    base = evaluate_retrieval_actions(ex, FakeDenseEmbedder(), k=3, dense_weight=0.5, retrieval_call_cost=0.03)
    contrasted = evaluate_retrieval_actions(
        ex,
        FakeDenseEmbedder(),
        k=3,
        dense_weight=0.5,
        retrieval_call_cost=0.03,
        retrieval_contrast_features=True,
    )

    assert len(base["features"]) == 14
    assert len(contrasted["features"]) == 22
    assert np.all(np.asarray(contrasted["features"][-8:]) >= 0.0)
    assert np.all(np.asarray(contrasted["features"][-8:]) <= 1.0)


def test_semantic_state_features_add_rank_aware_similarity_profile() -> None:
    ex = QAExample(
        qid="q1",
        question="Which author lived longer, Entity1A or Entity1B?",
        answer="Entity1A",
        passages=[
            Passage("Entity1A", "Entity1A", "Entity1A was a writer with a long biography."),
            Passage("Entity1B", "Entity1B", "Entity1B was another writer with a long biography."),
            Passage("Distractor1", "Distractor1", "This paragraph is unrelated."),
        ],
        gold_doc_ids={"Entity1A", "Entity1B"},
        level="easy",
        qtype="comparison",
    )
    initial_results = [
        RetrievalResult("Entity1A", 3.0, 1),
        RetrievalResult("Entity1B", 2.0, 2),
        RetrievalResult("Distractor1", 1.0, 3),
    ]

    features = _semantic_state_features(ex, initial_results, _FakeSemanticEmbedder())

    assert len(features) == 40
    assert all(value > 0.0 for value in features[9:12])
    assert features[12] == 0.0
    assert features[13] == 0.0
    assert np.isclose(features[14], features[9] - features[10])
    assert len(features[18:22]) == 4


def test_semantic_state_features_add_score_shape_features() -> None:
    similarities = np.asarray([0.9, 0.6, 0.3, -0.1, -0.2])

    features = semantic_feature_group_slices(60)
    score_shape_start, score_shape_end = features["semantic_score_shape"]
    shape = _semantic_state_features(
        QAExample(
            qid="q1",
            question="score shape query",
            answer="alpha",
            passages=[
                Passage(f"d{i}", f"D{i}", f"doc {i}")
                for i in range(5)
            ],
            gold_doc_ids={"d1"},
            level="easy",
            qtype="toy",
        ),
        [RetrievalResult(f"d{i}", float(5 - i), i + 1) for i in range(5)],
        _FixedSimilarityEmbedder(similarities),
    )[score_shape_start - 14 : score_shape_end - 14]

    assert np.allclose(shape, [0.6, -0.15, 0.75, 0.5])


def test_semantic_state_features_add_bm25_semantic_rank_agreement() -> None:
    ex = QAExample(
        qid="q1",
        question="Which author lived longer, Entity1A or Entity1B?",
        answer="Entity1A",
        passages=[
            Passage("bm25-first", "bm25-first", "alpha"),
            Passage("semantic-first", "semantic-first", "bravo"),
            Passage("semantic-second", "semantic-second", "charlie"),
        ],
        gold_doc_ids={"semantic-first"},
        level="easy",
        qtype="comparison",
    )
    initial_results = [
        RetrievalResult("bm25-first", 10.0, 1),
        RetrievalResult("semantic-first", 9.0, 2),
        RetrievalResult("semantic-second", 8.0, 3),
    ]

    features = _semantic_state_features(ex, initial_results, _RankAgreementEmbedder())
    rank_features = features[22:28]

    assert len(features) == 40
    assert rank_features[0] == 2 / 3
    assert rank_features[1] == 1 / 2
    assert rank_features[2] == 1.0
    assert rank_features[3] < 0.0
    assert rank_features[4] == 1 / 2
    assert np.isclose(rank_features[5], 5 / 12)


def test_semantic_state_features_can_use_deeper_rank_profile() -> None:
    ex = QAExample(
        qid="q1",
        question="Which passage is supported by alpha?",
        answer="alpha",
        passages=[
            Passage(f"d{i}", f"D{i}", f"alpha evidence paragraph {i}")
            for i in range(8)
        ],
        gold_doc_ids={"d1"},
        level="easy",
        qtype="toy",
    )
    initial_results = [RetrievalResult(f"d{i}", float(10 - i), i + 1) for i in range(8)]

    shallow = _semantic_state_features(ex, initial_results, _FakeSemanticEmbedder(), semantic_depth=3)
    deep = _semantic_state_features(ex, initial_results, _FakeSemanticEmbedder(), semantic_depth=8)

    assert len(shallow) == 36
    assert len(deep) == 46
    assert len(deep[9:24]) == 15


def test_evaluate_retrieval_actions_uses_semantic_depth_for_initial_embedding_pool() -> None:
    ex = QAExample(
        qid="q1",
        question="alpha beta evidence",
        answer="alpha",
        passages=[
            Passage("d1", "D1", "alpha beta evidence"),
            Passage("d2", "D2", "alpha beta more evidence"),
            Passage("d3", "D3", "alpha beta third evidence"),
            Passage("d4", "D4", "alpha beta fourth evidence"),
        ],
        gold_doc_ids={"d1"},
        level="easy",
        qtype="toy",
    )
    semantic_embedder = _TrackingSemanticEmbedder()

    evaluated = evaluate_retrieval_actions(
        ex,
        FakeDenseEmbedder(),
        k=2,
        dense_weight=0.5,
        retrieval_call_cost=0.03,
        semantic_embedder=semantic_embedder,
        semantic_depth=4,
    )

    assert len(evaluated["features"]) == 58
    assert semantic_embedder.document_batch_sizes == [4]


def test_evaluate_retrieval_actions_rejects_ambiguous_semantic_depth() -> None:
    ex = QAExample(
        qid="q1",
        question="alpha beta evidence",
        answer="alpha",
        passages=[Passage("d1", "D1", "alpha beta evidence")],
        gold_doc_ids={"d1"},
        level="easy",
        qtype="toy",
    )

    with pytest.raises(ValueError, match="semantic_depth"):
        evaluate_retrieval_actions(
            ex,
            FakeDenseEmbedder(),
            k=1,
            dense_weight=0.5,
            retrieval_call_cost=0.03,
            semantic_embedder=_FakeSemanticEmbedder(),
            semantic_depth=2,
        )


def test_run_retrieval_policy_records_feature_set(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(10)]), encoding="utf-8")

    metadata = run_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=10,
        seed=7,
        embedder_name="fake",
        feature_set="retrieval_only",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert metadata["feature_set"] == "retrieval_only"
    assert checkpoint["metadata"]["feature_set"] == "retrieval_only"


def test_run_scifact_retrieval_policy_experiment_uses_distinct_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "scifact"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    metadata = run_scifact_retrieval_policy_experiment(
        data_path=data_dir,
        output_dir=output_dir,
        num_train_examples=6,
        num_test_examples=4,
        seed=2,
        pool_size=4,
        embedder_name="fake",
        knn_k_candidates=[1, 3],
        tuning_folds=2,
    )

    assert metadata["pool_size"] == 4
    assert metadata["train_split"] == "train"
    assert metadata["test_split"] == "test"
    assert metadata["train_examples"] == 6
    assert metadata["test_examples"] == 4
    assert Path(metadata["outputs"]["summary_csv"]).name == "scifact_retrieval_policy_summary.csv"
    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert checkpoint["metadata"]["dataset"] == "BEIR SciFact retrieval-action policy"
    assert Path(metadata["outputs"]["checkpoint"]).name == "scifact_retrieval_policy.pkl"


def test_run_scifact_retrieval_policy_can_use_full_corpus(tmp_path: Path) -> None:
    data_dir = tmp_path / "scifact"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    metadata = run_scifact_retrieval_policy_experiment(
        data_path=data_dir,
        output_dir=output_dir,
        num_train_examples=6,
        num_test_examples=4,
        seed=2,
        pool_size=4,
        full_corpus=True,
        embedder_name="fake",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    detailed = Path(metadata["outputs"]["detailed_csv"]).read_text(encoding="utf-8")
    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert metadata["full_corpus"] is True
    assert metadata["corpus_size"] == 10
    assert checkpoint["metadata"]["full_corpus"] is True
    assert checkpoint["metadata"]["corpus_size"] == 10
    assert "d9" in detailed


def test_run_scifact_retrieval_policy_can_use_generated_actions_without_overwriting_main_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "scifact"
    output_dir = tmp_path / "outputs"
    cache_path = tmp_path / "generated_cache.jsonl"
    _write_scifact_fixture(data_dir, count=10)

    def provider(question: str, mode: str) -> list[str]:
        if mode == "multi_query":
            return [question, f"{question} evidence", f"{question} document"]
        if mode == "decompose":
            return [question, f"{question} evidence"]
        return [question]

    metadata = run_scifact_retrieval_policy_experiment(
        data_path=data_dir,
        output_dir=output_dir,
        num_train_examples=6,
        num_test_examples=4,
        seed=2,
        pool_size=4,
        embedder_name="fake",
        knn_k_candidates=[1],
        tuning_folds=2,
        generated_actions=True,
        llm_cache_path=cache_path,
        rewrite_provider=provider,
    )

    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    summary = Path(metadata["outputs"]["summary_csv"]).read_text(encoding="utf-8")
    assert metadata["generated_actions"] is True
    assert Path(metadata["outputs"]["summary_csv"]).name == "scifact_retrieval_policy_generated_summary.csv"
    assert Path(metadata["outputs"]["checkpoint"]).name == "scifact_retrieval_policy_generated.pkl"
    assert "bm25_hyde" in checkpoint["metadata"]["actions"]
    assert "hybrid_multi_query" in checkpoint["metadata"]["actions"]
    assert checkpoint["metadata"]["llm_cache_path"] == str(cache_path)
    assert "Gemini multi-query hybrid action" in summary
    assert cache_path.exists()


def test_run_nfcorpus_retrieval_policy_experiment_uses_distinct_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "nfcorpus"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    metadata = run_nfcorpus_retrieval_policy_experiment(
        data_path=data_dir,
        output_dir=output_dir,
        num_train_examples=6,
        num_test_examples=4,
        seed=2,
        pool_size=4,
        full_corpus=True,
        embedder_name="fake",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert metadata["dataset_key"] == "nfcorpus"
    assert metadata["full_corpus"] is True
    assert metadata["corpus_size"] == 10
    assert Path(metadata["outputs"]["summary_csv"]).name == "nfcorpus_retrieval_policy_summary.csv"
    assert Path(metadata["outputs"]["checkpoint"]).name == "nfcorpus_retrieval_policy.pkl"
    assert checkpoint["metadata"]["dataset"] == "BEIR NFCorpus retrieval-action policy"
    assert checkpoint["metadata"]["dataset_key"] == "nfcorpus"


def test_run_policy_learning_curve_writes_aggregate_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "nfcorpus"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    csv_path = run_policy_learning_curve(
        dataset="nfcorpus",
        data_path=data_dir,
        output_dir=output_dir,
        train_sizes=[2, 4],
        num_test_examples=4,
        seed=2,
        full_corpus=True,
        embedder_name="fake",
        policy_model="mlp",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    curve = pd.read_csv(csv_path)
    assert list(curve["train_size"]) == [2, 4]
    assert set(curve["selected_policy_model"]) == {"mlp"}
    assert "selective_reward" in curve.columns
    assert "best_fixed_reward" in curve.columns


def test_run_feature_ablation_writes_aggregate_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "nfcorpus"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    csv_path = run_feature_ablation(
        dataset="nfcorpus",
        data_path=data_dir,
        output_dir=output_dir,
        feature_sets=["full", "retrieval_only"],
        num_train_examples=4,
        num_test_examples=4,
        seed=2,
        full_corpus=True,
        embedder_name="fake",
        policy_model="ridge",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    ablation = pd.read_csv(csv_path)
    assert list(ablation["feature_set"]) == ["full", "retrieval_only"]
    assert "selected_policy_model" in ablation.columns
    assert "validation_reward" in ablation.columns
    assert "best_fixed_validation_reward" in ablation.columns
    assert "validation_reward_gap_vs_best_fixed" in ablation.columns
    assert "selective_reward" in ablation.columns
    assert "best_fixed_reward" in ablation.columns
    assert "best_fixed_retrieval_calls" in ablation.columns


def test_run_policy_model_sweep_writes_aggregate_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "nfcorpus"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    csv_path = run_policy_model_sweep(
        dataset="nfcorpus",
        data_path=data_dir,
        output_dir=output_dir,
        policy_models=["ridge_sweep", "extra_trees"],
        feature_set="full",
        num_train_examples=4,
        num_test_examples=4,
        seed=2,
        full_corpus=True,
        embedder_name="fake",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    sweep = pd.read_csv(csv_path)
    assert list(sweep["policy_model"]) == ["ridge_sweep", "extra_trees"]
    assert sweep.loc[sweep["policy_model"] == "ridge_sweep", "selected_policy_model"].iloc[0].startswith("ridge_l2=")
    assert sweep.loc[sweep["policy_model"] == "extra_trees", "selected_policy_model"].iloc[0] == "extra_trees"
    assert "validation_reward" in sweep.columns
    assert "best_fixed_validation_reward" in sweep.columns
    assert "validation_reward_gap_vs_best_fixed" in sweep.columns
    assert "selective_reward" in sweep.columns
    assert "best_fixed_reward" in sweep.columns
    assert "best_fixed_retrieval_calls" in sweep.columns
    assert "oracle_reward" in sweep.columns
    first = sweep.iloc[0]
    assert round(float(first["validation_reward_gap_vs_best_fixed"]), 12) == round(
        float(first["validation_reward"] - first["best_fixed_validation_reward"]),
        12,
    )


def test_run_policy_model_sweep_records_auto_candidate_models(tmp_path: Path) -> None:
    data_dir = tmp_path / "nfcorpus"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    csv_path = run_policy_model_sweep(
        dataset="nfcorpus",
        data_path=data_dir,
        output_dir=output_dir,
        policy_models=["auto"],
        feature_set="full",
        num_train_examples=4,
        num_test_examples=4,
        seed=2,
        full_corpus=True,
        embedder_name="fake",
        knn_k_candidates=[1],
        tuning_folds=2,
        auto_candidate_models=["knn", "ridge"],
    )

    sweep = pd.read_csv(csv_path)
    assert list(sweep["policy_model"]) == ["auto"]
    assert list(sweep["auto_candidate_models"]) == ["knn,ridge"]
    assert sweep["selected_policy_model"].iloc[0] in {"knn_k=1", "ridge_l2=1.0"}


def test_run_policy_model_sweep_can_grid_feature_sets_and_policy_models(tmp_path: Path) -> None:
    data_dir = tmp_path / "nfcorpus"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    csv_path = run_policy_model_sweep(
        dataset="nfcorpus",
        data_path=data_dir,
        output_dir=output_dir,
        policy_models=["ridge", "extra_trees"],
        feature_sets=["full", "retrieval_only"],
        num_train_examples=4,
        num_test_examples=4,
        seed=2,
        full_corpus=True,
        embedder_name="fake",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    sweep = pd.read_csv(csv_path)
    assert list(zip(sweep["feature_set"], sweep["policy_model"])) == [
        ("full", "ridge"),
        ("full", "extra_trees"),
        ("retrieval_only", "ridge"),
        ("retrieval_only", "extra_trees"),
    ]
    assert "feature_width" in sweep.columns
    assert "validation_reward" in sweep.columns
    assert "best_fixed_validation_reward" in sweep.columns
    assert "validation_reward_gap_vs_best_fixed" in sweep.columns


def test_run_policy_model_sweep_accepts_semantic_zscore_feature_set(tmp_path: Path) -> None:
    data_dir = tmp_path / "nfcorpus"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    csv_path = run_policy_model_sweep(
        dataset="nfcorpus",
        data_path=data_dir,
        output_dir=output_dir,
        policy_models=["ridge"],
        feature_sets=["semantic_zscore"],
        num_train_examples=4,
        num_test_examples=4,
        seed=2,
        full_corpus=True,
        embedder_name="fake",
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    sweep = pd.read_csv(csv_path)
    assert list(sweep["feature_set"]) == ["semantic_zscore"]
    assert sweep["feature_width"].iloc[0] == 14


def test_run_policy_model_sweep_can_enable_retrieval_contrast_features(tmp_path: Path) -> None:
    data_dir = tmp_path / "nfcorpus"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    csv_path = run_policy_model_sweep(
        dataset="nfcorpus",
        data_path=data_dir,
        output_dir=output_dir,
        policy_models=["ridge"],
        feature_sets=["full", "contrast_only"],
        num_train_examples=4,
        num_test_examples=4,
        seed=2,
        full_corpus=True,
        embedder_name="fake",
        knn_k_candidates=[1],
        tuning_folds=2,
        retrieval_contrast_features=True,
    )

    sweep = pd.read_csv(csv_path)
    diagnostics = pd.read_csv(output_dir / "results" / "nfcorpus_policy_feature_diagnostics.csv")
    reward_diagnostics = pd.read_csv(output_dir / "results" / "nfcorpus_policy_feature_reward_diagnostics.csv")
    predictive_diagnostics = pd.read_csv(
        output_dir / "results" / "nfcorpus_policy_feature_predictive_diagnostics.csv"
    )
    assert list(sweep["feature_set"]) == ["full", "contrast_only"]
    assert list(sweep["feature_width"]) == [22, 22]
    assert list(sweep["retrieval_contrast_features"]) == [True, True]
    assert set(diagnostics["split"]) == {"train", "test"}
    assert "retrieval_contrast" in set(diagnostics["group"])
    assert set(reward_diagnostics["split"]) == {"train", "test"}
    assert "dense_advantage_vs_bm25" in set(reward_diagnostics["target"])
    assert "dense_advantage_vs_bm25" in set(predictive_diagnostics["target"])


def test_run_policy_model_sweep_records_semantic_depth(tmp_path: Path) -> None:
    data_dir = tmp_path / "nfcorpus"
    output_dir = tmp_path / "outputs"
    _write_scifact_fixture(data_dir, count=10)

    csv_path = run_policy_model_sweep(
        dataset="nfcorpus",
        data_path=data_dir,
        output_dir=output_dir,
        policy_models=["ridge"],
        feature_set="full",
        num_train_examples=4,
        num_test_examples=4,
        seed=2,
        full_corpus=True,
        embedder_name="fake",
        knn_k_candidates=[1],
        tuning_folds=2,
        semantic_depth=8,
    )

    sweep = pd.read_csv(csv_path)
    assert list(sweep["semantic_depth"]) == [8]


def test_export_qualitative_examples_writes_cases(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "qualitative.csv"
    pd.DataFrame(
        [
            _detailed_row("q1", "Selective retrieval policy", "dense_keep", 1.0, "d1 | d2"),
            _detailed_row("q1", "Train-best retrieval action", "hybrid_keyword", 0.8, "d3 | d1"),
            _detailed_row("q1", "Oracle retrieval action", "dense_keep", 1.2, "d1 | d2"),
            _detailed_row("q2", "Selective retrieval policy", "hybrid_keyword", 0.9, "d5 | d6"),
            _detailed_row("q2", "Train-best retrieval action", "dense_keep", 0.7, "d6 | d7"),
            _detailed_row("q2", "Oracle retrieval action", "hybrid_keyword", 1.0, "d5 | d6"),
        ]
    ).to_csv(detailed_csv, index=False)

    exported = export_qualitative_examples(
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        dataset="toy",
        max_examples_per_case=2,
    )

    cases = pd.read_csv(exported)
    assert set(cases["case_type"]) >= {
        "policy_beats_train_best",
        "policy_uses_dense",
        "policy_uses_hybrid",
        "policy_avoids_hybrid",
    }
    assert set(cases["dataset"]) == {"toy"}
    assert "policy_top_docs" in cases.columns


def _example(i: int) -> dict[str, object]:
    left = f"Entity{i}A"
    right = f"Entity{i}B"
    return {
        "_id": f"q{i}",
        "answer": left,
        "question": f"Which author lived longer, {left} or {right}?",
        "supporting_facts": [[left, 0], [right, 0]],
        "context": [
            [left, [f"{left} was a writer with a long biography."]],
            [right, [f"{right} was another writer with a long biography."]],
            [f"Distractor{i}", ["This paragraph is unrelated."]],
        ],
        "type": "comparison",
        "level": "easy",
    }


def _write_scifact_fixture(data_dir: Path, count: int) -> None:
    qrels_dir = data_dir / "qrels"
    qrels_dir.mkdir(parents=True)
    corpus = []
    queries = []
    qrels = ["query-id\tcorpus-id\tscore"]
    for i in range(count):
        corpus.append(f'{{"_id": "d{i}", "title": "Doc {i}", "text": "evidence token{i}"}}')
        queries.append(f'{{"_id": "q{i}", "text": "claim token{i}"}}')
        qrels.append(f"q{i}\td{i}\t1")
    (data_dir / "corpus.jsonl").write_text("\n".join(corpus), encoding="utf-8")
    (data_dir / "queries.jsonl").write_text("\n".join(queries), encoding="utf-8")
    (qrels_dir / "test.tsv").write_text("\n".join(qrels), encoding="utf-8")
    (qrels_dir / "train.tsv").write_text("\n".join(qrels), encoding="utf-8")


def _detailed_row(qid: str, method: str, action: str, reward: float, top_docs: str) -> dict[str, object]:
    return {
        "split": "test",
        "method": method,
        "action": action,
        "qid": qid,
        "question": f"question for {qid}",
        "recall_at_5": reward,
        "mrr": reward,
        "ndcg_at_5": reward,
        "reward": reward,
        "rewrite_cost": 0.0,
        "retrieval_calls": 1,
        "rewrite_tokens": 3,
        "queries": f"query {qid}",
        "top_docs": top_docs,
        "gold_docs": top_docs.split(" | ")[0],
    }


class _FakeSemanticEmbedder:
    def embed_text(self, text: str, task_type: str) -> np.ndarray:
        return self.embed_texts([text], task_type=task_type)[0]

    def embed_texts(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        return [_normalized([len(text), text.count(" "), 1.0 if task_type == "RETRIEVAL_QUERY" else 0.5]) for text in texts]


class _RankAgreementEmbedder:
    def embed_text(self, text: str, task_type: str) -> np.ndarray:
        return self.embed_texts([text], task_type=task_type)[0]

    def embed_texts(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        vectors = {
            "Which author lived longer, Entity1A or Entity1B?": [1.0, 0.0, 0.0],
            "alpha": [0.0, 1.0, 0.0],
            "bravo": [1.0, 0.0, 0.0],
            "charlie": [0.8, 0.2, 0.0],
        }
        return [np.asarray(vectors[text], dtype=float) for text in texts]


class _FixedSimilarityEmbedder:
    def __init__(self, similarities: np.ndarray) -> None:
        self.similarities = np.asarray(similarities, dtype=float)

    def embed_text(self, text: str, task_type: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)

    def embed_texts(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        if task_type != "RETRIEVAL_DOCUMENT":
            return [self.embed_text(text, task_type) for text in texts]
        return [np.asarray([similarity, 0.0], dtype=float) for similarity in self.similarities[: len(texts)]]


class _TrackingSemanticEmbedder:
    def __init__(self) -> None:
        self.document_batch_sizes: list[int] = []

    def embed_text(self, text: str, task_type: str) -> np.ndarray:
        return self.embed_texts([text], task_type=task_type)[0]

    def embed_texts(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        if task_type == "RETRIEVAL_DOCUMENT":
            self.document_batch_sizes.append(len(texts))
        return [_normalized([len(text), text.count(" "), 1.0 if task_type == "RETRIEVAL_QUERY" else 0.5]) for text in texts]


def _normalized(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    return vector / np.linalg.norm(vector)
