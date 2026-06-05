from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.core.data import load_beir_dataset
from selective_rag_rl.core.dense_retriever import load_sentence_transformer
from selective_rag_rl.diagnostics.bandit_replay import (
    export_bandit_replay_diagnostics_from_detailed_csv,
    export_bandit_replay_diagnostics_from_evals,
    write_bandit_replay_figure,
)
from selective_rag_rl.experiments.dense_experiment import FakeDenseEmbedder
from selective_rag_rl.experiments.retrieval_policy_experiment import (
    BASE_RETRIEVAL_ACTIONS,
    FEATURE_SET_CHOICES,
    SEMANTIC_DEPTH_DEFAULT,
    _load_semantic_embedder,
    _prewarm_semantic_embeddings,
    _validate_semantic_depth,
    evaluate_retrieval_actions,
    fit_policy_feature_transform,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run selected-action bandit replay diagnostics with cumulative regret curves.",
    )
    parser.add_argument("--dataset", choices=["scifact", "nfcorpus"], required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--detailed-csv",
        type=Path,
        default=None,
        help="Use an existing full-information retrieval-policy detailed CSV instead of recomputing retrieval actions.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-train-examples", type=int, default=300)
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
    parser.add_argument("--moving-average-window", type=int, default=20)
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="full")
    parser.add_argument("--semantic-features", choices=["none", "vertex"], default="none")
    parser.add_argument("--semantic-cache-path", type=Path, default=None)
    parser.add_argument("--semantic-allow-api", action="store_true")
    parser.add_argument("--semantic-max-new-texts", type=int, default=0)
    parser.add_argument("--semantic-dry-run", action="store_true")
    parser.add_argument("--semantic-depth", type=int, default=SEMANTIC_DEPTH_DEFAULT)
    args = parser.parse_args()

    if args.detailed_csv is not None:
        outputs = run_bandit_replay_from_detailed_csv(
            dataset=args.dataset,
            detailed_csv=args.detailed_csv,
            output_dir=args.output_dir,
            split=args.split,
            alpha=args.alpha,
            l2=args.l2,
            epsilon=args.epsilon,
            posterior_scale=args.posterior_scale,
            seed=args.seed,
            moving_average_window=args.moving_average_window,
        )
    else:
        data_path = args.data_path or Path("data/raw") / args.dataset
        outputs = run_bandit_replay_diagnostics(
            dataset=args.dataset,
            data_path=data_path,
            output_dir=args.output_dir,
            num_train_examples=args.num_train_examples,
            seed=args.seed,
            k=args.k,
            pool_size=args.pool_size,
            full_corpus=args.full_corpus,
            embedder_name=args.embedder,
            dense_weight=args.dense_weight,
            retrieval_call_cost=args.retrieval_call_cost,
            alpha=args.alpha,
            l2=args.l2,
            epsilon=args.epsilon,
            posterior_scale=args.posterior_scale,
            moving_average_window=args.moving_average_window,
            feature_set=args.feature_set,
            semantic_features=args.semantic_features,
            semantic_cache_path=args.semantic_cache_path,
            semantic_allow_api=args.semantic_allow_api,
            semantic_max_new_texts=args.semantic_max_new_texts,
            semantic_dry_run=args.semantic_dry_run,
            semantic_depth=args.semantic_depth,
        )
    print(pd.read_csv(outputs["summary_csv"]).to_string(index=False))
    print(f"history_csv={outputs['history_csv']}")
    print(f"figure_png={outputs['figure_png']}")


def run_bandit_replay_from_detailed_csv(
    *,
    dataset: str,
    detailed_csv: Path,
    output_dir: Path,
    split: str,
    alpha: float,
    l2: float,
    epsilon: float,
    posterior_scale: float,
    seed: int,
    moving_average_window: int,
) -> dict[str, Path]:
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    paths = export_bandit_replay_diagnostics_from_detailed_csv(
        detailed_csv=detailed_csv,
        history_csv=results_dir / f"{dataset}_bandit_replay_history.csv",
        summary_csv=results_dir / f"{dataset}_bandit_replay_summary.csv",
        dataset=dataset,
        split=split,
        alpha=alpha,
        l2=l2,
        epsilon=epsilon,
        posterior_scale=posterior_scale,
        seed=seed,
        moving_average_window=moving_average_window,
    )
    history = pd.read_csv(paths["history_csv"])
    figure_png = write_bandit_replay_figure(
        history,
        output_png=figures_dir / f"{dataset}_bandit_replay_regret.png",
    )
    return {**paths, "figure_png": figure_png}


def run_bandit_replay_diagnostics(
    *,
    dataset: str,
    data_path: Path,
    output_dir: Path,
    num_train_examples: int,
    seed: int,
    k: int,
    pool_size: int,
    full_corpus: bool,
    embedder_name: str,
    dense_weight: float,
    retrieval_call_cost: float,
    alpha: float,
    l2: float,
    epsilon: float,
    posterior_scale: float,
    moving_average_window: int,
    feature_set: str,
    semantic_features: str,
    semantic_cache_path: Path | None,
    semantic_allow_api: bool,
    semantic_max_new_texts: int,
    semantic_dry_run: bool,
    semantic_depth: int,
) -> dict[str, Path]:
    semantic_depth = _validate_semantic_depth(semantic_depth)
    examples = load_beir_dataset(
        data_path,
        num_examples=num_train_examples,
        seed=seed,
        split="train",
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
        _prewarm_semantic_embeddings(examples, semantic_embedder, semantic_depth)

    retriever_cache: dict[str, object] = {}
    train_evals = [
        evaluate_retrieval_actions(
            ex,
            embedder,
            k,
            dense_weight,
            retrieval_call_cost,
            semantic_embedder,
            actions=BASE_RETRIEVAL_ACTIONS,
            retriever_cache=retriever_cache,
            semantic_depth=semantic_depth,
        )
        for ex in tqdm(examples, desc=f"{dataset} bandit replay actions")
    ]
    feature_transform = fit_policy_feature_transform(np.vstack([row["features"] for row in train_evals]), feature_set)
    transformed = [
        {**row, "features": feature_transform.transform(row["features"])}
        for row in train_evals
    ]
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    paths = export_bandit_replay_diagnostics_from_evals(
        train_evals=transformed,
        actions=BASE_RETRIEVAL_ACTIONS,
        history_csv=results_dir / f"{dataset}_bandit_replay_history.csv",
        summary_csv=results_dir / f"{dataset}_bandit_replay_summary.csv",
        dataset=dataset,
        alpha=alpha,
        l2=l2,
        epsilon=epsilon,
        posterior_scale=posterior_scale,
        seed=seed,
        moving_average_window=moving_average_window,
    )
    history = pd.read_csv(paths["history_csv"])
    figure_png = write_bandit_replay_figure(
        history,
        output_png=figures_dir / f"{dataset}_bandit_replay_regret.png",
    )
    return {**paths, "figure_png": figure_png}


if __name__ == "__main__":
    main()
