from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from selective_rag_rl.policy_model_sweep import run_policy_model_sweep
from selective_rag_rl.selection_diagnostics import export_selection_diagnostics
from selective_rag_rl.selection_stability import export_selection_stability

SweepRunner = Callable[..., Path]


def run_repeated_policy_model_selection(
    *,
    dataset: str,
    data_path: Path,
    output_dir: Path,
    seeds: list[int],
    policy_models: list[str] | None = None,
    feature_set: str = "full",
    feature_sets: list[str] | None = None,
    num_train_examples: int = 300,
    num_test_examples: int = 300,
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
    semantic_depth: int = 5,
    sweep_runner: SweepRunner = run_policy_model_sweep,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("At least one seed is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    grid_csvs: list[Path] = []
    manifest_rows = []
    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        grid_csv = sweep_runner(
            dataset=dataset,
            data_path=data_path,
            output_dir=run_dir,
            policy_models=policy_models,
            feature_set=feature_set,
            feature_sets=feature_sets,
            num_train_examples=num_train_examples,
            num_test_examples=num_test_examples,
            seed=seed,
            full_corpus=full_corpus,
            k=k,
            pool_size=pool_size,
            embedder_name=embedder_name,
            dense_weight=dense_weight,
            retrieval_call_cost=retrieval_call_cost,
            semantic_features=semantic_features,
            semantic_cache_path=semantic_cache_path,
            semantic_allow_api=semantic_allow_api,
            semantic_max_new_texts=semantic_max_new_texts,
            semantic_dry_run=semantic_dry_run,
            knn_k_candidates=knn_k_candidates,
            tuning_folds=tuning_folds,
            auto_candidate_models=auto_candidate_models,
            retrieval_contrast_features=retrieval_contrast_features,
            semantic_depth=semantic_depth,
        )
        grid_csvs.append(Path(grid_csv))
        manifest_rows.append({"seed": int(seed), "grid_csv": str(grid_csv)})

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = results_dir / f"{dataset}_repeated_selection_manifest.csv"
    diagnostics_csv = results_dir / f"{dataset}_repeated_selection_diagnostics.csv"
    stability_csv = results_dir / f"{dataset}_repeated_selection_stability.csv"
    metadata_json = results_dir / f"{dataset}_repeated_selection_metadata.json"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    diagnostics_rows = []
    for seed, grid_csv in zip(seeds, grid_csvs):
        tmp_csv = results_dir / f".{dataset}_repeated_selection_seed_{seed}.diagnostics.tmp.csv"
        try:
            export_selection_diagnostics(
                grid_csv,
                tmp_csv,
                dataset=f"{dataset}_repeated_selection_seed_{seed}",
            )
            row = pd.read_csv(tmp_csv).iloc[0].to_dict()
            diagnostics_rows.append({"seed": int(seed), "grid_csv": str(grid_csv), **row})
        finally:
            tmp_csv.unlink(missing_ok=True)
    pd.DataFrame(diagnostics_rows).to_csv(diagnostics_csv, index=False)
    export_selection_stability(grid_csvs, stability_csv, dataset=f"{dataset}_repeated_selection")

    metadata = {
        "dataset": dataset,
        "seeds": [int(seed) for seed in seeds],
        "policy_models": policy_models,
        "feature_set": feature_set,
        "feature_sets": feature_sets,
        "num_train_examples": int(num_train_examples),
        "num_test_examples": int(num_test_examples),
        "full_corpus": bool(full_corpus),
        "semantic_features": semantic_features,
        "semantic_depth": int(semantic_depth),
        "auto_candidate_models": auto_candidate_models,
        "retrieval_contrast_features": retrieval_contrast_features,
        "manifest_csv": str(manifest_csv),
        "diagnostics_csv": str(diagnostics_csv),
        "stability_csv": str(stability_csv),
        "grid_csvs": [str(path) for path in grid_csvs],
        "metadata_json": str(metadata_json),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
