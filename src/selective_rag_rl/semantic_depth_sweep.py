from __future__ import annotations

import json
from collections.abc import Callable
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.policy_model_sweep import run_policy_model_sweep
from selective_rag_rl.retrieval_policy_experiment import _validate_semantic_depth
from selective_rag_rl.selection_diagnostics import export_selection_diagnostics

SweepRunner = Callable[..., Path]
DepthSweepRunner = Callable[..., dict[str, object]]

DEPTH_EFFECT_COLUMNS = [
    "dataset",
    "feature_set",
    "policy_model",
    "semantic_depth",
    "baseline_semantic_depth",
    "n_pairs",
    "validation_reward_delta_mean",
    "validation_reward_delta_std",
    "validation_reward_win_rate",
    "selective_reward_delta_mean",
    "selective_reward_delta_std",
    "selective_reward_win_rate",
    "selective_recall_at_5_delta_mean",
    "selective_recall_at_5_delta_std",
    "selective_retrieval_calls_delta_mean",
    "selective_retrieval_calls_delta_std",
]

PREDICTIVE_EFFECT_COLUMNS = [
    "dataset",
    "group",
    "target",
    "semantic_depth",
    "baseline_semantic_depth",
    "n_pairs",
    "train_r2_delta",
    "test_r2_delta",
    "test_corr_delta",
    "active_columns_delta",
    "width_delta",
]

METRICS = [
    "validation_reward",
    "selective_reward",
    "selective_recall_at_5",
    "selective_retrieval_calls",
]

DEPTH_STABILITY_METRICS = [
    "validation_reward_delta_mean",
    "selective_reward_delta_mean",
    "selective_recall_at_5_delta_mean",
    "selective_retrieval_calls_delta_mean",
]

DEPTH_SELECTION_STABILITY_COLUMNS = [
    "dataset",
    "feature_set",
    "policy_model",
    "baseline_semantic_depth",
    "n_runs",
    "n_unique_validation_best_depths",
    "validation_best_depth_modal",
    "validation_best_depth_modal_count",
    "validation_best_depth_modal_share",
    "n_unique_heldout_best_depths",
    "heldout_best_depth_modal",
    "heldout_best_depth_modal_count",
    "heldout_best_depth_modal_share",
    "validation_matches_heldout_depth_rate",
    "validation_matches_heldout_depth_rate_ci_low",
    "validation_matches_heldout_depth_rate_ci_high",
    "validation_selected_heldout_delta_mean",
    "heldout_best_delta_mean",
    "depth_selection_reward_gap_mean",
    "depth_selection_reward_gap_std",
]

PREDICTIVE_STABILITY_METRICS = [
    "train_r2_delta",
    "test_r2_delta",
    "test_corr_delta",
]


def run_semantic_depth_sweep(
    *,
    dataset: str,
    data_path: Path,
    output_dir: Path,
    semantic_depths: list[int],
    baseline_semantic_depth: int | None = None,
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
    sweep_runner: SweepRunner = run_policy_model_sweep,
) -> dict[str, object]:
    depths = [_validate_semantic_depth(depth) for depth in semantic_depths]
    if not depths:
        raise ValueError("At least one semantic depth is required")
    if len(set(depths)) != len(depths):
        raise ValueError("Semantic depths must be unique")
    baseline_depth = _validate_semantic_depth(baseline_semantic_depth or depths[0])
    if baseline_depth not in depths:
        raise ValueError("Baseline semantic depth must be included in semantic_depths")

    output_dir.mkdir(parents=True, exist_ok=True)
    grid_frames = []
    predictive_frames = []
    manifest_rows = []
    for depth in depths:
        run_dir = output_dir / f"depth_{depth}"
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
            semantic_depth=depth,
        )
        grid = pd.read_csv(grid_csv)
        grid["semantic_depth"] = depth
        grid_frames.append(grid)
        predictive_csv = Path(grid_csv).with_name(f"{dataset}_policy_feature_predictive_diagnostics.csv")
        if predictive_csv.exists():
            predictive = pd.read_csv(predictive_csv)
            predictive["semantic_depth"] = depth
            predictive_frames.append(predictive)
        manifest_rows.append({"semantic_depth": depth, "grid_csv": str(grid_csv)})

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = results_dir / f"{dataset}_semantic_depth_sweep_manifest.csv"
    combined_csv = results_dir / f"{dataset}_semantic_depth_sweep.csv"
    depth_effects_csv = results_dir / f"{dataset}_semantic_depth_effects.csv"
    selection_diagnostics_csv = results_dir / f"{dataset}_semantic_depth_selection_diagnostics.csv"
    predictive_diagnostics_csv = results_dir / f"{dataset}_semantic_depth_predictive_diagnostics.csv"
    predictive_effects_csv = results_dir / f"{dataset}_semantic_depth_predictive_effects.csv"
    metadata_json = results_dir / f"{dataset}_semantic_depth_sweep_metadata.json"

    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    combined = pd.concat(grid_frames, ignore_index=True)
    combined.to_csv(combined_csv, index=False)
    export_selection_diagnostics(combined_csv, selection_diagnostics_csv, dataset=f"{dataset}_semantic_depth_sweep")
    _depth_effects_frame(combined, dataset=dataset, baseline_semantic_depth=baseline_depth).to_csv(
        depth_effects_csv,
        index=False,
    )
    if predictive_frames:
        predictive_diagnostics = pd.concat(predictive_frames, ignore_index=True)
        predictive_diagnostics.to_csv(predictive_diagnostics_csv, index=False)
        _predictive_effects_frame(
            predictive_diagnostics,
            dataset=dataset,
            baseline_semantic_depth=baseline_depth,
        ).to_csv(predictive_effects_csv, index=False)
    else:
        pd.DataFrame().to_csv(predictive_diagnostics_csv, index=False)
        pd.DataFrame(columns=PREDICTIVE_EFFECT_COLUMNS).to_csv(predictive_effects_csv, index=False)

    metadata = {
        "dataset": dataset,
        "semantic_depths": depths,
        "baseline_semantic_depth": baseline_depth,
        "policy_models": policy_models,
        "feature_set": feature_set,
        "feature_sets": feature_sets,
        "num_train_examples": int(num_train_examples),
        "num_test_examples": int(num_test_examples),
        "semantic_features": semantic_features,
        "retrieval_contrast_features": retrieval_contrast_features,
        "manifest_csv": str(manifest_csv),
        "combined_csv": str(combined_csv),
        "depth_effects_csv": str(depth_effects_csv),
        "selection_diagnostics_csv": str(selection_diagnostics_csv),
        "predictive_diagnostics_csv": str(predictive_diagnostics_csv),
        "predictive_effects_csv": str(predictive_effects_csv),
        "metadata_json": str(metadata_json),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_repeated_semantic_depth_sweep(
    *,
    dataset: str,
    data_path: Path,
    output_dir: Path,
    seeds: list[int],
    semantic_depths: list[int],
    baseline_semantic_depth: int | None = None,
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
    depth_sweep_runner: DepthSweepRunner = run_semantic_depth_sweep,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("At least one seed is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    depth_effect_frames = []
    predictive_effect_frames = []
    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        metadata = depth_sweep_runner(
            dataset=dataset,
            data_path=data_path,
            output_dir=run_dir,
            semantic_depths=semantic_depths,
            baseline_semantic_depth=baseline_semantic_depth,
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
        )
        depth_effects_csv = Path(str(metadata["depth_effects_csv"]))
        predictive_effects_csv = Path(str(metadata["predictive_effects_csv"]))
        depth_effects = pd.read_csv(depth_effects_csv)
        depth_effects["seed"] = int(seed)
        depth_effect_frames.append(depth_effects)
        predictive_effects = pd.read_csv(predictive_effects_csv)
        predictive_effects["seed"] = int(seed)
        predictive_effect_frames.append(predictive_effects)
        manifest_rows.append(
            {
                "seed": int(seed),
                "depth_effects_csv": str(depth_effects_csv),
                "predictive_effects_csv": str(predictive_effects_csv),
            }
        )

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = results_dir / f"{dataset}_repeated_semantic_depth_manifest.csv"
    depth_effects_csv = results_dir / f"{dataset}_repeated_semantic_depth_effects.csv"
    depth_stability_csv = results_dir / f"{dataset}_repeated_semantic_depth_stability.csv"
    depth_selection_diagnostics_csv = results_dir / f"{dataset}_repeated_semantic_depth_selection_diagnostics.csv"
    depth_selection_stability_csv = results_dir / f"{dataset}_repeated_semantic_depth_selection_stability.csv"
    predictive_effects_csv = results_dir / f"{dataset}_repeated_semantic_depth_predictive_effects.csv"
    predictive_stability_csv = results_dir / f"{dataset}_repeated_semantic_depth_predictive_stability.csv"
    metadata_json = results_dir / f"{dataset}_repeated_semantic_depth_metadata.json"

    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    depth_effects_all = pd.concat(depth_effect_frames, ignore_index=True)
    predictive_effects_all = pd.concat(predictive_effect_frames, ignore_index=True)
    depth_effects_all.to_csv(depth_effects_csv, index=False)
    predictive_effects_all.to_csv(predictive_effects_csv, index=False)
    _depth_stability_frame(depth_effects_all, dataset=f"{dataset}_repeated_semantic_depth").to_csv(
        depth_stability_csv,
        index=False,
    )
    _depth_selection_diagnostics_frame(
        depth_effects_all,
        dataset=f"{dataset}_repeated_semantic_depth",
    ).to_csv(depth_selection_diagnostics_csv, index=False)
    _depth_selection_stability_frame(
        depth_effects_all,
        dataset=f"{dataset}_repeated_semantic_depth",
    ).to_csv(depth_selection_stability_csv, index=False)
    _predictive_stability_frame(
        predictive_effects_all,
        dataset=f"{dataset}_repeated_semantic_depth",
    ).to_csv(predictive_stability_csv, index=False)

    metadata = {
        "dataset": dataset,
        "seeds": [int(seed) for seed in seeds],
        "semantic_depths": [_validate_semantic_depth(depth) for depth in semantic_depths],
        "baseline_semantic_depth": baseline_semantic_depth,
        "policy_models": policy_models,
        "feature_set": feature_set,
        "feature_sets": feature_sets,
        "num_train_examples": int(num_train_examples),
        "num_test_examples": int(num_test_examples),
        "semantic_features": semantic_features,
        "retrieval_contrast_features": retrieval_contrast_features,
        "manifest_csv": str(manifest_csv),
        "depth_effects_csv": str(depth_effects_csv),
        "depth_stability_csv": str(depth_stability_csv),
        "depth_selection_diagnostics_csv": str(depth_selection_diagnostics_csv),
        "depth_selection_stability_csv": str(depth_selection_stability_csv),
        "predictive_effects_csv": str(predictive_effects_csv),
        "predictive_stability_csv": str(predictive_stability_csv),
        "metadata_json": str(metadata_json),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _depth_effects_frame(
    grid: pd.DataFrame,
    dataset: str,
    baseline_semantic_depth: int,
) -> pd.DataFrame:
    pair_rows = []
    baseline_rows = grid[grid["semantic_depth"] == baseline_semantic_depth]
    for _, row in grid[grid["semantic_depth"] != baseline_semantic_depth].iterrows():
        matches = baseline_rows[
            (baseline_rows["feature_set"] == row["feature_set"])
            & (baseline_rows["policy_model"] == row["policy_model"])
        ]
        if matches.empty:
            continue
        baseline = matches.iloc[0]
        paired = {
            "feature_set": row["feature_set"],
            "policy_model": row["policy_model"],
            "semantic_depth": int(row["semantic_depth"]),
            "baseline_semantic_depth": int(baseline_semantic_depth),
        }
        for metric in METRICS:
            paired[f"{metric}_delta"] = round(float(row[metric]) - float(baseline[metric]), 12)
        pair_rows.append(paired)
    if not pair_rows:
        raise ValueError(f"No comparable semantic-depth rows found against baseline depth {baseline_semantic_depth}")

    pairs = pd.DataFrame(pair_rows)
    summaries = []
    for (feature_set, policy_model, semantic_depth), group in pairs.groupby(
        ["feature_set", "policy_model", "semantic_depth"],
        sort=True,
    ):
        summaries.append(
            _summary_row(dataset, feature_set, policy_model, int(semantic_depth), baseline_semantic_depth, group)
        )
    return pd.DataFrame(summaries, columns=DEPTH_EFFECT_COLUMNS)


def _depth_stability_frame(effects: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for (feature_set, policy_model, semantic_depth, baseline_depth), group in effects.groupby(
        ["feature_set", "policy_model", "semantic_depth", "baseline_semantic_depth"],
        sort=True,
    ):
        row = {
            "dataset": dataset,
            "feature_set": feature_set,
            "policy_model": policy_model,
            "semantic_depth": int(semantic_depth),
            "baseline_semantic_depth": int(baseline_depth),
            "n_runs": int(group["seed"].nunique()) if "seed" in group else int(len(group)),
        }
        for metric in DEPTH_STABILITY_METRICS:
            values = group[metric]
            prefix = metric.removesuffix("_mean")
            row[f"{prefix}_across_seed_mean"] = float(values.mean())
            row[f"{prefix}_across_seed_std"] = float(values.std(ddof=0)) if len(values) else 0.0
            ci_low, ci_high = _bootstrap_mean_interval(values)
            row[f"{prefix}_ci_low"] = ci_low
            row[f"{prefix}_ci_high"] = ci_high
            if metric in {"validation_reward_delta_mean", "selective_reward_delta_mean"}:
                row[f"{prefix}_win_rate"] = float((values > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _depth_selection_stability_frame(effects: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for (feature_set, policy_model, baseline_depth), group in effects.groupby(
        ["feature_set", "policy_model", "baseline_semantic_depth"],
        sort=True,
    ):
        choices = [_depth_choice_row(seed_group) for _, seed_group in group.groupby("seed", sort=True)]
        choices_frame = pd.DataFrame(choices)
        validation_modal, validation_count = _modal_depth(choices_frame["validation_best_depth"])
        heldout_modal, heldout_count = _modal_depth(choices_frame["heldout_best_depth"])
        n_runs = len(choices_frame)
        reward_gaps = choices_frame["depth_selection_reward_gap"]
        matches = int((choices_frame["validation_best_depth"] == choices_frame["heldout_best_depth"]).sum())
        match_ci_low, match_ci_high = _wilson_interval(matches, n_runs)
        rows.append(
            {
                "dataset": dataset,
                "feature_set": feature_set,
                "policy_model": policy_model,
                "baseline_semantic_depth": int(baseline_depth),
                "n_runs": int(n_runs),
                "n_unique_validation_best_depths": int(choices_frame["validation_best_depth"].nunique()),
                "validation_best_depth_modal": int(validation_modal),
                "validation_best_depth_modal_count": int(validation_count),
                "validation_best_depth_modal_share": validation_count / n_runs,
                "n_unique_heldout_best_depths": int(choices_frame["heldout_best_depth"].nunique()),
                "heldout_best_depth_modal": int(heldout_modal),
                "heldout_best_depth_modal_count": int(heldout_count),
                "heldout_best_depth_modal_share": heldout_count / n_runs,
                "validation_matches_heldout_depth_rate": matches / n_runs,
                "validation_matches_heldout_depth_rate_ci_low": match_ci_low,
                "validation_matches_heldout_depth_rate_ci_high": match_ci_high,
                "validation_selected_heldout_delta_mean": float(
                    choices_frame["validation_selected_heldout_delta"].mean()
                ),
                "heldout_best_delta_mean": float(choices_frame["heldout_best_delta"].mean()),
                "depth_selection_reward_gap_mean": float(reward_gaps.mean()),
                "depth_selection_reward_gap_std": float(reward_gaps.std(ddof=0)) if len(reward_gaps) else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=DEPTH_SELECTION_STABILITY_COLUMNS)


def _depth_selection_diagnostics_frame(effects: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for (seed, feature_set, policy_model, baseline_depth), group in effects.groupby(
        ["seed", "feature_set", "policy_model", "baseline_semantic_depth"],
        sort=True,
    ):
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "feature_set": feature_set,
                "policy_model": policy_model,
                "baseline_semantic_depth": int(baseline_depth),
                **_depth_choice_row(group),
            }
        )
    return pd.DataFrame(rows)


def _depth_choice_row(seed_group: pd.DataFrame) -> dict[str, object]:
    baseline_depth = int(seed_group["baseline_semantic_depth"].iloc[0])
    candidates = [
        {
            "semantic_depth": baseline_depth,
            "validation_reward_delta": 0.0,
            "selective_reward_delta": 0.0,
        }
    ]
    for _, row in seed_group.iterrows():
        candidates.append(
            {
                "semantic_depth": int(row["semantic_depth"]),
                "validation_reward_delta": float(row["validation_reward_delta_mean"]),
                "selective_reward_delta": float(row["selective_reward_delta_mean"]),
            }
        )
    candidates_frame = pd.DataFrame(candidates)
    validation_best = candidates_frame.sort_values(
        ["validation_reward_delta", "semantic_depth"],
        ascending=[False, True],
    ).iloc[0]
    heldout_best = candidates_frame.sort_values(
        ["selective_reward_delta", "semantic_depth"],
        ascending=[False, True],
    ).iloc[0]
    return {
        "validation_best_depth": int(validation_best["semantic_depth"]),
        "heldout_best_depth": int(heldout_best["semantic_depth"]),
        "validation_selected_heldout_delta": float(validation_best["selective_reward_delta"]),
        "heldout_best_delta": float(heldout_best["selective_reward_delta"]),
        "depth_selection_reward_gap": round(
            float(heldout_best["selective_reward_delta"]) - float(validation_best["selective_reward_delta"]),
            12,
        ),
    }


def _modal_depth(values: pd.Series) -> tuple[int, int]:
    counts = values.astype(int).value_counts()
    max_count = int(counts.max())
    modal_depth = int(counts[counts == max_count].sort_index().index[0])
    return modal_depth, max_count


def _bootstrap_mean_interval(values: pd.Series, ci: float = 0.95) -> tuple[float, float]:
    observed = [float(value) for value in values.dropna()]
    if not observed:
        return 0.0, 0.0
    if len(observed) == 1:
        return observed[0], observed[0]

    n = len(observed)
    if n**n <= 20000:
        means = [sum(sample) / n for sample in product(observed, repeat=n)]
    else:
        rng = np.random.default_rng(0)
        samples = rng.choice(np.asarray(observed, dtype=float), size=(5000, n), replace=True)
        means = samples.mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    return (
        round(float(np.quantile(means, alpha)), 12),
        round(float(np.quantile(means, 1.0 - alpha)), 12),
    )


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / denominator
    return round(max(0.0, center - margin), 12), round(min(1.0, center + margin), 12)


def _predictive_effects_frame(
    diagnostics: pd.DataFrame,
    dataset: str,
    baseline_semantic_depth: int,
) -> pd.DataFrame:
    pair_rows = []
    baseline_rows = diagnostics[diagnostics["semantic_depth"] == baseline_semantic_depth]
    for _, row in diagnostics[diagnostics["semantic_depth"] != baseline_semantic_depth].iterrows():
        matches = baseline_rows[
            (baseline_rows["group"] == row["group"])
            & (baseline_rows["target"] == row["target"])
        ]
        if matches.empty:
            continue
        baseline = matches.iloc[0]
        pair_rows.append(
            {
                "dataset": dataset,
                "group": row["group"],
                "target": row["target"],
                "semantic_depth": int(row["semantic_depth"]),
                "baseline_semantic_depth": int(baseline_semantic_depth),
                "train_r2_delta": round(float(row["train_r2"]) - float(baseline["train_r2"]), 12),
                "test_r2_delta": round(float(row["test_r2"]) - float(baseline["test_r2"]), 12),
                "test_corr_delta": round(float(row["test_corr"]) - float(baseline["test_corr"]), 12),
                "active_columns_delta": int(row["active_columns"]) - int(baseline["active_columns"]),
                "width_delta": int(row["width"]) - int(baseline["width"]),
            }
        )
    if not pair_rows:
        raise ValueError(
            f"No comparable semantic-depth predictive rows found against baseline depth {baseline_semantic_depth}"
        )

    pairs = pd.DataFrame(pair_rows)
    summaries = []
    for (group_name, target, semantic_depth), group in pairs.groupby(
        ["group", "target", "semantic_depth"],
        sort=True,
    ):
        summaries.append(
            {
                "dataset": dataset,
                "group": group_name,
                "target": target,
                "semantic_depth": int(semantic_depth),
                "baseline_semantic_depth": int(baseline_semantic_depth),
                "n_pairs": int(len(group)),
                "train_r2_delta": float(group["train_r2_delta"].mean()),
                "test_r2_delta": float(group["test_r2_delta"].mean()),
                "test_corr_delta": float(group["test_corr_delta"].mean()),
                "active_columns_delta": float(group["active_columns_delta"].mean()),
                "width_delta": float(group["width_delta"].mean()),
            }
        )
    return pd.DataFrame(summaries, columns=PREDICTIVE_EFFECT_COLUMNS)


def _predictive_stability_frame(effects: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for (group_name, target, semantic_depth, baseline_depth), group in effects.groupby(
        ["group", "target", "semantic_depth", "baseline_semantic_depth"],
        sort=True,
    ):
        row = {
            "dataset": dataset,
            "group": group_name,
            "target": target,
            "semantic_depth": int(semantic_depth),
            "baseline_semantic_depth": int(baseline_depth),
            "n_runs": int(group["seed"].nunique()) if "seed" in group else int(len(group)),
            "active_columns_delta_mean": float(group["active_columns_delta"].mean()),
            "width_delta_mean": float(group["width_delta"].mean()),
        }
        for metric in PREDICTIVE_STABILITY_METRICS:
            values = group[metric]
            row[f"{metric}_across_seed_mean"] = float(values.mean())
            row[f"{metric}_across_seed_std"] = float(values.std(ddof=0)) if len(values) else 0.0
            if metric in {"test_r2_delta", "test_corr_delta"}:
                row[f"{metric}_positive_rate"] = float((values > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_row(
    dataset: str,
    feature_set: str,
    policy_model: str,
    semantic_depth: int,
    baseline_semantic_depth: int,
    group: pd.DataFrame,
) -> dict[str, object]:
    row = {
        "dataset": dataset,
        "feature_set": feature_set,
        "policy_model": policy_model,
        "semantic_depth": int(semantic_depth),
        "baseline_semantic_depth": int(baseline_semantic_depth),
        "n_pairs": int(len(group)),
    }
    for metric in METRICS:
        deltas = group[f"{metric}_delta"]
        row[f"{metric}_delta_mean"] = float(deltas.mean())
        row[f"{metric}_delta_std"] = float(deltas.std(ddof=0)) if len(deltas) else 0.0
        if metric in {"validation_reward", "selective_reward"}:
            row[f"{metric}_win_rate"] = float((deltas > 0).mean())
    return row
