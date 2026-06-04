from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from selective_rag_rl.experiments.semantic_depth_sweep import run_repeated_semantic_depth_sweep, run_semantic_depth_sweep


def test_run_semantic_depth_sweep_writes_combined_grid_and_depth_effects(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_path.mkdir()

    metadata = run_semantic_depth_sweep(
        dataset="nfcorpus",
        data_path=data_path,
        output_dir=output_dir,
        semantic_depths=[5, 8],
        policy_models=["ridge"],
        feature_sets=["full", "no_semantic"],
        num_train_examples=4,
        num_test_examples=3,
        sweep_runner=_fake_sweep_runner,
    )

    manifest = pd.read_csv(metadata["manifest_csv"])
    combined = pd.read_csv(metadata["combined_csv"])
    effects = pd.read_csv(metadata["depth_effects_csv"])
    selection = pd.read_csv(metadata["selection_diagnostics_csv"]).iloc[0]
    predictive = pd.read_csv(metadata["predictive_diagnostics_csv"])
    predictive_effects = pd.read_csv(metadata["predictive_effects_csv"])
    full_effect = effects[
        (effects["feature_set"] == "full")
        & (effects["policy_model"] == "ridge")
        & (effects["semantic_depth"] == 8)
    ].iloc[0]
    rank_effect = predictive_effects[
        (predictive_effects["group"] == "semantic_rank_profile")
        & (predictive_effects["target"] == "dense_advantage_vs_bm25")
        & (predictive_effects["semantic_depth"] == 8)
    ].iloc[0]

    assert metadata["semantic_depths"] == [5, 8]
    assert list(manifest["semantic_depth"]) == [5, 8]
    assert set(combined["semantic_depth"]) == {5, 8}
    assert set(combined["feature_set"]) == {"full", "no_semantic"}
    assert selection["validation_selected_config"] == "depth=8/full/ridge/ridge_l2=1.0"
    assert selection["heldout_best_config"] == "depth=8/full/ridge/ridge_l2=1.0"
    assert full_effect["baseline_semantic_depth"] == 5
    assert full_effect["selective_reward_delta_mean"] == 0.08
    assert full_effect["validation_reward_delta_mean"] == 0.05
    assert set(predictive["semantic_depth"]) == {5, 8}
    assert rank_effect["test_r2_delta"] == 0.03
    assert rank_effect["test_corr_delta"] == 0.1


def test_run_semantic_depth_sweep_requires_depths() -> None:
    with pytest.raises(ValueError, match="semantic depth"):
        run_semantic_depth_sweep(
            dataset="nfcorpus",
            data_path=Path("data"),
            output_dir=Path("outputs"),
            semantic_depths=[],
            sweep_runner=_fake_sweep_runner,
        )


def test_run_repeated_semantic_depth_sweep_summarizes_depth_stability(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_path.mkdir()

    metadata = run_repeated_semantic_depth_sweep(
        dataset="nfcorpus",
        data_path=data_path,
        output_dir=output_dir,
        seeds=[1, 2],
        semantic_depths=[5, 8],
        policy_models=["ridge"],
        feature_sets=["full"],
        num_train_examples=4,
        num_test_examples=3,
        depth_sweep_runner=_fake_depth_sweep_runner,
    )

    manifest = pd.read_csv(metadata["manifest_csv"])
    stability = pd.read_csv(metadata["depth_stability_csv"]).iloc[0]
    selection_diagnostics = pd.read_csv(metadata["depth_selection_diagnostics_csv"])
    selection_stability = pd.read_csv(metadata["depth_selection_stability_csv"]).iloc[0]
    predictive_stability = pd.read_csv(metadata["predictive_stability_csv"]).iloc[0]

    assert list(manifest["seed"]) == [1, 2]
    assert list(selection_diagnostics["seed"]) == [1, 2]
    assert list(selection_diagnostics["validation_best_depth"]) == [8, 8]
    assert list(selection_diagnostics["heldout_best_depth"]) == [8, 5]
    assert list(selection_diagnostics["depth_selection_reward_gap"]) == [0.0, 0.02]
    assert stability["dataset"] == "nfcorpus_repeated_semantic_depth"
    assert stability["feature_set"] == "full"
    assert stability["policy_model"] == "ridge"
    assert stability["n_runs"] == 2
    assert stability["selective_reward_delta_across_seed_mean"] == 0.03
    assert stability["selective_reward_delta_ci_low"] <= 0.03 <= stability["selective_reward_delta_ci_high"]
    assert stability["validation_reward_delta_ci_low"] <= 0.025 <= stability["validation_reward_delta_ci_high"]
    assert stability["selective_reward_delta_win_rate"] == 0.5
    assert selection_stability["dataset"] == "nfcorpus_repeated_semantic_depth"
    assert selection_stability["feature_set"] == "full"
    assert selection_stability["policy_model"] == "ridge"
    assert selection_stability["n_runs"] == 2
    assert selection_stability["validation_best_depth_modal"] == 8
    assert selection_stability["heldout_best_depth_modal"] == 5
    assert selection_stability["validation_matches_heldout_depth_rate"] == 0.5
    assert selection_stability["validation_matches_heldout_depth_rate_ci_low"] == 0.094528654801
    assert selection_stability["validation_matches_heldout_depth_rate_ci_high"] == 0.905471345199
    assert selection_stability["depth_selection_reward_gap_mean"] == 0.01
    assert predictive_stability["group"] == "semantic_rank_profile"
    assert predictive_stability["test_r2_delta_across_seed_mean"] == 0.0


def test_run_repeated_semantic_depth_sweep_requires_seeds() -> None:
    with pytest.raises(ValueError, match="seed"):
        run_repeated_semantic_depth_sweep(
            dataset="nfcorpus",
            data_path=Path("data"),
            output_dir=Path("outputs"),
            seeds=[],
            semantic_depths=[5, 8],
            depth_sweep_runner=_fake_depth_sweep_runner,
        )


def _fake_sweep_runner(
    *,
    dataset: str,
    output_dir: Path,
    semantic_depth: int,
    **_: object,
) -> Path:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{dataset}_policy_model_sweep.csv"
    depth_bonus = 0.0 if semantic_depth == 5 else 0.08
    rows = [
        _row(dataset, "full", semantic_depth, validation=0.50 + depth_bonus * 0.625, reward=0.40 + depth_bonus),
        _row(dataset, "no_semantic", semantic_depth, validation=0.45, reward=0.42),
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "group": "semantic_rank_profile",
                "target": "dense_advantage_vs_bm25",
                "start": 23,
                "end": 32 if semantic_depth == 5 else 38,
                "width": 9 if semantic_depth == 5 else 15,
                "active_columns": 9 if semantic_depth == 5 else 15,
                "alpha": 1.0,
                "train_rows": 4,
                "test_rows": 3,
                "train_r2": 0.10 + depth_bonus,
                "test_r2": 0.20 + (0.03 if semantic_depth == 8 else 0.0),
                "test_corr": 0.40 + (0.10 if semantic_depth == 8 else 0.0),
                "target_train_std": 0.2,
                "target_test_std": 0.3,
            }
        ]
    ).to_csv(results_dir / f"{dataset}_policy_feature_predictive_diagnostics.csv", index=False)
    return csv_path


def _row(
    dataset: str,
    feature_set: str,
    semantic_depth: int,
    validation: float,
    reward: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "policy_model": "ridge",
        "selected_policy_model": "ridge_l2=1.0",
        "semantic_depth": semantic_depth,
        "train_size": 4,
        "test_size": 3,
        "feature_width": 14 + semantic_depth,
        "validation_reward": validation,
        "selective_reward": reward,
        "best_fixed_reward": 0.40,
        "oracle_reward": 1.00,
        "selective_recall_at_5": reward / 2.0,
        "best_fixed_recall_at_5": 0.08,
        "selective_retrieval_calls": 1.0,
    }


def _fake_depth_sweep_runner(
    *,
    dataset: str,
    output_dir: Path,
    seed: int,
    **_: object,
) -> dict[str, object]:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    reward_delta = 0.08 if seed == 1 else -0.02
    validation_delta = 0.04 if seed == 1 else 0.01
    r2_delta = 0.04 if seed == 1 else -0.04
    depth_effects_csv = results_dir / f"{dataset}_semantic_depth_effects.csv"
    predictive_effects_csv = results_dir / f"{dataset}_semantic_depth_predictive_effects.csv"
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "feature_set": "full",
                "policy_model": "ridge",
                "semantic_depth": 8,
                "baseline_semantic_depth": 5,
                "n_pairs": 1,
                "validation_reward_delta_mean": validation_delta,
                "validation_reward_delta_std": 0.0,
                "validation_reward_win_rate": 1.0 if validation_delta > 0 else 0.0,
                "selective_reward_delta_mean": reward_delta,
                "selective_reward_delta_std": 0.0,
                "selective_reward_win_rate": 1.0 if reward_delta > 0 else 0.0,
                "selective_recall_at_5_delta_mean": reward_delta / 2,
                "selective_recall_at_5_delta_std": 0.0,
                "selective_retrieval_calls_delta_mean": 0.0,
                "selective_retrieval_calls_delta_std": 0.0,
            }
        ]
    ).to_csv(depth_effects_csv, index=False)
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "group": "semantic_rank_profile",
                "target": "dense_advantage_vs_bm25",
                "semantic_depth": 8,
                "baseline_semantic_depth": 5,
                "n_pairs": 1,
                "train_r2_delta": r2_delta / 2,
                "test_r2_delta": r2_delta,
                "test_corr_delta": r2_delta * 2,
                "active_columns_delta": 6,
                "width_delta": 6,
            }
        ]
    ).to_csv(predictive_effects_csv, index=False)
    return {
        "depth_effects_csv": str(depth_effects_csv),
        "predictive_effects_csv": str(predictive_effects_csv),
    }
