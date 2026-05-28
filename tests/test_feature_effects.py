from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.feature_effects import export_feature_effects


def test_export_feature_effects_compares_feature_sets_against_baseline(tmp_path: Path) -> None:
    grid_one = tmp_path / "seed_1.csv"
    grid_two = tmp_path / "seed_2.csv"
    output_csv = tmp_path / "feature_effects.csv"
    pd.DataFrame(
        [
            _row("full", "ridge", validation=0.50, reward=0.60, recall=0.40, calls=1.20),
            _row("no_semantic", "ridge", validation=0.45, reward=0.50, recall=0.30, calls=1.00),
            _row("full", "auto", validation=0.55, reward=0.40, recall=0.25, calls=1.30),
            _row("no_semantic", "auto", validation=0.60, reward=0.45, recall=0.30, calls=1.10),
        ]
    ).to_csv(grid_one, index=False)
    pd.DataFrame(
        [
            _row("full", "ridge", validation=0.40, reward=0.45, recall=0.35, calls=1.10),
            _row("no_semantic", "ridge", validation=0.50, reward=0.55, recall=0.40, calls=1.00),
            _row("full", "auto", validation=0.70, reward=0.80, recall=0.60, calls=1.50),
            _row("no_semantic", "auto", validation=0.65, reward=0.75, recall=0.55, calls=1.20),
        ]
    ).to_csv(grid_two, index=False)

    exported = export_feature_effects(
        [grid_one, grid_two],
        output_csv,
        dataset="toy",
        baseline_feature_set="no_semantic",
    )
    effects = pd.read_csv(exported)

    assert list(effects["policy_model"]) == ["auto", "ridge", "__all__"]
    auto = effects[effects["policy_model"] == "auto"].iloc[0]
    ridge = effects[effects["policy_model"] == "ridge"].iloc[0]
    all_row = effects[effects["policy_model"] == "__all__"].iloc[0]
    assert auto["n_pairs"] == 2
    assert round(float(auto["selective_reward_delta_mean"]), 6) == 0.0
    assert round(float(auto["selective_reward_win_rate"]), 6) == 0.5
    assert round(float(ridge["selective_reward_delta_mean"]), 6) == 0.0
    assert round(float(ridge["validation_reward_delta_mean"]), 6) == -0.025
    assert all_row["n_pairs"] == 4
    assert round(float(all_row["selective_retrieval_calls_delta_mean"]), 6) == 0.2


def test_export_feature_effects_accepts_feature_ablation_csv_without_policy_model(tmp_path: Path) -> None:
    grid_csv = tmp_path / "feature_ablation.csv"
    output_csv = tmp_path / "feature_effects.csv"
    pd.DataFrame(
        [
            _ablation_row("full", "ridge_l2=1.0", reward=0.38, recall=0.14, calls=1.10),
            _ablation_row("no_score_shape", "ridge_l2=1.0", reward=0.38, recall=0.14, calls=1.10),
            _ablation_row("score_shape_only", "ridge_l2=1.0", reward=0.41, recall=0.15, calls=1.00),
            _ablation_row("no_semantic", "ridge_l2=1.0", reward=0.43, recall=0.15, calls=1.00),
        ]
    ).to_csv(grid_csv, index=False)

    exported = export_feature_effects(
        [grid_csv],
        output_csv,
        dataset="toy_ablation",
        baseline_feature_set="no_semantic",
    )
    effects = pd.read_csv(exported)

    full = effects[(effects["feature_set"] == "full") & (effects["policy_model"] == "ridge_l2=1.0")].iloc[0]
    shape_only = effects[
        (effects["feature_set"] == "score_shape_only") & (effects["policy_model"] == "ridge_l2=1.0")
    ].iloc[0]
    assert round(float(full["selective_reward_delta_mean"]), 6) == -0.05
    assert round(float(shape_only["selective_reward_delta_mean"]), 6) == -0.02
    assert pd.isna(full["validation_reward_delta_mean"])


def _row(
    feature_set: str,
    policy_model: str,
    validation: float,
    reward: float,
    recall: float,
    calls: float,
) -> dict[str, object]:
    return {
        "dataset": "toy",
        "feature_set": feature_set,
        "policy_model": policy_model,
        "selected_policy_model": "ridge_l2=1.0",
        "train_size": 4,
        "test_size": 3,
        "feature_width": 56 if feature_set == "full" else 28,
        "validation_reward": validation,
        "selective_reward": reward,
        "best_fixed_reward": 0.40,
        "oracle_reward": 1.00,
        "selective_recall_at_5": recall,
        "best_fixed_recall_at_5": 0.20,
        "selective_retrieval_calls": calls,
    }


def _ablation_row(
    feature_set: str,
    selected_policy_model: str,
    reward: float,
    recall: float,
    calls: float,
) -> dict[str, object]:
    return {
        "dataset": "toy",
        "feature_set": feature_set,
        "selected_policy_model": selected_policy_model,
        "train_size": 4,
        "test_size": 3,
        "selective_reward": reward,
        "best_fixed_reward": 0.40,
        "oracle_reward": 1.00,
        "selective_recall_at_5": recall,
        "best_fixed_recall_at_5": 0.20,
        "selective_retrieval_calls": calls,
    }
