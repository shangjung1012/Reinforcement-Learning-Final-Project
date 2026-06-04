from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.diagnostics.selection_diagnostics import export_selection_diagnostics


def test_export_selection_diagnostics_writes_validation_and_heldout_gap(tmp_path: Path) -> None:
    grid_csv = tmp_path / "grid.csv"
    output_csv = tmp_path / "selection.csv"
    pd.DataFrame(
        [
            _row("full", "ridge", "ridge_l2=1.0", 0.20, 0.50),
            _row("full", "mlp", "mlp", 0.40, 0.45),
            _row("no_profile", "ridge", "ridge_l2=1.0", 0.30, 0.60),
        ]
    ).to_csv(grid_csv, index=False)

    exported = export_selection_diagnostics(grid_csv, output_csv, dataset="toy")
    diagnostics = pd.read_csv(exported)
    row = diagnostics.iloc[0]

    assert row["dataset"] == "toy"
    assert row["n_configurations"] == 3
    assert row["validation_selected_config"] == "full/mlp/mlp"
    assert row["heldout_best_config"] == "no_profile/ridge/ridge_l2=1.0"
    assert row["validation_selected_reward"] == 0.45
    assert row["heldout_best_reward"] == 0.60
    assert row["selection_reward_gap"] == 0.15
    assert row["heldout_rank_of_validation_selected"] == 3
    assert row["validation_top2_margin"] == 0.10
    assert row["heldout_top2_margin"] == 0.10
    assert row["top3_validation_heldout_overlap"] == 1.0
    assert row["validation_selected_reward_gap_vs_best_fixed"] == 0.05
    assert row["heldout_best_reward_gap_vs_best_fixed"] == 0.20
    assert row["validation_selected_call_gap_vs_best_fixed"] == 0.50
    assert row["heldout_best_call_gap_vs_best_fixed"] == 0.50
    assert row["validation_selected_validation_gap_vs_best_fixed"] == 0.05
    assert not bool(row["validation_selected_dominated_by_best_fixed"])
    assert not bool(row["heldout_best_dominated_by_best_fixed"])
    assert "validation_heldout_spearman" in diagnostics.columns


def test_export_selection_diagnostics_flags_best_fixed_dominance(tmp_path: Path) -> None:
    grid_csv = tmp_path / "dominated_grid.csv"
    output_csv = tmp_path / "selection.csv"
    pd.DataFrame(
        [
            _row("full", "ridge", "ridge_l2=1.0", 0.90, 0.30),
            _row("full", "mlp", "mlp", 0.80, 0.35),
        ]
    ).to_csv(grid_csv, index=False)

    exported = export_selection_diagnostics(grid_csv, output_csv, dataset="toy")
    row = pd.read_csv(exported).iloc[0]

    assert row["validation_selected_reward_gap_vs_best_fixed"] == -0.10
    assert row["validation_selected_call_gap_vs_best_fixed"] == 0.50
    assert bool(row["validation_selected_dominated_by_best_fixed"])
    assert row["guardrail_decision"] == "fallback_train_best_fixed"
    assert row["guardrail_reward"] == 0.40
    assert row["guardrail_calls"] == 0.50
    assert row["guardrail_reward_delta_vs_validation_selected"] == 0.10
    assert row["guardrail_call_delta_vs_validation_selected"] == -0.50


def test_export_selection_diagnostics_keeps_non_dominated_validation_selection(tmp_path: Path) -> None:
    grid_csv = tmp_path / "accepted_grid.csv"
    output_csv = tmp_path / "selection.csv"
    pd.DataFrame(
        [
            _row("full", "ridge", "ridge_l2=1.0", 0.90, 0.45) | {"selective_retrieval_calls": 0.50},
            _row("full", "mlp", "mlp", 0.80, 0.35),
        ]
    ).to_csv(grid_csv, index=False)

    exported = export_selection_diagnostics(grid_csv, output_csv, dataset="toy")
    row = pd.read_csv(exported).iloc[0]

    assert row["guardrail_decision"] == "keep_validation_selected"
    assert row["guardrail_reward"] == 0.45
    assert row["guardrail_calls"] == 0.50
    assert row["guardrail_reward_delta_vs_validation_selected"] == 0.0
    assert row["guardrail_call_delta_vs_validation_selected"] == 0.0


def test_export_selection_diagnostics_quantifies_top_k_ranking_mismatch(tmp_path: Path) -> None:
    grid_csv = tmp_path / "grid.csv"
    output_csv = tmp_path / "selection.csv"
    pd.DataFrame(
        [
            _row("full", "ridge", "ridge_l2=1.0", 0.90, 0.20),
            _row("full", "mlp", "mlp", 0.80, 0.30),
            _row("no_profile", "ridge", "ridge_l2=1.0", 0.70, 0.95),
            _row("semantic_only", "ridge", "ridge_l2=1.0", 0.10, 0.90),
        ]
    ).to_csv(grid_csv, index=False)

    exported = export_selection_diagnostics(grid_csv, output_csv, dataset="toy")
    row = pd.read_csv(exported).iloc[0]

    assert row["validation_selected_config"] == "full/ridge/ridge_l2=1.0"
    assert row["heldout_best_config"] == "no_profile/ridge/ridge_l2=1.0"
    assert row["top1_validation_heldout_overlap"] == 0.0
    assert row["top2_validation_heldout_overlap"] == 0.0
    assert row["top3_validation_heldout_overlap"] == 2 / 3
    assert row["validation_selected_reward_gap_vs_best_fixed"] == -0.20
    assert row["heldout_best_reward_gap_vs_best_fixed"] == 0.55


def test_export_selection_diagnostics_distinguishes_semantic_depth_configs(tmp_path: Path) -> None:
    grid_csv = tmp_path / "depth_grid.csv"
    output_csv = tmp_path / "selection.csv"
    rows = [
        _row("full", "ridge", "ridge_l2=1.0", 0.90, 0.20) | {"semantic_depth": 5},
        _row("full", "ridge", "ridge_l2=1.0", 0.80, 0.60) | {"semantic_depth": 8},
    ]
    pd.DataFrame(rows).to_csv(grid_csv, index=False)

    exported = export_selection_diagnostics(grid_csv, output_csv, dataset="toy_depth")
    row = pd.read_csv(exported).iloc[0]

    assert row["validation_selected_config"] == "depth=5/full/ridge/ridge_l2=1.0"
    assert row["heldout_best_config"] == "depth=8/full/ridge/ridge_l2=1.0"
    assert row["selection_reward_gap"] == 0.40
    assert row["heldout_rank_of_validation_selected"] == 2
    assert row["top1_validation_heldout_overlap"] == 0.0


def _row(feature_set: str, policy_model: str, selected: str, validation: float, reward: float) -> dict[str, object]:
    return {
        "dataset": "toy",
        "feature_set": feature_set,
        "policy_model": policy_model,
        "selected_policy_model": selected,
        "validation_reward": validation,
        "best_fixed_validation_reward": 0.35,
        "selective_reward": reward,
        "best_fixed_reward": 0.40,
        "best_fixed_retrieval_calls": 0.50,
        "oracle_reward": 0.70,
        "selective_retrieval_calls": 1.0,
    }
