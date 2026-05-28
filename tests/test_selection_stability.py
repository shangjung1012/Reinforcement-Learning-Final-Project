from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.selection_stability import export_selection_stability


def test_export_selection_stability_summarizes_multiple_grids(tmp_path: Path) -> None:
    grid_a = tmp_path / "grid_a.csv"
    grid_b = tmp_path / "grid_b.csv"
    output_csv = tmp_path / "stability.csv"
    pd.DataFrame(
        [
            _row("full", "ridge", "ridge_l2=1.0", 0.90, 0.20),
            _row("full", "mlp", "mlp", 0.80, 0.30),
            _row("no_semantic", "ridge", "ridge_l2=1.0", 0.70, 0.95),
        ]
    ).to_csv(grid_a, index=False)
    pd.DataFrame(
        [
            _row("full", "ridge", "ridge_l2=1.0", 0.30, 0.20),
            _row("full", "mlp", "mlp", 0.40, 0.30),
            _row("no_semantic", "ridge", "ridge_l2=1.0", 0.95, 0.95),
        ]
    ).to_csv(grid_b, index=False)

    exported = export_selection_stability([grid_a, grid_b], output_csv, dataset="toy")
    row = pd.read_csv(exported).iloc[0]

    assert row["dataset"] == "toy"
    assert row["n_runs"] == 2
    assert row["n_unique_validation_selected"] == 2
    assert row["n_unique_heldout_best"] == 1
    assert row["validation_selected_modal_share"] == 0.5
    assert row["heldout_best_modal_share"] == 1.0
    assert row["validation_matches_heldout_rate"] == 0.5
    assert row["selection_reward_gap_mean"] == 0.375
    assert row["heldout_rank_mean"] == 2.0
    assert row["top1_overlap_mean"] == 0.5
    assert row["validation_selected_beats_best_fixed_rate"] == 0.5
    assert row["validation_selected_validation_gap_vs_best_fixed_mean"] == 0.575
    assert row["heldout_best_beats_best_fixed_rate"] == 1.0
    assert row["validation_selected_call_gap_vs_best_fixed_mean"] == 0.5
    assert row["heldout_best_call_gap_vs_best_fixed_mean"] == 0.5
    assert row["validation_selected_dominated_by_best_fixed_rate"] == 0.5
    assert row["heldout_best_dominated_by_best_fixed_rate"] == 0.0
    assert row["guardrail_fallback_to_best_fixed_rate"] == 0.5
    assert row["guardrail_reward_delta_vs_validation_selected_mean"] == 0.10
    assert row["guardrail_call_delta_vs_validation_selected_mean"] == -0.25


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
        "oracle_reward": 1.00,
        "selective_retrieval_calls": 1.0,
    }
