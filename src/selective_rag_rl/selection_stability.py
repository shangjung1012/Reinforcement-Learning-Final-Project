from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from selective_rag_rl.selection_diagnostics import export_selection_diagnostics

STABILITY_COLUMNS = [
    "dataset",
    "n_runs",
    "n_unique_validation_selected",
    "validation_selected_modal_config",
    "validation_selected_modal_count",
    "validation_selected_modal_share",
    "n_unique_heldout_best",
    "heldout_best_modal_config",
    "heldout_best_modal_count",
    "heldout_best_modal_share",
    "validation_matches_heldout_rate",
    "selection_reward_gap_mean",
    "selection_reward_gap_std",
    "heldout_rank_mean",
    "validation_heldout_spearman_mean",
    "top1_overlap_mean",
    "top2_overlap_mean",
    "top3_overlap_mean",
    "validation_selected_gap_vs_best_fixed_mean",
    "validation_selected_validation_gap_vs_best_fixed_mean",
    "validation_selected_beats_best_fixed_rate",
    "validation_selected_call_gap_vs_best_fixed_mean",
    "validation_selected_dominated_by_best_fixed_rate",
    "heldout_best_gap_vs_best_fixed_mean",
    "heldout_best_beats_best_fixed_rate",
    "heldout_best_call_gap_vs_best_fixed_mean",
    "heldout_best_dominated_by_best_fixed_rate",
    "guardrail_fallback_to_best_fixed_rate",
    "guardrail_reward_delta_vs_validation_selected_mean",
    "guardrail_call_delta_vs_validation_selected_mean",
]


def export_selection_stability(grid_csvs: list[Path], output_csv: Path, dataset: str | None = None) -> Path:
    if not grid_csvs:
        raise ValueError("At least one grid CSV is required")

    diagnostics = pd.DataFrame(
        [
            _diagnostic_row(grid_csv, dataset=dataset or grid_csv.stem)
            for grid_csv in grid_csvs
        ]
    )
    validation_modal, validation_count = _modal(diagnostics["validation_selected_config"])
    heldout_modal, heldout_count = _modal(diagnostics["heldout_best_config"])
    n_runs = len(diagnostics)
    summary = {
        "dataset": dataset or str(diagnostics["dataset"].iloc[0]),
        "n_runs": int(n_runs),
        "n_unique_validation_selected": int(diagnostics["validation_selected_config"].nunique()),
        "validation_selected_modal_config": validation_modal,
        "validation_selected_modal_count": int(validation_count),
        "validation_selected_modal_share": validation_count / n_runs,
        "n_unique_heldout_best": int(diagnostics["heldout_best_config"].nunique()),
        "heldout_best_modal_config": heldout_modal,
        "heldout_best_modal_count": int(heldout_count),
        "heldout_best_modal_share": heldout_count / n_runs,
        "validation_matches_heldout_rate": float(
            (diagnostics["validation_selected_config"] == diagnostics["heldout_best_config"]).mean()
        ),
        "selection_reward_gap_mean": float(diagnostics["selection_reward_gap"].mean()),
        "selection_reward_gap_std": float(diagnostics["selection_reward_gap"].std(ddof=0)),
        "heldout_rank_mean": float(diagnostics["heldout_rank_of_validation_selected"].mean()),
        "validation_heldout_spearman_mean": float(diagnostics["validation_heldout_spearman"].mean()),
        "top1_overlap_mean": float(diagnostics["top1_validation_heldout_overlap"].mean()),
        "top2_overlap_mean": float(diagnostics["top2_validation_heldout_overlap"].mean()),
        "top3_overlap_mean": float(diagnostics["top3_validation_heldout_overlap"].mean()),
        "validation_selected_gap_vs_best_fixed_mean": float(
            diagnostics["validation_selected_reward_gap_vs_best_fixed"].mean()
        ),
        "validation_selected_validation_gap_vs_best_fixed_mean": float(
            diagnostics["validation_selected_validation_gap_vs_best_fixed"].mean()
        ),
        "validation_selected_beats_best_fixed_rate": float(
            (diagnostics["validation_selected_reward_gap_vs_best_fixed"] > 0).mean()
        ),
        "validation_selected_call_gap_vs_best_fixed_mean": float(
            diagnostics["validation_selected_call_gap_vs_best_fixed"].mean()
        ),
        "validation_selected_dominated_by_best_fixed_rate": float(
            diagnostics["validation_selected_dominated_by_best_fixed"].mean()
        ),
        "heldout_best_gap_vs_best_fixed_mean": float(diagnostics["heldout_best_reward_gap_vs_best_fixed"].mean()),
        "heldout_best_beats_best_fixed_rate": float(
            (diagnostics["heldout_best_reward_gap_vs_best_fixed"] > 0).mean()
        ),
        "heldout_best_call_gap_vs_best_fixed_mean": float(
            diagnostics["heldout_best_call_gap_vs_best_fixed"].mean()
        ),
        "heldout_best_dominated_by_best_fixed_rate": float(
            diagnostics["heldout_best_dominated_by_best_fixed"].mean()
        ),
        "guardrail_fallback_to_best_fixed_rate": float(
            (diagnostics["guardrail_decision"] == "fallback_train_best_fixed").mean()
        ),
        "guardrail_reward_delta_vs_validation_selected_mean": float(
            diagnostics["guardrail_reward_delta_vs_validation_selected"].mean()
        ),
        "guardrail_call_delta_vs_validation_selected_mean": float(
            diagnostics["guardrail_call_delta_vs_validation_selected"].mean()
        ),
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary], columns=STABILITY_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _diagnostic_row(grid_csv: Path, dataset: str) -> dict[str, object]:
    output_csv = grid_csv.parent / f".{grid_csv.stem}.selection_diagnostics.tmp.csv"
    try:
        export_selection_diagnostics(grid_csv, output_csv, dataset=dataset)
        return pd.read_csv(output_csv).iloc[0].to_dict()
    finally:
        output_csv.unlink(missing_ok=True)


def _modal(values: pd.Series) -> tuple[str, int]:
    counts = Counter(str(value) for value in values)
    value, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return value, count
