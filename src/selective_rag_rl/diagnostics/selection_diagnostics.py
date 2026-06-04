from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_GRID_COLUMNS = {
    "feature_set",
    "policy_model",
    "selected_policy_model",
    "validation_reward",
    "selective_reward",
    "best_fixed_reward",
    "oracle_reward",
    "selective_retrieval_calls",
}

SELECTION_DIAGNOSTIC_COLUMNS = [
    "dataset",
    "n_configurations",
    "validation_selected_config",
    "validation_selected_validation_reward",
    "validation_selected_reward",
    "validation_selected_calls",
    "heldout_best_config",
    "heldout_best_validation_reward",
    "heldout_best_reward",
    "heldout_best_calls",
    "selection_reward_gap",
    "heldout_rank_of_validation_selected",
    "validation_heldout_spearman",
    "validation_top2_margin",
    "heldout_top2_margin",
    "top1_validation_heldout_overlap",
    "top2_validation_heldout_overlap",
    "top3_validation_heldout_overlap",
    "validation_selected_reward_gap_vs_best_fixed",
    "validation_selected_validation_gap_vs_best_fixed",
    "heldout_best_reward_gap_vs_best_fixed",
    "validation_selected_call_gap_vs_best_fixed",
    "heldout_best_call_gap_vs_best_fixed",
    "validation_selected_dominated_by_best_fixed",
    "heldout_best_dominated_by_best_fixed",
    "guardrail_decision",
    "guardrail_reward",
    "guardrail_calls",
    "guardrail_reward_delta_vs_validation_selected",
    "guardrail_call_delta_vs_validation_selected",
    "best_fixed_reward",
    "best_fixed_calls",
    "oracle_reward",
]


def export_selection_diagnostics(grid_csv: Path, output_csv: Path, dataset: str | None = None) -> Path:
    grid = pd.read_csv(grid_csv)
    missing = sorted(REQUIRED_GRID_COLUMNS - set(grid.columns))
    if missing:
        raise ValueError(f"Grid CSV missing required column(s): {', '.join(missing)}")
    if grid.empty:
        raise ValueError("Grid CSV contains no configurations")

    validation_selected = _best_row(grid, "validation_reward")
    heldout_best = _best_row(grid, "selective_reward")
    heldout_ranked = grid.sort_values(
        ["selective_reward", "validation_reward"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    validation_ranked = grid.sort_values(
        ["validation_reward", "selective_reward"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    validation_key = _config_key(validation_selected)
    heldout_rank = int(
        heldout_ranked.index[
            heldout_ranked.apply(lambda row: _config_key(row) == validation_key, axis=1)
        ][0]
        + 1
    )
    best_fixed_reward = float(grid["best_fixed_reward"].iloc[0])
    best_fixed_calls = _best_fixed_calls(grid)
    validation_reward_gap = round(float(validation_selected["selective_reward"] - best_fixed_reward), 12)
    best_fixed_validation_reward = _best_fixed_validation_reward(grid)
    validation_validation_gap = round(
        float(validation_selected["validation_reward"] - best_fixed_validation_reward),
        12,
    )
    heldout_reward_gap = round(float(heldout_best["selective_reward"] - best_fixed_reward), 12)
    validation_call_gap = round(float(validation_selected["selective_retrieval_calls"] - best_fixed_calls), 12)
    heldout_call_gap = round(float(heldout_best["selective_retrieval_calls"] - best_fixed_calls), 12)
    validation_dominated = validation_reward_gap <= 0 and validation_call_gap >= 0
    heldout_dominated = heldout_reward_gap <= 0 and heldout_call_gap >= 0
    guardrail_reward = best_fixed_reward if validation_dominated else float(validation_selected["selective_reward"])
    guardrail_calls = best_fixed_calls if validation_dominated else float(validation_selected["selective_retrieval_calls"])
    diagnostic = {
        "dataset": dataset or str(validation_selected.get("dataset", "unknown")),
        "n_configurations": int(len(grid)),
        "validation_selected_config": validation_key,
        "validation_selected_validation_reward": float(validation_selected["validation_reward"]),
        "validation_selected_reward": float(validation_selected["selective_reward"]),
        "validation_selected_calls": float(validation_selected["selective_retrieval_calls"]),
        "heldout_best_config": _config_key(heldout_best),
        "heldout_best_validation_reward": float(heldout_best["validation_reward"]),
        "heldout_best_reward": float(heldout_best["selective_reward"]),
        "heldout_best_calls": float(heldout_best["selective_retrieval_calls"]),
        "selection_reward_gap": round(float(heldout_best["selective_reward"] - validation_selected["selective_reward"]), 12),
        "heldout_rank_of_validation_selected": heldout_rank,
        "validation_heldout_spearman": float(grid["validation_reward"].corr(grid["selective_reward"], method="spearman")),
        "validation_top2_margin": _top2_margin(validation_ranked, "validation_reward"),
        "heldout_top2_margin": _top2_margin(heldout_ranked, "selective_reward"),
        "top1_validation_heldout_overlap": _top_k_overlap(validation_ranked, heldout_ranked, k=1),
        "top2_validation_heldout_overlap": _top_k_overlap(validation_ranked, heldout_ranked, k=2),
        "top3_validation_heldout_overlap": _top_k_overlap(validation_ranked, heldout_ranked, k=3),
        "validation_selected_reward_gap_vs_best_fixed": validation_reward_gap,
        "validation_selected_validation_gap_vs_best_fixed": validation_validation_gap,
        "heldout_best_reward_gap_vs_best_fixed": heldout_reward_gap,
        "validation_selected_call_gap_vs_best_fixed": validation_call_gap,
        "heldout_best_call_gap_vs_best_fixed": heldout_call_gap,
        "validation_selected_dominated_by_best_fixed": validation_dominated,
        "heldout_best_dominated_by_best_fixed": heldout_dominated,
        "guardrail_decision": "fallback_train_best_fixed" if validation_dominated else "keep_validation_selected",
        "guardrail_reward": guardrail_reward,
        "guardrail_calls": guardrail_calls,
        "guardrail_reward_delta_vs_validation_selected": round(
            float(guardrail_reward - validation_selected["selective_reward"]),
            12,
        ),
        "guardrail_call_delta_vs_validation_selected": round(
            float(guardrail_calls - validation_selected["selective_retrieval_calls"]),
            12,
        ),
        "best_fixed_reward": best_fixed_reward,
        "best_fixed_calls": best_fixed_calls,
        "oracle_reward": float(grid["oracle_reward"].iloc[0]),
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([diagnostic], columns=SELECTION_DIAGNOSTIC_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _best_row(grid: pd.DataFrame, metric: str) -> pd.Series:
    ranked = grid.sort_values([metric, "validation_reward"], ascending=[False, False], kind="mergesort")
    return ranked.iloc[0]


def _top2_margin(ranked: pd.DataFrame, metric: str) -> float:
    if len(ranked) < 2:
        return 0.0
    return round(float(ranked[metric].iloc[0] - ranked[metric].iloc[1]), 12)


def _top_k_overlap(left_ranked: pd.DataFrame, right_ranked: pd.DataFrame, k: int) -> float:
    top_k = min(k, len(left_ranked), len(right_ranked))
    if top_k <= 0:
        return 0.0
    left = {_config_key(row) for _, row in left_ranked.head(top_k).iterrows()}
    right = {_config_key(row) for _, row in right_ranked.head(top_k).iterrows()}
    return len(left & right) / top_k


def _config_key(row: pd.Series) -> str:
    key = f"{row['feature_set']}/{row['policy_model']}/{row['selected_policy_model']}"
    if "semantic_depth" not in row or pd.isna(row["semantic_depth"]):
        return key
    return f"depth={int(row['semantic_depth'])}/{key}"


def _best_fixed_calls(grid: pd.DataFrame) -> float:
    if "best_fixed_retrieval_calls" not in grid:
        return 0.0
    return float(grid["best_fixed_retrieval_calls"].iloc[0])


def _best_fixed_validation_reward(grid: pd.DataFrame) -> float:
    if "best_fixed_validation_reward" in grid:
        return float(grid["best_fixed_validation_reward"].iloc[0])
    return float(grid["best_fixed_reward"].iloc[0])
