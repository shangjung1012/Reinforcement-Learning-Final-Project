from __future__ import annotations

from pathlib import Path

import pandas as pd

EFFECT_COLUMNS = [
    "dataset",
    "feature_set",
    "baseline_feature_set",
    "policy_model",
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

METRICS = [
    "validation_reward",
    "selective_reward",
    "selective_recall_at_5",
    "selective_retrieval_calls",
]


def export_feature_effects(
    grid_csvs: list[Path],
    output_csv: Path,
    dataset: str,
    baseline_feature_set: str = "no_semantic",
) -> Path:
    pair_rows = _paired_feature_rows(grid_csvs, baseline_feature_set)
    summaries = []
    for (feature_set, policy_model), group in pair_rows.groupby(["feature_set", "policy_model"], sort=True):
        summaries.append(_summary_row(dataset, feature_set, baseline_feature_set, policy_model, group))
    for feature_set, group in pair_rows.groupby("feature_set", sort=True):
        summaries.append(_summary_row(dataset, feature_set, baseline_feature_set, "__all__", group))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries, columns=EFFECT_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _paired_feature_rows(grid_csvs: list[Path], baseline_feature_set: str) -> pd.DataFrame:
    rows = []
    for run_index, grid_csv in enumerate(grid_csvs):
        grid = pd.read_csv(grid_csv)
        baseline_rows = grid[grid["feature_set"] == baseline_feature_set]
        policy_column = _policy_pairing_column(grid)
        for _, row in grid[grid["feature_set"] != baseline_feature_set].iterrows():
            matches = baseline_rows[baseline_rows[policy_column] == row[policy_column]]
            if matches.empty:
                continue
            baseline = matches.iloc[0]
            paired = {
                "run_index": run_index,
                "grid_csv": str(grid_csv),
                "feature_set": row["feature_set"],
                "baseline_feature_set": baseline_feature_set,
                "policy_model": row[policy_column],
            }
            for metric in METRICS:
                paired[f"{metric}_delta"] = (
                    round(float(row[metric]) - float(baseline[metric]), 12)
                    if metric in grid.columns
                    else float("nan")
                )
            rows.append(paired)
    if not rows:
        raise ValueError(f"No comparable feature-set rows found against baseline '{baseline_feature_set}'")
    return pd.DataFrame(rows)


def _policy_pairing_column(grid: pd.DataFrame) -> str:
    if "policy_model" in grid.columns:
        return "policy_model"
    if "selected_policy_model" in grid.columns:
        return "selected_policy_model"
    raise ValueError("Feature-effect inputs must include policy_model or selected_policy_model")


def _summary_row(
    dataset: str,
    feature_set: str,
    baseline_feature_set: str,
    policy_model: str,
    group: pd.DataFrame,
) -> dict[str, object]:
    row = {
        "dataset": dataset,
        "feature_set": feature_set,
        "baseline_feature_set": baseline_feature_set,
        "policy_model": policy_model,
        "n_pairs": int(len(group)),
    }
    for metric in METRICS:
        deltas = pd.to_numeric(group[f"{metric}_delta"], errors="coerce").dropna()
        row[f"{metric}_delta_mean"] = float(deltas.mean()) if len(deltas) else float("nan")
        row[f"{metric}_delta_std"] = float(deltas.std(ddof=0)) if len(deltas) else float("nan")
        if metric in {"validation_reward", "selective_reward"}:
            row[f"{metric}_win_rate"] = float((deltas > 0).mean()) if len(deltas) else float("nan")
    return row
