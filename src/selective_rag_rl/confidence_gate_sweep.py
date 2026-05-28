from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = ["recall_at_5", "mrr", "ndcg_at_5", "reward", "rewrite_cost", "retrieval_calls"]
SELECTIVE_METHOD = "Selective retrieval policy"
TRAIN_BEST_METHOD = "Train-best retrieval action"


def export_confidence_gate_sweep(
    *,
    detailed_csv: Path,
    output_csv: Path,
    dataset: str,
    margins: list[float],
) -> Path:
    df = pd.read_csv(detailed_csv)
    _validate_inputs(df, margins)
    rows: list[dict[str, object]] = []
    for split_name, split_df in df.groupby("split", sort=True):
        selective = split_df[split_df["method"] == SELECTIVE_METHOD]
        train_best = split_df[split_df["method"] == TRAIN_BEST_METHOD]
        paired = selective.merge(
            train_best,
            on=["split", "qid"],
            suffixes=("_selective", "_train_best"),
            validate="one_to_one",
        )
        for margin in margins:
            fallback = paired["predicted_action_margin_selective"].astype(float).to_numpy() < float(margin)
            row = {
                "dataset": dataset,
                "split": split_name,
                "confidence_gate_margin": float(margin),
                "n_queries": int(len(paired)),
                "fallback_count": int(fallback.sum()),
                "fallback_rate": float(fallback.mean()) if len(fallback) else 0.0,
            }
            for metric in METRIC_COLUMNS:
                selective_values = paired[f"{metric}_selective"].astype(float).to_numpy()
                train_best_values = paired[f"{metric}_train_best"].astype(float).to_numpy()
                gated_values = np.where(fallback, train_best_values, selective_values)
                selective_mean = _mean(selective_values)
                train_best_mean = _mean(train_best_values)
                gated_mean = _mean(gated_values)
                row[f"selective_{metric}"] = selective_mean
                row[f"train_best_{metric}"] = train_best_mean
                row[f"gated_{metric}"] = gated_mean
                if metric == "reward":
                    row["gated_reward_delta_vs_selective"] = round(float(gated_mean - selective_mean), 12)
                    row["gated_reward_delta_vs_train_best"] = round(float(gated_mean - train_best_mean), 12)
                if metric == "retrieval_calls":
                    row["gated_call_delta_vs_selective"] = round(float(gated_mean - selective_mean), 12)
                    row["gated_call_delta_vs_train_best"] = round(float(gated_mean - train_best_mean), 12)
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def _validate_inputs(df: pd.DataFrame, margins: list[float]) -> None:
    required_columns = {"split", "method", "qid", "predicted_action_margin", *METRIC_COLUMNS}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Detailed CSV is missing required column(s): {', '.join(missing)}")
    if not margins:
        raise ValueError("margins must not be empty")
    if any(margin < 0 for margin in margins):
        raise ValueError("margins must be non-negative")
    methods = set(df["method"])
    required_methods = {SELECTIVE_METHOD, TRAIN_BEST_METHOD}
    missing_methods = sorted(required_methods - methods)
    if missing_methods:
        raise ValueError(f"Detailed CSV is missing required method row(s): {', '.join(missing_methods)}")


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0
