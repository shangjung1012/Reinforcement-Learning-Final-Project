from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "dataset",
    "split",
    "examples",
    "fqi_reward",
    "train_best_reward",
    "oracle_reward",
    "reward_gap_vs_train_best",
    "oracle_gap",
    "fqi_calls",
    "train_best_calls",
    "call_gap_vs_train_best",
    "oracle_trace_match_rate",
    "stop_rate",
    "refine_rate",
]

TRACE_COLUMNS = [
    "dataset",
    "split",
    "action_trace",
    "count",
    "rate",
    "mean_reward",
    "mean_retrieval_calls",
]


def export_fqi_diagnostics(
    *,
    detailed_csv: Path,
    summary_csv: Path,
    trace_csv: Path,
    dataset: str,
    split: str = "test",
) -> dict[str, Path]:
    frame = pd.read_csv(detailed_csv)
    summary, traces = build_fqi_diagnostics(frame, dataset=dataset, split=split)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    trace_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    traces.to_csv(trace_csv, index=False)
    return {"summary_csv": summary_csv, "trace_csv": trace_csv}


def build_fqi_diagnostics(frame: pd.DataFrame, *, dataset: str, split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"split", "method", "qid", "action_trace", "reward", "retrieval_calls"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"multistep detailed CSV missing column(s): {', '.join(missing)}")
    subset = frame[frame["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"multistep detailed CSV has no rows for split={split!r}")

    fqi = _method(subset, "Multi-step FQI")
    train_best = _method(subset, "Train-best fixed trace")
    oracle = _method(subset, "Oracle two-step")
    joined = fqi[["qid", "action_trace"]].merge(
        oracle[["qid", "action_trace"]],
        on="qid",
        suffixes=("_fqi", "_oracle"),
    )
    if joined.empty:
        oracle_trace_match_rate = 0.0
    else:
        oracle_trace_match_rate = float((joined["action_trace_fqi"] == joined["action_trace_oracle"]).mean())

    fqi_reward = float(fqi["reward"].mean())
    train_best_reward = float(train_best["reward"].mean())
    oracle_reward = float(oracle["reward"].mean())
    fqi_calls = float(fqi["retrieval_calls"].mean())
    train_best_calls = float(train_best["retrieval_calls"].mean())
    traces = _trace_distribution(fqi, dataset=dataset, split=split)
    stop_rate = float((fqi["action_trace"] == "stop").mean())
    summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "split": split,
                "examples": int(fqi["qid"].nunique()),
                "fqi_reward": fqi_reward,
                "train_best_reward": train_best_reward,
                "oracle_reward": oracle_reward,
                "reward_gap_vs_train_best": fqi_reward - train_best_reward,
                "oracle_gap": oracle_reward - fqi_reward,
                "fqi_calls": fqi_calls,
                "train_best_calls": train_best_calls,
                "call_gap_vs_train_best": fqi_calls - train_best_calls,
                "oracle_trace_match_rate": oracle_trace_match_rate,
                "stop_rate": stop_rate,
                "refine_rate": 1.0 - stop_rate,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    return summary, traces


def _method(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = frame[frame["method"] == method].copy()
    if rows.empty:
        raise ValueError(f"Missing method row(s): {method}")
    return rows


def _trace_distribution(fqi: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    grouped = (
        fqi.groupby("action_trace", sort=False)
        .agg(
            count=("qid", "count"),
            mean_reward=("reward", "mean"),
            mean_retrieval_calls=("retrieval_calls", "mean"),
        )
        .reset_index()
    )
    total = max(int(grouped["count"].sum()), 1)
    grouped["dataset"] = dataset
    grouped["split"] = split
    grouped["rate"] = grouped["count"] / total
    grouped = grouped.sort_values(["count", "mean_reward", "action_trace"], ascending=[False, False, True])
    return grouped[TRACE_COLUMNS]
