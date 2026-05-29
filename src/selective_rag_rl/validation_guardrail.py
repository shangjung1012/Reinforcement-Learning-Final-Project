from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CANDIDATE_METHOD = "Selective retrieval policy"
DEFAULT_TRAIN_BEST_METHOD = "Train-best retrieval action"
REQUIRED_SUMMARY_COLUMNS = {"method", "reward", "retrieval_calls"}


def recommend_guardrail(
    candidate_reward: float,
    candidate_calls: float,
    train_best_reward: float,
    train_best_calls: float,
    has_validation: bool,
) -> dict[str, object]:
    reward_gap = float(candidate_reward - train_best_reward)
    call_gap = float(candidate_calls - train_best_calls)
    dominated = candidate_reward <= train_best_reward and candidate_calls >= train_best_calls
    if not has_validation:
        return {
            "reward_gap": reward_gap,
            "call_gap": call_gap,
            "dominated_by_train_best": bool(dominated),
            "recommendation": "analysis_only_no_validation",
            "reason": "no_validation_split_available",
        }
    if dominated:
        return {
            "reward_gap": reward_gap,
            "call_gap": call_gap,
            "dominated_by_train_best": True,
            "recommendation": "fallback_to_train_best_fixed",
            "reason": "dominated_by_train_best_on_validation",
        }
    if candidate_reward > train_best_reward and candidate_calls > train_best_calls:
        return {
            "reward_gap": reward_gap,
            "call_gap": call_gap,
            "dominated_by_train_best": False,
            "recommendation": "review_cost_reward_tradeoff",
            "reason": "higher_reward_with_higher_calls",
        }
    if candidate_reward == train_best_reward and candidate_calls < train_best_calls:
        return {
            "reward_gap": reward_gap,
            "call_gap": call_gap,
            "dominated_by_train_best": False,
            "recommendation": "select_candidate",
            "reason": "lower_calls_with_equal_reward",
        }
    if candidate_reward >= train_best_reward and candidate_calls <= train_best_calls:
        return {
            "reward_gap": reward_gap,
            "call_gap": call_gap,
            "dominated_by_train_best": False,
            "recommendation": "select_candidate",
            "reason": "higher_or_equal_reward_with_lower_or_equal_calls",
        }
    return {
        "reward_gap": reward_gap,
        "call_gap": call_gap,
        "dominated_by_train_best": False,
        "recommendation": "review_cost_reward_tradeoff",
        "reason": "lower_reward_with_lower_calls",
    }


def evaluate_summary_frame(
    df: pd.DataFrame,
    dataset: str,
    candidate_method: str = DEFAULT_CANDIDATE_METHOD,
    train_best_method: str = DEFAULT_TRAIN_BEST_METHOD,
    seed: int | None = None,
) -> dict[str, object]:
    missing = REQUIRED_SUMMARY_COLUMNS - set(df.columns)
    if missing:
        return _blocked_row(dataset, seed, "blocked_missing_columns", "missing_required_columns", sorted(missing))
    candidate = _method_row(df, candidate_method)
    train_best = _method_row(df, train_best_method)
    if candidate is None or train_best is None:
        return _blocked_row(dataset, seed, "blocked_missing_data", "missing_candidate_or_train_best_method", [])

    heldout_best = _heldout_best_method(df)
    rec = recommend_guardrail(
        candidate_reward=float(candidate["reward"]),
        candidate_calls=float(candidate["retrieval_calls"]),
        train_best_reward=float(train_best["reward"]),
        train_best_calls=float(train_best["retrieval_calls"]),
        has_validation=False,
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "selected_config": candidate_method,
        "heldout_best_config": heldout_best,
        "validation_reward": None,
        "heldout_reward": float(candidate["reward"]),
        "reward_gap": rec["reward_gap"],
        "call_gap": rec["call_gap"],
        "dominated_by_train_best": rec["dominated_by_train_best"],
        "recommendation": rec["recommendation"],
        "reason": rec["reason"],
    }


def evaluate_detailed_frame(
    df: pd.DataFrame,
    dataset: str,
    candidate_method: str = DEFAULT_CANDIDATE_METHOD,
    train_best_method: str = DEFAULT_TRAIN_BEST_METHOD,
    heldout_split: str = "test",
    seed: int | None = None,
) -> dict[str, object]:
    missing = REQUIRED_SUMMARY_COLUMNS - set(df.columns)
    if missing:
        return _blocked_row(dataset, seed, "blocked_missing_columns", "missing_required_columns", sorted(missing))
    heldout = df[df["split"] == heldout_split] if "split" in df.columns else df
    summary = (
        heldout.groupby("method", as_index=False)[["reward", "retrieval_calls"]]
        .mean(numeric_only=True)
        .sort_values("method")
    )
    return evaluate_summary_frame(summary, dataset, candidate_method, train_best_method, seed)


def evaluate_grid_frame(df: pd.DataFrame, dataset: str, seed: int | None = None) -> list[dict[str, object]]:
    required = {
        "feature_set",
        "policy_model",
        "validation_reward",
        "selective_reward",
        "best_fixed_reward",
        "selective_retrieval_calls",
        "best_fixed_retrieval_calls",
    }
    missing = required - set(df.columns)
    if missing:
        return [_blocked_row(dataset, seed, "blocked_missing_columns", "missing_required_columns", sorted(missing))]

    rows = []
    for _, row in df.iterrows():
        selected_config = _grid_config(row)
        rec = recommend_guardrail(
            candidate_reward=float(row["selective_reward"]),
            candidate_calls=float(row["selective_retrieval_calls"]),
            train_best_reward=float(row["best_fixed_reward"]),
            train_best_calls=float(row["best_fixed_retrieval_calls"]),
            has_validation=True,
        )
        rows.append(
            {
                "dataset": str(row.get("dataset", dataset)),
                "seed": seed,
                "selected_config": selected_config,
                "heldout_best_config": "",
                "validation_reward": float(row["validation_reward"]),
                "heldout_reward": float(row["selective_reward"]),
                "reward_gap": rec["reward_gap"],
                "call_gap": rec["call_gap"],
                "dominated_by_train_best": rec["dominated_by_train_best"],
                "recommendation": rec["recommendation"],
                "reason": rec["reason"],
            }
        )
    return rows


def aggregate_guardrail_rows(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        return {
            "dataset": "",
            "n_runs": 0,
            "support_rate": 0.0,
            "fallback_rate": 0.0,
            "mean_reward_gap": 0.0,
            "mean_call_gap": 0.0,
        }
    df = pd.DataFrame(rows)
    return {
        "dataset": str(df["dataset"].iloc[0]) if "dataset" in df else "",
        "n_runs": int(len(df)),
        "support_rate": float((df["recommendation"] == "select_candidate").mean()),
        "fallback_rate": float((df["recommendation"] == "fallback_to_train_best_fixed").mean()),
        "mean_reward_gap": float(pd.to_numeric(df["reward_gap"], errors="coerce").mean()),
        "mean_call_gap": float(pd.to_numeric(df["call_gap"], errors="coerce").mean()),
    }


def run_validation_guardrail(
    dataset: str,
    output_csv: Path,
    detailed_csv: Path | None = None,
    summary_csv: Path | None = None,
    grid_csvs: list[Path] | None = None,
    candidate_method: str = DEFAULT_CANDIDATE_METHOD,
    train_best_method: str = DEFAULT_TRAIN_BEST_METHOD,
    heldout_split: str = "test",
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if detailed_csv is not None:
        rows.append(
            evaluate_detailed_frame(
                pd.read_csv(detailed_csv),
                dataset=dataset,
                candidate_method=candidate_method,
                train_best_method=train_best_method,
                heldout_split=heldout_split,
            )
        )
    if summary_csv is not None:
        rows.append(
            evaluate_summary_frame(
                pd.read_csv(summary_csv),
                dataset=dataset,
                candidate_method=candidate_method,
                train_best_method=train_best_method,
            )
        )
    for grid_csv in grid_csvs or []:
        rows.extend(evaluate_grid_frame(pd.read_csv(grid_csv), dataset=dataset, seed=_seed_from_path(grid_csv)))
    if not rows:
        rows.append(_blocked_row(dataset, None, "blocked_missing_data", "no_input_csv_provided", []))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_json = output_csv.with_suffix(".json")
    summary = {
        "rows": rows,
        "aggregate": aggregate_guardrail_rows(rows),
        "outputs": {
            "summary_csv": str(output_csv),
            "summary_json": str(output_json),
        },
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _method_row(df: pd.DataFrame, method: str) -> pd.Series | None:
    part = df[df["method"] == method]
    if part.empty:
        return None
    return part.iloc[0]


def _heldout_best_method(df: pd.DataFrame) -> str:
    candidates = df[~df["method"].str.contains("Oracle", case=False, na=False)]
    if candidates.empty:
        candidates = df
    return str(candidates.sort_values("reward", ascending=False).iloc[0]["method"])


def _blocked_row(
    dataset: str,
    seed: int | None,
    recommendation: str,
    reason: str,
    missing_columns: list[str],
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": seed,
        "selected_config": "",
        "heldout_best_config": "",
        "validation_reward": None,
        "heldout_reward": None,
        "reward_gap": None,
        "call_gap": None,
        "dominated_by_train_best": False,
        "recommendation": recommendation,
        "reason": reason,
        "missing_columns": "|".join(missing_columns),
    }


def _grid_config(row: pd.Series) -> str:
    parts = [str(row["feature_set"]), str(row["policy_model"])]
    if "selected_policy_model" in row and pd.notna(row["selected_policy_model"]):
        parts.append(str(row["selected_policy_model"]))
    if "semantic_depth" in row and pd.notna(row["semantic_depth"]):
        parts.append(f"depth={int(row['semantic_depth'])}")
    return "/".join(parts)


def _seed_from_path(path: Path) -> int | None:
    stem = path.stem
    marker = "seed"
    if marker not in stem:
        return None
    suffix = stem.split(marker, 1)[1]
    digits = ""
    for char in suffix:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None
