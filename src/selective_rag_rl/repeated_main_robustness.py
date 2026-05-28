from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BANDIT_BASELINE_METHODS = [
    "LinUCB retrieval policy",
    "Epsilon-greedy retrieval policy",
    "Linear Thompson retrieval policy",
]
PER_SEED_COLUMNS = [
    "dataset",
    "seed",
    "train_best_reward",
    "heuristic_reward",
    "best_selected_action_bandit_method",
    "best_selected_action_bandit_reward",
    "best_selected_action_bandit_delta",
    "selective_policy_reward",
    "selective_delta",
    "constrained_utility",
    "constrained_delta",
    "oracle_reward",
    "best_selected_action_bandit_win",
    "selective_win",
    "constrained_win",
    "train_best_calls",
    "best_selected_action_bandit_calls",
    "selective_calls",
    "constrained_calls",
]
AGGREGATE_COLUMNS = [
    "dataset",
    "n_seeds",
    "best_selected_action_bandit_win_rate",
    "selective_win_rate",
    "constrained_win_rate",
    "best_selected_action_bandit_delta_mean",
    "best_selected_action_bandit_delta_std",
    "selective_delta_mean",
    "selective_delta_std",
    "selective_delta_min",
    "constrained_delta_mean",
    "constrained_delta_std",
    "train_best_reward_mean",
    "best_selected_action_bandit_reward_mean",
    "selective_policy_reward_mean",
    "constrained_utility_mean",
    "train_best_calls_mean",
    "best_selected_action_bandit_calls_mean",
    "selective_calls_mean",
    "constrained_calls_mean",
]


def export_repeated_main_robustness(
    *,
    manifest_csv: Path,
    per_seed_csv: Path,
    aggregate_csv: Path,
    call_penalty: float = 0.03,
) -> dict[str, Path]:
    manifest = pd.read_csv(manifest_csv)
    rows = [_per_seed_row(row, call_penalty) for _, row in manifest.iterrows()]
    per_seed = pd.DataFrame(rows, columns=PER_SEED_COLUMNS)
    aggregate = _aggregate_frame(per_seed)
    per_seed_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(per_seed_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)
    return {"per_seed_csv": per_seed_csv, "aggregate_csv": aggregate_csv}


def _per_seed_row(manifest_row: pd.Series, call_penalty: float) -> dict[str, object]:
    dataset = str(manifest_row["dataset"])
    seed = int(manifest_row["seed"])
    policy = pd.read_csv(Path(manifest_row["policy_summary_csv"]))
    bandit = pd.read_csv(Path(manifest_row["bandit_summary_csv"]))
    constrained = _read_optional_frame(manifest_row.get("constrained_sweep_csv"))

    train_best = _method(policy, "Train-best retrieval action")
    heuristic = _method(policy, "Heuristic retrieval router")
    selective = _method(policy, "Selective retrieval policy")
    oracle = _method(policy, "Oracle retrieval action")
    best_bandit = _best_bandit_method(bandit)
    constrained_row = _constrained_row(constrained, call_penalty)

    constrained_utility = _value_or_nan(constrained_row, "policy_utility")
    constrained_delta = (
        float(constrained_row["policy_utility"]) - float(constrained_row["train_best_utility"])
        if constrained_row is not None
        else np.nan
    )
    constrained_calls = _value_or_nan(constrained_row, "policy_retrieval_calls")
    return {
        "dataset": dataset,
        "seed": seed,
        "train_best_reward": float(train_best["reward"]),
        "heuristic_reward": float(heuristic["reward"]),
        "best_selected_action_bandit_method": str(best_bandit["method"]),
        "best_selected_action_bandit_reward": float(best_bandit["reward"]),
        "best_selected_action_bandit_delta": float(best_bandit["reward"]) - float(train_best["reward"]),
        "selective_policy_reward": float(selective["reward"]),
        "selective_delta": float(selective["reward"]) - float(train_best["reward"]),
        "constrained_utility": constrained_utility,
        "constrained_delta": constrained_delta,
        "oracle_reward": float(oracle["reward"]),
        "best_selected_action_bandit_win": float(best_bandit["reward"]) > float(train_best["reward"]),
        "selective_win": float(selective["reward"]) > float(train_best["reward"]),
        "constrained_win": constrained_delta > 0 if not np.isnan(constrained_delta) else np.nan,
        "train_best_calls": float(train_best["retrieval_calls"]),
        "best_selected_action_bandit_calls": float(best_bandit["retrieval_calls"]),
        "selective_calls": float(selective["retrieval_calls"]),
        "constrained_calls": constrained_calls,
    }


def _aggregate_frame(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in per_seed.groupby("dataset", sort=False):
        rows.append(
            {
                "dataset": dataset,
                "n_seeds": int(len(group)),
                "best_selected_action_bandit_win_rate": _mean(group["best_selected_action_bandit_win"]),
                "selective_win_rate": _mean(group["selective_win"]),
                "constrained_win_rate": _mean(group["constrained_win"]),
                "best_selected_action_bandit_delta_mean": _mean(group["best_selected_action_bandit_delta"]),
                "best_selected_action_bandit_delta_std": _std(group["best_selected_action_bandit_delta"]),
                "selective_delta_mean": _mean(group["selective_delta"]),
                "selective_delta_std": _std(group["selective_delta"]),
                "selective_delta_min": _min(group["selective_delta"]),
                "constrained_delta_mean": _mean(group["constrained_delta"]),
                "constrained_delta_std": _std(group["constrained_delta"]),
                "train_best_reward_mean": _mean(group["train_best_reward"]),
                "best_selected_action_bandit_reward_mean": _mean(group["best_selected_action_bandit_reward"]),
                "selective_policy_reward_mean": _mean(group["selective_policy_reward"]),
                "constrained_utility_mean": _mean(group["constrained_utility"]),
                "train_best_calls_mean": _mean(group["train_best_calls"]),
                "best_selected_action_bandit_calls_mean": _mean(group["best_selected_action_bandit_calls"]),
                "selective_calls_mean": _mean(group["selective_calls"]),
                "constrained_calls_mean": _mean(group["constrained_calls"]),
            }
        )
    return pd.DataFrame(rows, columns=AGGREGATE_COLUMNS)


def _method(frame: pd.DataFrame, method: str) -> pd.Series:
    matches = frame[frame["method"] == method]
    if matches.empty:
        raise ValueError(f"Missing method row: {method}")
    return matches.iloc[0]


def _best_bandit_method(frame: pd.DataFrame) -> pd.Series:
    matches = frame[frame["method"].isin(BANDIT_BASELINE_METHODS)]
    if matches.empty:
        raise ValueError("Missing selected-action bandit baseline rows")
    return matches.sort_values("reward", ascending=False).iloc[0]


def _constrained_row(frame: pd.DataFrame | None, call_penalty: float) -> pd.Series | None:
    if frame is None:
        return None
    matches = frame[frame["call_penalty"].round(8) == round(call_penalty, 8)]
    if matches.empty:
        raise ValueError(f"Missing constrained row for call_penalty={call_penalty:g}")
    return matches.iloc[0]


def _read_optional_frame(value: object) -> pd.DataFrame | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return pd.read_csv(Path(value))


def _value_or_nan(row: pd.Series | None, column: str) -> float:
    return float(row[column]) if row is not None else np.nan


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else np.nan


def _std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0


def _min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.min()) if len(numeric) else np.nan
