from __future__ import annotations

from pathlib import Path

import pandas as pd

DIAGNOSTIC_COLUMNS = [
    "dataset",
    "qid",
    "question",
    "selected_action",
    "train_best_action",
    "oracle_action",
    "policy_reward",
    "train_best_reward",
    "oracle_reward",
    "policy_reward_delta_vs_train_best",
    "policy_regret",
    "oracle_margin_vs_train_best",
    "oracle_margin_vs_second_best",
    "candidate_action_reward_spread",
    "oracle_tie_count",
    "beats_train_best",
    "matches_train_best_action",
    "matches_oracle_action",
]

SUMMARY_METHODS = {
    "Selective retrieval policy",
    "Train-best retrieval action",
    "Oracle retrieval action",
}


def export_policy_diagnostics(detailed_csv: Path, output_csv: Path, dataset: str) -> Path:
    detailed = pd.read_csv(detailed_csv)
    test_rows = detailed[detailed["split"] == "test"]
    rows = []
    for _, group in test_rows.groupby("qid", sort=False):
        policy = _first(group, "Selective retrieval policy")
        train_best = _first(group, "Train-best retrieval action")
        oracle = _first(group, "Oracle retrieval action")
        if policy is None or train_best is None or oracle is None:
            continue
        policy_reward = float(policy["reward"])
        train_best_reward = float(train_best["reward"])
        oracle_reward = float(oracle["reward"])
        selected_action = str(policy["action"])
        train_best_action = str(train_best["action"])
        oracle_action = str(oracle["action"])
        gap_stats = _action_gap_stats(group, oracle_reward)
        rows.append(
            {
                "dataset": dataset,
                "qid": policy["qid"],
                "question": policy["question"],
                "selected_action": selected_action,
                "train_best_action": train_best_action,
                "oracle_action": oracle_action,
                "policy_reward": policy_reward,
                "train_best_reward": train_best_reward,
                "oracle_reward": oracle_reward,
                "policy_reward_delta_vs_train_best": round(policy_reward - train_best_reward, 12),
                "policy_regret": round(oracle_reward - policy_reward, 12),
                "oracle_margin_vs_train_best": round(oracle_reward - train_best_reward, 12),
                **gap_stats,
                "beats_train_best": policy_reward > train_best_reward,
                "matches_train_best_action": selected_action == train_best_action,
                "matches_oracle_action": selected_action == oracle_action,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _action_gap_stats(group: pd.DataFrame, oracle_reward: float) -> dict[str, float | int]:
    candidate_rows = group[~group["method"].isin(SUMMARY_METHODS)]
    if candidate_rows.empty:
        candidate_rows = group[group["method"].isin(SUMMARY_METHODS)]
    rewards = sorted((float(value) for value in candidate_rows["reward"]), reverse=True)
    if not rewards:
        return {
            "oracle_margin_vs_second_best": 0.0,
            "candidate_action_reward_spread": 0.0,
            "oracle_tie_count": 0,
        }
    second_best = rewards[1] if len(rewards) > 1 else rewards[0]
    return {
        "oracle_margin_vs_second_best": round(oracle_reward - second_best, 12),
        "candidate_action_reward_spread": round(rewards[0] - rewards[-1], 12),
        "oracle_tie_count": sum(abs(reward - oracle_reward) <= 1e-12 for reward in rewards),
    }


def _first(group: pd.DataFrame, method: str) -> pd.Series | None:
    rows = group[group["method"] == method]
    if rows.empty:
        return None
    return rows.iloc[0]
