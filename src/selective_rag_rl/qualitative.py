from __future__ import annotations

from pathlib import Path

import pandas as pd

QUALITATIVE_COLUMNS = [
    "case_type",
    "dataset",
    "qid",
    "question",
    "selected_action",
    "train_best_action",
    "policy_reward",
    "train_best_reward",
    "oracle_reward",
    "policy_top_docs",
    "train_best_top_docs",
    "gold_docs",
    "policy_queries",
]


def export_qualitative_examples(
    detailed_csv: Path,
    output_csv: Path,
    dataset: str,
    max_examples_per_case: int = 3,
) -> Path:
    detailed = pd.read_csv(detailed_csv)
    test_rows = detailed[detailed["split"] == "test"]
    counters = {
        "policy_beats_train_best": 0,
        "policy_uses_dense": 0,
        "policy_uses_hybrid": 0,
        "policy_avoids_hybrid": 0,
    }
    rows: list[dict[str, object]] = []

    for _, group in test_rows.groupby("qid", sort=False):
        policy = _method_row(group, "Selective retrieval policy")
        train_best = _method_row(group, "Train-best retrieval action")
        oracle = _method_row(group, "Oracle retrieval action")
        if policy is None or train_best is None or oracle is None:
            continue

        policy_reward = float(policy["reward"])
        train_best_reward = float(train_best["reward"])
        selected_action = str(policy["action"])
        train_best_action = str(train_best["action"])

        candidates = []
        if policy_reward > train_best_reward:
            candidates.append("policy_beats_train_best")
        if selected_action.startswith("dense") and policy_reward >= train_best_reward:
            candidates.append("policy_uses_dense")
        if selected_action.startswith("hybrid") and policy_reward >= train_best_reward:
            candidates.append("policy_uses_hybrid")
        if train_best_action.startswith("hybrid") and not selected_action.startswith("hybrid") and policy_reward >= train_best_reward:
            candidates.append("policy_avoids_hybrid")

        for case_type in candidates:
            if counters[case_type] >= max_examples_per_case:
                continue
            rows.append(_case_row(case_type, dataset, policy, train_best, oracle))
            counters[case_type] += 1

        if all(count >= max_examples_per_case for count in counters.values()):
            break

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=QUALITATIVE_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _method_row(group: pd.DataFrame, method: str) -> pd.Series | None:
    rows = group[group["method"] == method]
    if rows.empty:
        return None
    return rows.iloc[0]


def _case_row(
    case_type: str,
    dataset: str,
    policy: pd.Series,
    train_best: pd.Series,
    oracle: pd.Series,
) -> dict[str, object]:
    return {
        "case_type": case_type,
        "dataset": dataset,
        "qid": policy["qid"],
        "question": policy["question"],
        "selected_action": policy["action"],
        "train_best_action": train_best["action"],
        "policy_reward": float(policy["reward"]),
        "train_best_reward": float(train_best["reward"]),
        "oracle_reward": float(oracle["reward"]),
        "policy_top_docs": policy["top_docs"],
        "train_best_top_docs": train_best["top_docs"],
        "gold_docs": policy["gold_docs"],
        "policy_queries": policy["queries"],
    }
