from __future__ import annotations

from pathlib import Path

import pandas as pd

COMPLEXITY_FEATURES = [
    "state_question_length",
    "state_bm25_top1",
    "state_bm25_gap",
    "state_bm25_entropy",
    "predicted_action_margin",
    "oracle_reward_margin",
    "action_reward_std",
    "bm25_dense_doc_overlap",
    "dense_new_doc_rate",
    "hybrid_new_doc_rate",
]

METHODS = [
    "Train-best retrieval action",
    "Heuristic retrieval router",
    "Selective retrieval policy",
    "Confidence-gated retrieval policy",
    "Oracle retrieval action",
]

BUCKET_COLUMNS = [
    "dataset",
    "split",
    "bucket_feature",
    "bucket",
    "method",
    "n_queries",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "reward",
    "reward_delta_vs_train_best",
    "reward_delta_vs_heuristic",
    "rewrite_cost",
    "retrieval_calls",
    "oracle_margin",
    "oracle_tie_count",
]

ACTION_COLUMNS = [
    "dataset",
    "split",
    "bucket_feature",
    "bucket",
    "n_queries",
    "bm25_action_rate",
    "dense_action_rate",
    "hybrid_action_rate",
    "keyword_action_rate",
    "mean_reward",
    "mean_retrieval_calls",
]


def export_complexity_diagnostics(
    dataset: str,
    detailed_csv: Path,
    output_csv: Path,
    action_distribution_csv: Path,
    split: str = "test",
) -> tuple[Path, Path]:
    detailed = pd.read_csv(detailed_csv)
    rows = detailed[detailed["split"] == split].copy() if "split" in detailed.columns else detailed.copy()
    rows = _with_oracle_diagnostics(rows)
    features = [feature for feature in COMPLEXITY_FEATURES if feature in rows.columns]

    bucket_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for feature in features:
        bucketed = _assign_feature_buckets(rows, feature)
        for bucket in ["low", "mid", "high"]:
            bucket_rows.extend(_bucket_method_rows(bucketed[bucketed["_bucket"] == bucket], dataset, split, feature, bucket))
            action_row = _action_distribution_row(bucketed[bucketed["_bucket"] == bucket], dataset, split, feature, bucket)
            if action_row is not None:
                action_rows.append(action_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    action_distribution_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bucket_rows, columns=BUCKET_COLUMNS).to_csv(output_csv, index=False)
    pd.DataFrame(action_rows, columns=ACTION_COLUMNS).to_csv(action_distribution_csv, index=False)
    return output_csv, action_distribution_csv


def _assign_feature_buckets(rows: pd.DataFrame, feature: str) -> pd.DataFrame:
    bucketed = rows.copy()
    query_features = (
        rows[["qid", feature]]
        .assign(**{feature: pd.to_numeric(rows[feature], errors="coerce")})
        .groupby("qid", sort=False, as_index=False)[feature]
        .first()
    )
    values = pd.to_numeric(query_features[feature], errors="coerce")
    query_features["_bucket"] = _bucket_values(values)
    return bucketed.merge(query_features[["qid", "_bucket"]], on="qid", how="left")


def _bucket_values(values: pd.Series) -> pd.Series:
    if values.dropna().nunique() >= 3:
        try:
            return pd.qcut(values, q=3, labels=["low", "mid", "high"])
        except ValueError:
            pass
    median = values.median()
    if pd.isna(median):
        return pd.Series(pd.NA, index=values.index, dtype="object")
    buckets = []
    for value in values:
        if pd.isna(value):
            buckets.append(pd.NA)
        elif value <= median:
            buckets.append("low")
        else:
            buckets.append("high")
    return pd.Series(buckets, index=values.index, dtype="object")


def _bucket_method_rows(
    bucket_rows: pd.DataFrame,
    dataset: str,
    split: str,
    feature: str,
    bucket: str,
) -> list[dict[str, object]]:
    if bucket_rows.empty:
        return []
    present_methods = [method for method in METHODS if method in set(bucket_rows["method"])]
    summaries = {
        method: _method_summary(bucket_rows[bucket_rows["method"] == method])
        for method in present_methods
    }

    rows = []
    for method in present_methods:
        method_rows = bucket_rows[bucket_rows["method"] == method]
        summary = summaries[method]
        reward = summary["reward"]
        method_qids = set(method_rows["qid"])
        train_best_reward = _method_reward_for_qids(bucket_rows, "Train-best retrieval action", method_qids)
        heuristic_reward = _method_reward_for_qids(bucket_rows, "Heuristic retrieval router", method_qids)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "bucket_feature": feature,
                "bucket": bucket,
                "method": method,
                **summary,
                "reward_delta_vs_train_best": _delta(reward, train_best_reward),
                "reward_delta_vs_heuristic": _delta(reward, heuristic_reward),
            }
        )
    return rows


def _method_reward_for_qids(bucket_rows: pd.DataFrame, method: str, qids: set[object]) -> float | None:
    aligned_rows = bucket_rows[(bucket_rows["method"] == method) & (bucket_rows["qid"].isin(qids))]
    return _mean(aligned_rows, "reward")


def _method_summary(rows: pd.DataFrame) -> dict[str, object]:
    return {
        "n_queries": int(rows["qid"].nunique()),
        "recall_at_5": _mean(rows, "recall_at_5"),
        "mrr": _mean(rows, "mrr"),
        "ndcg_at_5": _mean(rows, "ndcg_at_5"),
        "reward": _mean(rows, "reward"),
        "rewrite_cost": _mean(rows, "rewrite_cost"),
        "retrieval_calls": _mean(rows, "retrieval_calls"),
        "oracle_margin": _mean(rows, "_oracle_margin"),
        "oracle_tie_count": _mean(rows, "_oracle_tie_count"),
    }


def _action_distribution_row(
    bucket_rows: pd.DataFrame,
    dataset: str,
    split: str,
    feature: str,
    bucket: str,
) -> dict[str, object] | None:
    policy = bucket_rows[bucket_rows["method"] == "Selective retrieval policy"]
    if policy.empty:
        return None
    actions = policy["action"].fillna("").astype(str)
    return {
        "dataset": dataset,
        "split": split,
        "bucket_feature": feature,
        "bucket": bucket,
        "n_queries": int(policy["qid"].nunique()),
        "bm25_action_rate": float(actions.str.startswith("bm25").mean()),
        "dense_action_rate": float(actions.str.startswith("dense").mean()),
        "hybrid_action_rate": float(actions.str.startswith("hybrid").mean()),
        "keyword_action_rate": float(actions.str.contains("keyword", regex=False).mean()),
        "mean_reward": _mean(policy, "reward"),
        "mean_retrieval_calls": _mean(policy, "retrieval_calls"),
    }


def _with_oracle_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        rows["_oracle_margin"] = pd.Series(dtype=float)
        rows["_oracle_tie_count"] = pd.Series(dtype=float)
        return rows

    diagnostics = []
    for qid, group in rows.groupby("qid", sort=False):
        oracle_reward = _first_reward(group, "Oracle retrieval action")
        train_best_reward = _first_reward(group, "Train-best retrieval action")
        rewards = pd.to_numeric(group["reward"], errors="coerce").dropna()
        max_reward = rewards.max() if not rewards.empty else pd.NA
        diagnostics.append(
            {
                "qid": qid,
                "_oracle_margin": _oracle_margin(group, oracle_reward, train_best_reward),
                "_oracle_tie_count": int((rewards == max_reward).sum()) if not pd.isna(max_reward) else pd.NA,
            }
        )
    return rows.merge(pd.DataFrame(diagnostics), on="qid", how="left")


def _oracle_margin(group: pd.DataFrame, oracle_reward: float | None, train_best_reward: float | None) -> float | None:
    if "oracle_reward_margin" in group.columns:
        value = pd.to_numeric(group["oracle_reward_margin"], errors="coerce").dropna()
        if not value.empty:
            return float(value.iloc[0])
    if oracle_reward is None or train_best_reward is None:
        return None
    return float(oracle_reward - train_best_reward)


def _first_reward(group: pd.DataFrame, method: str) -> float | None:
    rows = group[group["method"] == method]
    if rows.empty:
        return None
    value = pd.to_numeric(rows["reward"], errors="coerce").dropna()
    if value.empty:
        return None
    return float(value.iloc[0])


def _delta(value: object, baseline: float | None) -> float | None:
    if baseline is None or pd.isna(value):
        return None
    return float(value) - baseline


def _mean(rows: pd.DataFrame, column: str) -> float | None:
    if column not in rows.columns:
        return None
    value = pd.to_numeric(rows[column], errors="coerce").mean()
    return None if pd.isna(value) else float(value)
