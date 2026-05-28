from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.complexity_diagnostics import export_complexity_diagnostics


def test_export_complexity_diagnostics_writes_bucket_and_action_tables(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "complexity.csv"
    action_distribution_csv = tmp_path / "actions.csv"
    pd.DataFrame(
        [
            _row("train", "q0", "Train-best retrieval action", "bm25_keep", 0.9, 3, 0.20, 0.20),
            _row("train", "q0", "Heuristic retrieval router", "bm25_keep", 0.8, 3, 0.20, 0.20),
            _row("train", "q0", "Selective retrieval policy", "bm25_keep", 0.7, 3, 0.20, 0.20),
            _row("train", "q0", "Oracle retrieval action", "bm25_keep", 1.0, 3, 0.20, 0.20),
            _row("test", "q1", "Train-best retrieval action", "bm25_keep", 0.4, 4, 0.10, 0.30),
            _row("test", "q1", "Heuristic retrieval router", "bm25_keep", 0.5, 4, 0.10, 0.30),
            _row("test", "q1", "Oracle retrieval action", "dense_keep", 0.9, 4, 0.10, 0.30),
            _row("test", "q2", "Train-best retrieval action", "bm25_keep", 0.8, 8, 0.10, 0.20),
            _row("test", "q2", "Heuristic retrieval router", "dense_keep", 0.6, 8, 0.10, 0.20),
            _row("test", "q2", "Selective retrieval policy", "hybrid_keyword", 1.0, 8, 0.10, 0.20),
            _row("test", "q2", "Oracle retrieval action", "hybrid_keyword", 1.0, 8, 0.10, 0.20),
            _row("test", "q3", "Train-best retrieval action", "bm25_keep", 0.3, 12, 0.90, 0.10),
            _row("test", "q3", "Heuristic retrieval router", "bm25_keyword", 0.4, 12, 0.90, 0.10),
            _row("test", "q3", "Selective retrieval policy", "bm25_keyword", 0.2, 12, 0.90, 0.10),
            _row("test", "q3", "Oracle retrieval action", "dense_keyword", 0.6, 12, 0.90, 0.10),
            _row("test", "q4", "Train-best retrieval action", "bm25_keep", 0.1, 16, None, 0.10, bm25_top1=None),
            _row("test", "q4", "Heuristic retrieval router", "bm25_keep", 0.1, 16, None, 0.10, bm25_top1=None),
            _row("test", "q4", "Selective retrieval policy", "dense_keep", 0.1, 16, None, 0.10, bm25_top1=None),
            _row("test", "q4", "Oracle retrieval action", "dense_keep", 0.1, 16, None, 0.10, bm25_top1=None),
        ]
    ).to_csv(detailed_csv, index=False)

    bucket_path, action_path = export_complexity_diagnostics(
        dataset="toy",
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        action_distribution_csv=action_distribution_csv,
    )

    buckets = pd.read_csv(bucket_path)
    actions = pd.read_csv(action_path)

    assert {
        "dataset",
        "split",
        "bucket_feature",
        "bucket",
        "method",
        "n_queries",
        "reward",
        "reward_delta_vs_train_best",
        "reward_delta_vs_heuristic",
        "retrieval_calls",
    }.issubset(buckets.columns)
    assert {
        "Train-best retrieval action",
        "Heuristic retrieval router",
        "Selective retrieval policy",
        "Oracle retrieval action",
    }.issubset(set(buckets["method"]))
    assert {"dense_action_rate", "hybrid_action_rate"}.issubset(actions.columns)
    assert actions["n_queries"].sum() > 0
    assert not buckets["bucket"].isna().any()
    assert "nan" not in set(buckets["bucket"].astype(str).str.lower())

    top1_buckets = buckets[buckets["bucket_feature"] == "state_bm25_top1"]
    assert set(top1_buckets["bucket"]) == {"low", "mid", "high"}
    selective_high = top1_buckets[
        (top1_buckets["bucket"] == "high") & (top1_buckets["method"] == "Selective retrieval policy")
    ].iloc[0]
    assert round(float(selective_high["reward_delta_vs_train_best"]), 12) == 0.2

    high_actions = actions[(actions["bucket_feature"] == "state_bm25_top1") & (actions["bucket"] == "high")].iloc[0]
    assert float(high_actions["bm25_action_rate"]) == 0.0
    assert float(high_actions["dense_action_rate"]) == 0.0
    assert float(high_actions["hybrid_action_rate"]) == 1.0
    assert float(high_actions["keyword_action_rate"]) == 1.0

    selective_top1_count = top1_buckets[top1_buckets["method"] == "Selective retrieval policy"]["n_queries"].sum()
    assert selective_top1_count == 2
    high_train_best = top1_buckets[
        (top1_buckets["bucket"] == "high") & (top1_buckets["method"] == "Train-best retrieval action")
    ].iloc[0]
    assert int(high_train_best["n_queries"]) == 1

    margin_low_selective = buckets[
        (buckets["bucket_feature"] == "predicted_action_margin")
        & (buckets["bucket"] == "low")
        & (buckets["method"] == "Selective retrieval policy")
    ].iloc[0]
    assert round(float(margin_low_selective["reward_delta_vs_train_best"]), 12) == 0.2


def _row(
    split: str,
    qid: str,
    method: str,
    action: str,
    reward: float,
    question_length: int,
    predicted_margin: float | None,
    oracle_margin: float,
    bm25_top1: float | None = None,
) -> dict[str, object]:
    if bm25_top1 is None and qid != "q4":
        bm25_top1 = reward
    return {
        "split": split,
        "qid": qid,
        "question": f"question {qid}",
        "method": method,
        "action": action,
        "recall_at_5": reward / 2,
        "mrr": reward / 3,
        "ndcg_at_5": reward / 4,
        "reward": reward,
        "rewrite_cost": 0.05 if "keyword" in action else 0.0,
        "retrieval_calls": 1,
        "state_question_length": question_length,
        "state_bm25_top1": bm25_top1,
        "predicted_action_margin": predicted_margin,
        "oracle_reward_margin": oracle_margin,
        "action_reward_std": 0.1 + reward,
    }
