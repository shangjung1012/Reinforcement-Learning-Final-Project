from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.off_policy_evaluation import (
    BehaviorPolicySpec,
    behavior_probabilities,
    estimate_off_policy_value,
    export_ope_diagnostics,
    export_ope_stability_diagnostics,
)


def test_estimate_off_policy_value_computes_ips_snips_and_dr() -> None:
    logged = pd.DataFrame(
        [
            {"logged_action": "a", "logged_reward": 1.0, "propensity": 0.5},
            {"logged_action": "b", "logged_reward": 0.0, "propensity": 0.5},
        ]
    )

    estimates = estimate_off_policy_value(
        logged=logged,
        target_actions=["a", "a"],
        q_target=np.asarray([0.8, 0.2]),
        q_logged=np.asarray([0.8, 0.6]),
    )

    assert estimates["ips"] == 1.0
    assert estimates["snips"] == 1.0
    assert estimates["doubly_robust"] == 0.7
    assert estimates["match_rate"] == 0.5
    assert estimates["effective_sample_size"] == 1.0


def test_estimate_off_policy_value_reports_no_coverage_when_actions_never_match() -> None:
    logged = pd.DataFrame(
        [
            {"logged_action": "a", "logged_reward": 1.0, "propensity": 0.5},
            {"logged_action": "a", "logged_reward": 0.0, "propensity": 0.5},
        ]
    )

    estimates = estimate_off_policy_value(
        logged=logged,
        target_actions=["b", "b"],
        q_target=np.asarray([0.4, 0.6]),
        q_logged=np.asarray([0.9, 0.1]),
    )

    assert estimates["ips"] == 0.0
    assert np.isnan(estimates["snips"])
    assert estimates["doubly_robust"] == 0.5
    assert estimates["match_rate"] == 0.0
    assert estimates["effective_sample_size"] == 0.0


def test_behavior_probabilities_support_uniform_and_epsilon_reference_policy() -> None:
    actions = ["a", "b", "c"]

    uniform = behavior_probabilities(actions, BehaviorPolicySpec(name="uniform", kind="uniform"))
    epsilon = behavior_probabilities(
        actions,
        BehaviorPolicySpec(name="heuristic_eps_0.2", kind="reference_epsilon", epsilon=0.2),
        reference_action="b",
    )

    assert uniform == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    assert epsilon == {"a": 0.2 / 3, "b": 0.8 + 0.2 / 3, "c": 0.2 / 3}


def test_export_ope_diagnostics_from_detailed_csv(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "ope.csv"
    _toy_detailed_frame().to_csv(detailed_csv, index=False)

    exported = export_ope_diagnostics(
        dataset="toy",
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        split="test",
        target_methods=["Selective retrieval policy", "Heuristic retrieval router"],
        seed=7,
    )

    rows = pd.read_csv(exported)
    assert set(rows["behavior_policy"]) == {"uniform", "train_best_eps_0.2", "heuristic_eps_0.2"}
    assert set(rows["target_method"]) == {"Selective retrieval policy", "Heuristic retrieval router"}
    assert set(rows["estimator"]) == {"direct_method", "ips", "snips", "doubly_robust"}
    assert set(rows["n_queries"]) == {3}
    selective = rows[
        (rows["target_method"] == "Selective retrieval policy")
        & (rows["behavior_policy"] == "uniform")
    ]
    assert selective["true_value"].iloc[0] == 2 / 3
    finite_estimators = selective[selective["estimator"].isin(["direct_method", "doubly_robust"])]
    assert finite_estimators["estimated_value"].notna().all()


def test_export_ope_stability_diagnostics_aggregates_multiple_logging_seeds(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "ope_stability.csv"
    _toy_detailed_frame().to_csv(detailed_csv, index=False)

    exported = export_ope_stability_diagnostics(
        dataset="toy",
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        seeds=[7, 8, 9],
        split="test",
        target_methods=["Selective retrieval policy"],
    )

    rows = pd.read_csv(exported)
    assert set(rows["estimator"]) == {"direct_method", "ips", "snips", "doubly_robust"}
    assert set(rows["seed_count"]) == {3}
    stable_estimators = rows[rows["estimator"].isin(["direct_method", "doubly_robust"])]
    assert stable_estimators["mean_absolute_error"].notna().all()
    assert stable_estimators["std_absolute_error"].notna().all()
    assert stable_estimators["mean_effective_sample_size"].notna().all()
    assert stable_estimators["ci95_absolute_error_low"].le(stable_estimators["ci95_absolute_error_high"]).all()


def _toy_detailed_frame() -> pd.DataFrame:
    rows = []
    for split in ["train", "test"]:
        for idx in range(3):
            qid = f"{split}-{idx}"
            feature = float(idx + 1)
            action_rewards = {"a": 1.0, "b": 0.0} if idx < 2 else {"a": 0.0, "b": 1.0}
            for action, method in {"a": "Vanilla BM25", "b": "Dense original"}.items():
                rows.append(_row(split, method, action, qid, feature, action_rewards[action]))
            rows.append(_row(split, "Train-best retrieval action", "a", qid, feature, action_rewards["a"]))
            rows.append(_row(split, "Heuristic retrieval router", "b", qid, feature, action_rewards["b"]))
            rows.append(_row(split, "Selective retrieval policy", "a", qid, feature, action_rewards["a"]))
    return pd.DataFrame(rows)


def _row(split: str, method: str, action: str, qid: str, feature: float, reward: float) -> dict[str, object]:
    return {
        "split": split,
        "method": method,
        "action": action,
        "qid": qid,
        "question": f"question {qid}",
        "reward": reward,
        "recall_at_5": reward,
        "mrr": reward,
        "ndcg_at_5": reward,
        "rewrite_cost": 0.0,
        "retrieval_calls": 1.0,
        "state_question_length": feature,
        "state_capitalized_spans": 0.0,
        "state_bm25_top1": feature,
        "state_bm25_gap": 0.0,
        "state_bm25_entropy": 0.0,
    }
