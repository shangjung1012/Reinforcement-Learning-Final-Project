from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.artifact_index import final_project_artifact_specs
from selective_rag_rl.final_claims import CLAIM_COLUMNS, export_final_claims_matrix


def test_export_final_claims_matrix_derives_key_policy_deltas(tmp_path: Path) -> None:
    _write_minimal_claim_inputs(tmp_path)

    output_csv = tmp_path / "outputs" / "results" / "final_claims_matrix.csv"
    exported = export_final_claims_matrix(root=tmp_path, output_csv=output_csv)

    claims = pd.read_csv(exported).set_index("claim_id")

    assert list(pd.read_csv(exported).columns) == CLAIM_COLUMNS
    assert len(claims) >= 8

    scifact = claims.loc["scifact_policy_reward_gain"]
    assert scifact["primary_dataset"] == "scifact"
    assert scifact["metric"] == "reward"
    assert scifact["value"] == 1.2
    assert scifact["baseline_value"] == 1.0
    assert scifact["delta"] == 0.2
    assert scifact["ci_low"] == 0.03
    assert scifact["ci_high"] == 0.37
    assert scifact["call_delta"] == -0.5
    assert scifact["evidence_artifact_id"] == "scifact_bootstrap"

    constrained = claims.loc["scifact_constrained_bootstrap_gain"]
    assert constrained["metric"] == "utility_delta"
    assert constrained["value"] == 0.07
    assert constrained["ci_low"] == 0.02
    assert constrained["ci_high"] == 0.11
    assert constrained["call_delta"] == -0.4
    assert constrained["evidence_artifact_id"] == "scifact_constrained_policy_bootstrap"

    ope = claims.loc["nfcorpus_ope_dr_stability"]
    assert ope["metric"] == "mean_absolute_error"
    assert ope["value"] == 0.03
    assert ope["baseline"] == "Direct method"
    assert ope["baseline_value"] == 0.05
    assert ope["delta"] == -0.02


def test_final_claims_matrix_references_indexed_artifacts(tmp_path: Path) -> None:
    _write_minimal_claim_inputs(tmp_path)

    exported = export_final_claims_matrix(
        root=tmp_path,
        output_csv=tmp_path / "outputs" / "results" / "final_claims_matrix.csv",
    )
    indexed_ids = {spec.artifact_id for spec in final_project_artifact_specs(Path("."))}
    claims = pd.read_csv(exported)

    assert set(claims["evidence_artifact_id"]).issubset(indexed_ids)
    assert claims["claim"].str.len().min() > 20
    assert claims["presentation_use"].str.len().min() > 10
    assert not claims[["claim_type", "producer_command", "evidence_path"]].isna().any().any()


def _write_minimal_claim_inputs(root: Path) -> None:
    results = root / "outputs" / "results"
    results.mkdir(parents=True, exist_ok=True)

    _summary(results / "scifact_retrieval_policy_summary.csv", policy_reward=1.2, train_reward=1.0, policy_calls=1.5, train_calls=2.0)
    _summary(results / "nfcorpus_retrieval_policy_summary.csv", policy_reward=0.5, train_reward=0.4, policy_calls=1.0, train_calls=1.0)
    _bootstrap(results / "scifact_bootstrap_diagnostics.csv", dataset="scifact", mean_delta=0.2, ci_low=0.03, ci_high=0.37)
    _bootstrap(results / "nfcorpus_bootstrap_diagnostics.csv", dataset="nfcorpus", mean_delta=0.1, ci_low=0.01, ci_high=0.19)
    _linucb(results / "scifact_linucb_baseline_summary.csv", reward=1.01, train_reward=1.0, calls=1.2, train_calls=2.0)
    _linucb(results / "nfcorpus_linucb_baseline_summary.csv", reward=0.43, train_reward=0.4, calls=1.3, train_calls=1.0)
    _constrained(results / "scifact_constrained_policy_bootstrap.csv", dataset="scifact", utility_delta=0.07, calls_delta=-0.4)
    _constrained(results / "nfcorpus_constrained_policy_bootstrap.csv", dataset="nfcorpus", utility_delta=0.06, calls_delta=0.0)
    _ope(results / "scifact_ope_stability.csv", dataset="scifact", dr_error=0.09, dm_error=0.02)
    _ope(results / "nfcorpus_ope_stability.csv", dataset="nfcorpus", dr_error=0.03, dm_error=0.05)


def _summary(path: Path, *, policy_reward: float, train_reward: float, policy_calls: float, train_calls: float) -> None:
    pd.DataFrame(
        [
            {
                "method": "Train-best retrieval action",
                "recall_at_5": 0.1,
                "mrr": 0.2,
                "reward": train_reward,
                "retrieval_calls": train_calls,
            },
            {
                "method": "Selective retrieval policy",
                "recall_at_5": 0.2,
                "mrr": 0.3,
                "reward": policy_reward,
                "retrieval_calls": policy_calls,
            },
        ]
    ).to_csv(path, index=False)


def _linucb(path: Path, *, reward: float, train_reward: float, calls: float, train_calls: float) -> None:
    pd.DataFrame(
        [
            {"method": "LinUCB retrieval policy", "reward": reward, "retrieval_calls": calls},
            {"method": "Train-best retrieval action", "reward": train_reward, "retrieval_calls": train_calls},
        ]
    ).to_csv(path, index=False)


def _bootstrap(path: Path, *, dataset: str, mean_delta: float, ci_low: float, ci_high: float) -> None:
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "comparison": "Selective retrieval policy - Train-best retrieval action",
                "metric": "reward",
                "mean_policy": 1.2,
                "mean_baseline": 1.0,
                "mean_delta": mean_delta,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "prob_delta_gt_0": 0.99,
            }
        ]
    ).to_csv(path, index=False)


def _constrained(path: Path, *, dataset: str, utility_delta: float, calls_delta: float) -> None:
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "call_penalty": 0.03,
                "policy_utility_mean": 1.1,
                "train_best_utility_mean": 1.03,
                "utility_delta_mean": utility_delta,
                "utility_delta_ci_low": 0.02,
                "utility_delta_ci_high": 0.11,
                "policy_calls_mean": 1.6,
                "train_best_calls_mean": 2.0,
                "call_delta_mean": calls_delta,
            }
        ]
    ).to_csv(path, index=False)


def _ope(path: Path, *, dataset: str, dr_error: float, dm_error: float) -> None:
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "behavior_policy": "uniform",
                "target_method": "Selective retrieval policy",
                "estimator": "direct_method",
                "mean_absolute_error": dm_error,
                "mean_effective_sample_size": 50.0,
            },
            {
                "dataset": dataset,
                "behavior_policy": "uniform",
                "target_method": "Selective retrieval policy",
                "estimator": "ips",
                "mean_absolute_error": 0.1,
                "mean_effective_sample_size": 50.0,
            },
            {
                "dataset": dataset,
                "behavior_policy": "uniform",
                "target_method": "Selective retrieval policy",
                "estimator": "doubly_robust",
                "mean_absolute_error": dr_error,
                "mean_effective_sample_size": 50.0,
            },
            {
                "dataset": dataset,
                "behavior_policy": "train_best_eps_0.2",
                "target_method": "Selective retrieval policy",
                "estimator": "ips",
                "mean_absolute_error": 0.2,
                "mean_effective_sample_size": 12.0,
            },
        ]
    ).to_csv(path, index=False)
