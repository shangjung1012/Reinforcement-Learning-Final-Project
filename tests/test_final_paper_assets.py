from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.reports.final_paper_assets import PAPER_TABLE_COLUMNS, export_final_paper_assets


def test_export_final_paper_assets_writes_tables_and_figures(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    outputs = export_final_paper_assets(
        root=tmp_path,
        results_dir=tmp_path / "outputs" / "results",
        figures_dir=tmp_path / "outputs" / "figures",
    )

    table = pd.read_csv(outputs["main_results_csv"])
    assert list(table.columns) == PAPER_TABLE_COLUMNS
    assert set(table["dataset"]) == {"scifact", "nfcorpus"}
    assert set(table["method_group"]) >= {"fixed", "heuristic", "bandit", "learned", "constrained", "oracle"}

    scifact_policy = table[(table["dataset"] == "scifact") & (table["method"] == "Selective retrieval policy")].iloc[0]
    assert scifact_policy["objective"] == "reward"
    assert scifact_policy["objective_value"] == 1.2
    assert scifact_policy["delta_vs_train_best"] == 0.2
    assert scifact_policy["ci_low"] == 0.03
    assert scifact_policy["ci_high"] == 0.37
    scifact_methods = set(table[table["dataset"] == "scifact"]["method"])
    assert "Best selected-action bandit baseline" in scifact_methods
    assert "Linear Thompson retrieval policy" not in scifact_methods

    scifact_constrained = table[
        (table["dataset"] == "scifact") & (table["method"] == "Constrained policy lambda=0.03")
    ].iloc[0]
    assert scifact_constrained["objective"] == "utility_lambda_0.03"
    assert scifact_constrained["objective_value"] == 1.15
    assert scifact_constrained["delta_vs_train_best"] == 0.07
    assert scifact_constrained["evidence_artifact_id"] == "scifact_constrained_policy_bootstrap"

    assert outputs["main_results_tex"].read_text(encoding="utf-8").startswith("\\begin{tabular}")
    for key in [
        "reward_delta_ci_png",
        "cost_reward_frontier_png",
        "ope_estimator_error_png",
        "linucb_comparison_png",
    ]:
        assert outputs[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _write_inputs(root: Path) -> None:
    results = root / "outputs" / "results"
    results.mkdir(parents=True, exist_ok=True)

    for dataset, policy_reward, train_reward in [
        ("scifact", 1.2, 1.0),
        ("nfcorpus", 0.5, 0.4),
    ]:
        _summary(results / f"{dataset}_retrieval_policy_summary.csv", policy_reward, train_reward)
        _bootstrap(results / f"{dataset}_bootstrap_diagnostics.csv", dataset, policy_reward - train_reward)
        _linucb(results / f"{dataset}_linucb_baseline_summary.csv", train_reward + 0.01, train_reward)
        _constrained_sweep(results / f"{dataset}_constrained_policy_sweep.csv", dataset)
        _constrained_bootstrap(results / f"{dataset}_constrained_policy_bootstrap.csv", dataset)
        _ope(results / f"{dataset}_ope_stability.csv", dataset)

    claims = pd.DataFrame(
        [
            {
                "claim_id": "scifact_policy_reward_gain",
                "primary_dataset": "scifact",
                "metric": "reward",
                "value": 1.2,
                "baseline_value": 1.0,
                "delta": 0.2,
                "ci_low": 0.03,
                "ci_high": 0.37,
                "evidence_artifact_id": "scifact_bootstrap",
            },
            {
                "claim_id": "nfcorpus_policy_reward_gain",
                "primary_dataset": "nfcorpus",
                "metric": "reward",
                "value": 0.5,
                "baseline_value": 0.4,
                "delta": 0.1,
                "ci_low": 0.01,
                "ci_high": 0.19,
                "evidence_artifact_id": "nfcorpus_bootstrap",
            },
        ]
    )
    claims.to_csv(results / "final_claims_matrix.csv", index=False)


def _summary(path: Path, policy_reward: float, train_reward: float) -> None:
    pd.DataFrame(
        [
            {
                "method": "Train-best retrieval action",
                "recall_at_5": 0.1,
                "mrr": 0.2,
                "ndcg_at_5": 0.3,
                "reward": train_reward,
                "retrieval_calls": 2.0,
            },
            {
                "method": "Heuristic retrieval router",
                "recall_at_5": 0.15,
                "mrr": 0.25,
                "ndcg_at_5": 0.35,
                "reward": train_reward + 0.05,
                "retrieval_calls": 1.2,
            },
            {
                "method": "Selective retrieval policy",
                "recall_at_5": 0.2,
                "mrr": 0.3,
                "ndcg_at_5": 0.4,
                "reward": policy_reward,
                "retrieval_calls": 1.5,
            },
            {
                "method": "Oracle retrieval action",
                "recall_at_5": 0.3,
                "mrr": 0.4,
                "ndcg_at_5": 0.5,
                "reward": policy_reward + 0.2,
                "retrieval_calls": 1.1,
            },
        ]
    ).to_csv(path, index=False)


def _bootstrap(path: Path, dataset: str, delta: float) -> None:
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "metric": "reward",
                "mean_policy": 1.0 + delta,
                "mean_baseline": 1.0,
                "mean_delta": delta,
                "ci_lower": 0.03 if dataset == "scifact" else 0.01,
                "ci_upper": 0.37 if dataset == "scifact" else 0.19,
            }
        ]
    ).to_csv(path, index=False)


def _linucb(path: Path, reward: float, train_reward: float) -> None:
    pd.DataFrame(
        [
            {"method": "LinUCB retrieval policy", "reward": reward, "retrieval_calls": 1.3},
            {"method": "Epsilon-greedy retrieval policy", "reward": reward - 0.02, "retrieval_calls": 1.1},
            {"method": "Linear Thompson retrieval policy", "reward": reward + 0.02, "retrieval_calls": 1.4},
            {"method": "Train-best retrieval action", "reward": train_reward, "retrieval_calls": 2.0},
        ]
    ).to_csv(path, index=False)


def _constrained_sweep(path: Path, dataset: str) -> None:
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "call_penalty": 0.03,
                "policy_utility": 1.15,
                "policy_reward": 1.18,
                "policy_recall_at_5": 0.22,
                "policy_mrr": 0.31,
                "policy_ndcg_at_5": 0.42,
                "policy_retrieval_calls": 1.4,
                "train_best_utility": 1.08,
                "train_best_retrieval_calls": 2.0,
            }
        ]
    ).to_csv(path, index=False)


def _constrained_bootstrap(path: Path, dataset: str) -> None:
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "call_penalty": 0.03,
                "policy_utility_mean": 1.15,
                "train_best_utility_mean": 1.08,
                "utility_delta_mean": 0.07,
                "utility_delta_ci_low": 0.02,
                "utility_delta_ci_high": 0.11,
                "policy_calls_mean": 1.4,
                "train_best_calls_mean": 2.0,
                "call_delta_mean": -0.6,
            }
        ]
    ).to_csv(path, index=False)


def _ope(path: Path, dataset: str) -> None:
    rows = []
    for estimator, error in [
        ("direct_method", 0.05),
        ("ips", 0.1),
        ("snips", 0.08),
        ("doubly_robust", 0.04),
    ]:
        rows.append(
            {
                "dataset": dataset,
                "behavior_policy": "uniform",
                "target_method": "Selective retrieval policy",
                "estimator": estimator,
                "mean_absolute_error": error,
                "ci95_absolute_error_low": max(error - 0.01, 0.0),
                "ci95_absolute_error_high": error + 0.01,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
