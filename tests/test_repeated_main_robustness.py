from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.experiments.repeated_main_robustness import export_repeated_main_robustness


def test_export_repeated_main_robustness_summarizes_seed_deltas(tmp_path: Path) -> None:
    rows = []
    for seed, policy_reward, bandit_reward, constrained_utility in [
        (41, 1.10, 1.04, 1.08),
        (42, 1.05, 1.06, 1.03),
    ]:
        policy_csv = tmp_path / f"policy_{seed}.csv"
        bandit_csv = tmp_path / f"bandit_{seed}.csv"
        constrained_csv = tmp_path / f"constrained_{seed}.csv"
        _policy_summary(policy_csv, train_best=1.00, heuristic=0.98, policy=policy_reward, oracle=1.20)
        _bandit_summary(bandit_csv, linucb=1.01, thompson=bandit_reward, train_best=1.00)
        _constrained_sweep(constrained_csv, utility=constrained_utility, train_best=1.00)
        rows.append(
            {
                "dataset": "toy",
                "seed": seed,
                "policy_summary_csv": policy_csv,
                "bandit_summary_csv": bandit_csv,
                "constrained_sweep_csv": constrained_csv,
            }
        )
    manifest_csv = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)

    per_seed_csv = tmp_path / "per_seed.csv"
    aggregate_csv = tmp_path / "aggregate.csv"
    export_repeated_main_robustness(
        manifest_csv=manifest_csv,
        per_seed_csv=per_seed_csv,
        aggregate_csv=aggregate_csv,
        call_penalty=0.03,
    )

    per_seed = pd.read_csv(per_seed_csv)
    aggregate = pd.read_csv(aggregate_csv).iloc[0]
    assert list(per_seed["best_selected_action_bandit_method"]) == [
        "Linear Thompson retrieval policy",
        "Linear Thompson retrieval policy",
    ]
    assert list(per_seed["selective_delta"].round(2)) == [0.10, 0.05]
    assert list(per_seed["best_selected_action_bandit_delta"].round(2)) == [0.04, 0.06]
    assert aggregate["dataset"] == "toy"
    assert aggregate["n_seeds"] == 2
    assert aggregate["selective_win_rate"] == 1.0
    assert aggregate["best_selected_action_bandit_win_rate"] == 1.0
    assert round(float(aggregate["selective_delta_mean"]), 3) == 0.075
    assert round(float(aggregate["constrained_delta_mean"]), 3) == 0.055


def _policy_summary(path: Path, *, train_best: float, heuristic: float, policy: float, oracle: float) -> None:
    pd.DataFrame(
        [
            {"method": "Train-best retrieval action", "reward": train_best, "recall_at_5": 0.1, "mrr": 0.2, "retrieval_calls": 1.0},
            {"method": "Heuristic retrieval router", "reward": heuristic, "recall_at_5": 0.1, "mrr": 0.2, "retrieval_calls": 1.0},
            {"method": "Selective retrieval policy", "reward": policy, "recall_at_5": 0.2, "mrr": 0.3, "retrieval_calls": 1.2},
            {"method": "Oracle retrieval action", "reward": oracle, "recall_at_5": 0.3, "mrr": 0.4, "retrieval_calls": 1.0},
        ]
    ).to_csv(path, index=False)


def _bandit_summary(path: Path, *, linucb: float, thompson: float, train_best: float) -> None:
    pd.DataFrame(
        [
            {"method": "LinUCB retrieval policy", "reward": linucb, "recall_at_5": 0.1, "mrr": 0.2, "retrieval_calls": 1.1},
            {
                "method": "Linear Thompson retrieval policy",
                "reward": thompson,
                "recall_at_5": 0.2,
                "mrr": 0.3,
                "retrieval_calls": 1.0,
            },
            {"method": "Train-best retrieval action", "reward": train_best, "recall_at_5": 0.1, "mrr": 0.2, "retrieval_calls": 1.0},
        ]
    ).to_csv(path, index=False)


def _constrained_sweep(path: Path, *, utility: float, train_best: float) -> None:
    pd.DataFrame(
        [
            {
                "call_penalty": 0.03,
                "policy_utility": utility,
                "train_best_utility": train_best,
                "policy_retrieval_calls": 1.1,
            }
        ]
    ).to_csv(path, index=False)
