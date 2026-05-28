from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.bandit_baselines import (
    EpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    export_linucb_baseline_from_evals,
    run_beir_linucb_baseline,
)


def test_linucb_policy_learns_from_logged_chosen_rewards() -> None:
    actions = ["left", "right"]
    features = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    rewards = {
        "left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "right": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }

    policy = LinUCBPolicy(actions=actions, alpha=0.5, l2=1.0)
    history = policy.fit(features, rewards)

    assert len(history) == len(features)
    assert set(history["chosen_action"]) == set(actions)
    assert policy.predict(np.asarray([1.0, 1.0])) == "right"


def test_epsilon_greedy_policy_replays_selected_action_feedback() -> None:
    actions = ["left", "right"]
    features = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    rewards = {
        "left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "right": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }

    policy = EpsilonGreedyPolicy(actions=actions, epsilon=0.0, l2=1.0, seed=7)
    history = policy.fit(features, rewards)

    assert len(history) == len(features)
    assert set(history["chosen_action"]) <= set(actions)
    assert policy.predict(np.asarray([1.0, 1.0])) == "right"


def test_linear_thompson_policy_replays_selected_action_feedback() -> None:
    actions = ["left", "right"]
    features = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    rewards = {
        "left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "right": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }

    policy = LinearThompsonSamplingPolicy(actions=actions, posterior_scale=0.1, l2=1.0, seed=3)
    history = policy.fit(features, rewards)

    assert len(history) == len(features)
    assert set(history["chosen_action"]) == set(actions)
    assert policy.predict(np.asarray([1.0, 1.0])) == "right"


def test_export_linucb_baseline_from_evals_writes_summary(tmp_path: Path) -> None:
    actions = ["left", "right"]
    train_evals = [
        _eval([1.0, 0.0], {"left": 1.0, "right": 0.0}),
        _eval([1.0, 1.0], {"left": 0.0, "right": 1.0}),
        _eval([1.0, 1.0], {"left": 0.0, "right": 1.0}),
        _eval([1.0, 1.0], {"left": 0.0, "right": 1.0}),
    ]
    test_evals = [
        _eval([1.0, 1.0], {"left": 0.0, "right": 1.0}),
        _eval([1.0, 1.0], {"left": 0.0, "right": 1.0}),
    ]

    summary_csv = export_linucb_baseline_from_evals(
        train_evals=train_evals,
        test_evals=test_evals,
        actions=actions,
        output_csv=tmp_path / "summary.csv",
        history_csv=tmp_path / "history.csv",
        alpha=0.5,
    )

    summary = pd.read_csv(summary_csv)
    history = pd.read_csv(tmp_path / "history.csv")
    assert {
        "LinUCB retrieval policy",
        "Epsilon-greedy retrieval policy",
        "Linear Thompson retrieval policy",
        "Train-best retrieval action",
        "Oracle retrieval action",
    } <= set(summary["method"])
    assert float(summary.loc[summary["method"] == "LinUCB retrieval policy", "reward"].iloc[0]) == 1.0
    assert list(history.columns) == ["step", "chosen_action", "reward", "oracle_reward", "regret"]


def test_run_beir_linucb_baseline_writes_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "scifact"
    output_dir = tmp_path / "outputs"
    _write_beir_fixture(data_dir, count=10)

    summary_csv = run_beir_linucb_baseline(
        dataset="scifact",
        data_path=data_dir,
        output_dir=output_dir,
        num_train_examples=6,
        num_test_examples=4,
        seed=2,
        pool_size=4,
        embedder_name="fake",
        alpha=0.5,
    )

    summary = pd.read_csv(summary_csv)
    assert Path(summary_csv).name == "scifact_linucb_baseline_summary.csv"
    assert (output_dir / "results" / "scifact_linucb_baseline_history.csv").exists()
    assert {
        "LinUCB retrieval policy",
        "Epsilon-greedy retrieval policy",
        "Linear Thompson retrieval policy",
        "Train-best retrieval action",
        "Oracle retrieval action",
    } <= set(summary["method"])


def _eval(features: list[float], rewards: dict[str, float]) -> dict[str, object]:
    return {
        "features": np.asarray(features, dtype=float),
        "actions": {
            action: {
                "reward": reward,
                "recall_at_5": reward,
                "mrr": reward,
                "ndcg_at_5": reward,
                "rewrite_cost": 0.0,
                "retrieval_calls": 1.0,
            }
            for action, reward in rewards.items()
        },
    }


def _write_beir_fixture(data_dir: Path, count: int) -> None:
    qrels_dir = data_dir / "qrels"
    qrels_dir.mkdir(parents=True)
    corpus = []
    queries = []
    qrels = ["query-id\tcorpus-id\tscore"]
    for i in range(count):
        corpus.append(f'{{"_id": "d{i}", "title": "Doc {i}", "text": "evidence token{i}"}}')
        queries.append(f'{{"_id": "q{i}", "text": "claim token{i}"}}')
        qrels.append(f"q{i}\td{i}\t1")
    (data_dir / "corpus.jsonl").write_text("\n".join(corpus), encoding="utf-8")
    (data_dir / "queries.jsonl").write_text("\n".join(queries), encoding="utf-8")
    (qrels_dir / "test.tsv").write_text("\n".join(qrels), encoding="utf-8")
    (qrels_dir / "train.tsv").write_text("\n".join(qrels), encoding="utf-8")
