from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.diagnostics.bandit_replay import (
    REPLAY_HISTORY_COLUMNS,
    REPLAY_SUMMARY_COLUMNS,
    export_bandit_replay_diagnostics_from_evals,
    train_evals_from_detailed_frame,
    write_bandit_replay_figure,
)


def test_bandit_replay_diagnostics_compare_selected_feedback_to_full_information(tmp_path: Path) -> None:
    actions = ["left", "right"]
    train_evals = [
        _eval([1.0, 0.0], {"left": 1.0, "right": 0.0}),
        _eval([1.0, 1.0], {"left": 0.0, "right": 1.0}),
        _eval([1.0, 1.0], {"left": 0.0, "right": 1.0}),
        _eval([1.0, 1.0], {"left": 0.0, "right": 1.0}),
        _eval([1.0, 1.0], {"left": 0.0, "right": 1.0}),
    ]

    history_csv = tmp_path / "history.csv"
    summary_csv = tmp_path / "summary.csv"
    exported = export_bandit_replay_diagnostics_from_evals(
        train_evals=train_evals,
        actions=actions,
        history_csv=history_csv,
        summary_csv=summary_csv,
        dataset="toy",
        alpha=0.0,
        epsilon=0.0,
        posterior_scale=0.0,
        seed=3,
        moving_average_window=2,
    )

    history = pd.read_csv(exported["history_csv"])
    summary = pd.read_csv(exported["summary_csv"]).set_index("policy")

    assert list(history.columns) == REPLAY_HISTORY_COLUMNS
    assert list(pd.read_csv(summary_csv).columns) == REPLAY_SUMMARY_COLUMNS
    assert {
        "Full-information direct method",
        "LinUCB selected-action replay",
        "Epsilon-greedy selected-action replay",
        "Linear Thompson selected-action replay",
        "Train-best fixed action",
        "Oracle action",
    } <= set(history["policy"])

    oracle = history[history["policy"] == "Oracle action"]
    assert oracle["cumulative_regret"].iloc[-1] == 0.0
    assert summary.loc["Oracle action", "final_cumulative_regret"] == 0.0

    linucb = history[history["policy"] == "LinUCB selected-action replay"]
    assert linucb["cumulative_regret"].is_monotonic_increasing
    assert linucb["moving_average_reward"].notna().all()
    assert 0.0 <= float(summary.loc["LinUCB selected-action replay", "oracle_match_rate"]) <= 1.0
    assert float(summary.loc["LinUCB selected-action replay", "action_entropy"]) >= 0.0
    assert float(summary.loc["Oracle action", "action_entropy"]) > 0.0

    figure_path = write_bandit_replay_figure(history, output_png=tmp_path / "regret.png")
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_train_evals_from_detailed_frame_reconstructs_full_information_rewards() -> None:
    frame = pd.DataFrame(
        [
            {
                "split": "train",
                "qid": "q1",
                "action": "left",
                "reward": 1.0,
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "ndcg_at_5": 1.0,
                "rewrite_cost": 0.0,
                "retrieval_calls": 1,
                "state_a": 0.1,
                "state_b": 0.2,
            },
            {
                "split": "train",
                "qid": "q1",
                "action": "right",
                "reward": 0.0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "ndcg_at_5": 0.0,
                "rewrite_cost": 0.0,
                "retrieval_calls": 1,
                "state_a": 0.1,
                "state_b": 0.2,
            },
            {
                "split": "test",
                "qid": "q2",
                "action": "left",
                "reward": 0.0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "ndcg_at_5": 0.0,
                "rewrite_cost": 0.0,
                "retrieval_calls": 1,
                "state_a": 0.9,
                "state_b": 0.8,
            },
        ]
    )

    evals, actions = train_evals_from_detailed_frame(frame, split="train")

    assert actions == ["left", "right"]
    assert len(evals) == 1
    assert np.allclose(evals[0]["features"], [0.1, 0.2])
    assert evals[0]["actions"]["left"]["reward"] == 1.0
    assert evals[0]["actions"]["right"]["reward"] == 0.0


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
