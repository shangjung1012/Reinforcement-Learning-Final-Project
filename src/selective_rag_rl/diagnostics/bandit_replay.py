from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.policies.bandit import DirectMethodBandit
from selective_rag_rl.policies.bandit_baselines import (
    EpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
)


REPLAY_HISTORY_COLUMNS = [
    "dataset",
    "policy",
    "step",
    "chosen_action",
    "oracle_action",
    "reward",
    "oracle_reward",
    "regret",
    "cumulative_reward",
    "cumulative_oracle_reward",
    "cumulative_regret",
    "moving_average_reward",
    "oracle_match_rate",
    "action_entropy",
]

REPLAY_SUMMARY_COLUMNS = [
    "dataset",
    "policy",
    "steps",
    "mean_reward",
    "mean_oracle_reward",
    "mean_regret",
    "final_cumulative_reward",
    "final_cumulative_oracle_reward",
    "final_cumulative_regret",
    "oracle_match_rate",
    "action_entropy",
    "unique_actions",
]

ACTION_METRIC_COLUMNS = {
    "reward",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "rewrite_cost",
    "retrieval_calls",
}


def export_bandit_replay_diagnostics_from_evals(
    *,
    train_evals: list[dict[str, object]],
    actions: list[str],
    history_csv: Path,
    summary_csv: Path,
    dataset: str,
    alpha: float = 1.0,
    l2: float = 1.0,
    epsilon: float = 0.1,
    posterior_scale: float = 1.0,
    seed: int = 42,
    moving_average_window: int = 20,
) -> dict[str, Path]:
    history, summary = build_bandit_replay_diagnostics(
        train_evals=train_evals,
        actions=actions,
        dataset=dataset,
        alpha=alpha,
        l2=l2,
        epsilon=epsilon,
        posterior_scale=posterior_scale,
        seed=seed,
        moving_average_window=moving_average_window,
    )
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    return {"history_csv": history_csv, "summary_csv": summary_csv}


def export_bandit_replay_diagnostics_from_detailed_csv(
    *,
    detailed_csv: Path,
    history_csv: Path,
    summary_csv: Path,
    dataset: str,
    split: str = "train",
    alpha: float = 1.0,
    l2: float = 1.0,
    epsilon: float = 0.1,
    posterior_scale: float = 1.0,
    seed: int = 42,
    moving_average_window: int = 20,
) -> dict[str, Path]:
    frame = pd.read_csv(detailed_csv)
    train_evals, actions = train_evals_from_detailed_frame(frame, split=split)
    return export_bandit_replay_diagnostics_from_evals(
        train_evals=train_evals,
        actions=actions,
        history_csv=history_csv,
        summary_csv=summary_csv,
        dataset=dataset,
        alpha=alpha,
        l2=l2,
        epsilon=epsilon,
        posterior_scale=posterior_scale,
        seed=seed,
        moving_average_window=moving_average_window,
    )


def train_evals_from_detailed_frame(
    frame: pd.DataFrame,
    *,
    split: str = "train",
    feature_columns: list[str] | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    required = {"split", "qid", "action", *ACTION_METRIC_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"detailed frame missing required column(s): {', '.join(missing)}")
    subset = frame[frame["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"detailed frame has no rows for split={split!r}")
    features = feature_columns or [column for column in subset.columns if column.startswith("state_")]
    if not features:
        raise ValueError("detailed frame contains no state_* feature columns")
    actions = list(dict.fromkeys(str(action) for action in subset["action"]))
    evals: list[dict[str, object]] = []
    for _qid, group in subset.groupby("qid", sort=False):
        action_rows = {str(row["action"]): row for _, row in group.iterrows()}
        missing_actions = [action for action in actions if action not in action_rows]
        if missing_actions:
            raise ValueError(f"qid has missing action row(s): {', '.join(missing_actions)}")
        first = group.iloc[0]
        evals.append(
            {
                "features": first[features].to_numpy(dtype=float),
                "actions": {
                    action: {
                        metric: float(action_rows[action][metric])
                        for metric in ACTION_METRIC_COLUMNS
                    }
                    for action in actions
                },
            }
        )
    return evals, actions


def write_bandit_replay_figure(history: pd.DataFrame, *, output_png: Path) -> Path:
    import matplotlib.pyplot as plt

    if "policy" not in history or "step" not in history or "cumulative_regret" not in history:
        raise ValueError("history must contain policy, step, and cumulative_regret columns")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, group in history.groupby("policy", sort=False):
        ordered = group.sort_values("step")
        ax.plot(ordered["step"], ordered["cumulative_regret"], label=policy, linewidth=1.8)
    ax.set_title("Selected-action bandit replay regret")
    ax.set_xlabel("Replay step")
    ax.set_ylabel("Cumulative regret vs oracle action")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return output_png


def build_bandit_replay_diagnostics(
    *,
    train_evals: list[dict[str, object]],
    actions: list[str],
    dataset: str,
    alpha: float = 1.0,
    l2: float = 1.0,
    epsilon: float = 0.1,
    posterior_scale: float = 1.0,
    seed: int = 42,
    moving_average_window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not train_evals:
        raise ValueError("train_evals must not be empty")
    if not actions:
        raise ValueError("actions must not be empty")
    features = np.vstack([np.asarray(row["features"], dtype=float) for row in train_evals])
    rewards = {
        action: [float(row["actions"][action]["reward"]) for row in train_evals]
        for action in actions
    }
    oracle_actions = [_oracle_action(row, actions) for row in train_evals]
    histories = [
        _direct_method_history(features, rewards, actions, l2=l2),
        _selected_policy_history(
            "LinUCB selected-action replay",
            LinUCBPolicy(actions=actions, alpha=alpha, l2=l2).fit(features, rewards),
            oracle_actions,
        ),
        _selected_policy_history(
            "Epsilon-greedy selected-action replay",
            EpsilonGreedyPolicy(actions=actions, epsilon=epsilon, l2=l2, seed=seed).fit(features, rewards),
            oracle_actions,
        ),
        _selected_policy_history(
            "Linear Thompson selected-action replay",
            LinearThompsonSamplingPolicy(
                actions=actions,
                posterior_scale=posterior_scale,
                l2=l2,
                seed=seed,
            ).fit(features, rewards),
            oracle_actions,
        ),
        _fixed_action_history("Train-best fixed action", _best_fixed_action(rewards, actions), rewards, oracle_actions),
        _oracle_history(rewards, oracle_actions),
    ]
    history = pd.concat(histories, ignore_index=True)
    history = _add_running_diagnostics(history, dataset=dataset, moving_average_window=moving_average_window)
    summary = _summarize(history)
    return history[REPLAY_HISTORY_COLUMNS], summary[REPLAY_SUMMARY_COLUMNS]


def _direct_method_history(
    features: np.ndarray,
    rewards: dict[str, list[float]],
    actions: list[str],
    *,
    l2: float,
) -> pd.DataFrame:
    policy = DirectMethodBandit(actions=actions, l2=l2)
    policy.fit(features, rewards)
    rows = []
    for step, feature in enumerate(features):
        chosen_action = policy.predict(feature)
        oracle_action = max(actions, key=lambda action: (float(rewards[action][step]), -actions.index(action)))
        reward = float(rewards[chosen_action][step])
        oracle_reward = float(rewards[oracle_action][step])
        rows.append(_history_row("Full-information direct method", step, chosen_action, oracle_action, reward, oracle_reward))
    return pd.DataFrame(rows)


def _selected_policy_history(policy_name: str, raw_history: pd.DataFrame, oracle_actions: list[str]) -> pd.DataFrame:
    rows = []
    for row, oracle_action in zip(raw_history.itertuples(index=False), oracle_actions, strict=True):
        rows.append(
            _history_row(
                policy_name,
                int(row.step),
                str(row.chosen_action),
                oracle_action,
                float(row.reward),
                float(row.oracle_reward),
            )
        )
    return pd.DataFrame(rows)


def _fixed_action_history(
    policy_name: str,
    action: str,
    rewards: dict[str, list[float]],
    oracle_actions: list[str],
) -> pd.DataFrame:
    rows = []
    actions = list(rewards)
    for step, oracle_action in enumerate(oracle_actions):
        reward = float(rewards[action][step])
        oracle_reward = max(float(rewards[item][step]) for item in actions)
        rows.append(_history_row(policy_name, step, action, oracle_action, reward, oracle_reward))
    return pd.DataFrame(rows)


def _oracle_history(rewards: dict[str, list[float]], oracle_actions: list[str]) -> pd.DataFrame:
    rows = []
    for step, oracle_action in enumerate(oracle_actions):
        reward = float(rewards[oracle_action][step])
        rows.append(_history_row("Oracle action", step, oracle_action, oracle_action, reward, reward))
    return pd.DataFrame(rows)


def _history_row(
    policy: str,
    step: int,
    chosen_action: str,
    oracle_action: str,
    reward: float,
    oracle_reward: float,
) -> dict[str, object]:
    return {
        "policy": policy,
        "step": int(step),
        "chosen_action": chosen_action,
        "oracle_action": oracle_action,
        "reward": float(reward),
        "oracle_reward": float(oracle_reward),
        "regret": float(oracle_reward - reward),
    }


def _add_running_diagnostics(history: pd.DataFrame, *, dataset: str, moving_average_window: int) -> pd.DataFrame:
    frames = []
    window = max(int(moving_average_window), 1)
    for policy, group in history.groupby("policy", sort=False):
        ordered = group.sort_values("step").copy()
        ordered["dataset"] = dataset
        ordered["cumulative_reward"] = ordered["reward"].cumsum()
        ordered["cumulative_oracle_reward"] = ordered["oracle_reward"].cumsum()
        ordered["cumulative_regret"] = ordered["regret"].cumsum()
        ordered["moving_average_reward"] = ordered["reward"].rolling(window=window, min_periods=1).mean()
        ordered["oracle_match_rate"] = (ordered["chosen_action"] == ordered["oracle_action"]).expanding().mean()
        ordered["action_entropy"] = [
            _entropy(ordered["chosen_action"].iloc[: idx + 1])
            for idx in range(len(ordered))
        ]
        frames.append(ordered)
    return pd.concat(frames, ignore_index=True)


def _summarize(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in history.groupby("policy", sort=False):
        final = group.sort_values("step").iloc[-1]
        rows.append(
            {
                "dataset": str(final["dataset"]),
                "policy": policy,
                "steps": int(len(group)),
                "mean_reward": float(group["reward"].mean()),
                "mean_oracle_reward": float(group["oracle_reward"].mean()),
                "mean_regret": float(group["regret"].mean()),
                "final_cumulative_reward": float(final["cumulative_reward"]),
                "final_cumulative_oracle_reward": float(final["cumulative_oracle_reward"]),
                "final_cumulative_regret": float(final["cumulative_regret"]),
                "oracle_match_rate": float(group["chosen_action"].eq(group["oracle_action"]).mean()),
                "action_entropy": _entropy(group["chosen_action"]),
                "unique_actions": int(group["chosen_action"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _oracle_action(row: dict[str, object], actions: list[str]) -> str:
    return max(actions, key=lambda action: (float(row["actions"][action]["reward"]), -actions.index(action)))


def _best_fixed_action(rewards: dict[str, list[float]], actions: list[str]) -> str:
    return max(actions, key=lambda action: (float(np.mean(rewards[action])), -actions.index(action)))


def _entropy(values: pd.Series) -> float:
    counts = values.value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())
