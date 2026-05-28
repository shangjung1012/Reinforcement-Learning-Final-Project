from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.off_policy_evaluation import AGGREGATE_METHODS


CONSTRAINED_SWEEP_COLUMNS = [
    "dataset",
    "call_penalty",
    "policy_utility",
    "policy_reward",
    "policy_recall_at_5",
    "policy_mrr",
    "policy_ndcg_at_5",
    "policy_rewrite_cost",
    "policy_retrieval_calls",
    "policy_primary_action",
    "policy_primary_action_share",
    "train_best_action",
    "train_best_utility",
    "train_best_retrieval_calls",
    "oracle_utility",
    "oracle_retrieval_calls",
]
CONSTRAINED_BOOTSTRAP_COLUMNS = [
    "dataset",
    "call_penalty",
    "n_queries",
    "bootstrap_samples",
    "policy_utility_mean",
    "train_best_utility_mean",
    "utility_delta_mean",
    "utility_delta_ci_low",
    "utility_delta_ci_high",
    "policy_calls_mean",
    "train_best_calls_mean",
    "call_delta_mean",
    "call_delta_ci_low",
    "call_delta_ci_high",
]


@dataclass
class PerActionLinearUtilityPolicy:
    actions: list[str]
    l2: float = 1.0

    def fit(self, features: pd.DataFrame, utilities: dict[str, list[float]]) -> None:
        x = _design_matrix(features)
        reg = self.l2 * np.eye(x.shape[1], dtype=float)
        reg[0, 0] = 0.0
        self.weights_: dict[str, np.ndarray] = {}
        for action in self.actions:
            y = np.asarray(utilities[action], dtype=float)
            try:
                self.weights_[action] = np.linalg.solve(x.T @ x + reg, x.T @ y)
            except np.linalg.LinAlgError:
                self.weights_[action] = np.linalg.pinv(x.T @ x + reg) @ x.T @ y

    def predict(self, features: pd.DataFrame) -> list[str]:
        x = _design_matrix(features)
        scores = {action: x @ weights for action, weights in self.weights_.items()}
        selected = []
        for row_idx in range(len(features)):
            selected.append(max(self.actions, key=lambda action: (scores[action][row_idx], -self.actions.index(action))))
        return selected


def export_constrained_policy_sweep(
    dataset: str,
    detailed_csv: Path,
    output_csv: Path,
    call_penalties: list[float],
    split: str = "test",
    l2: float = 1.0,
    mrr_weight: float = 0.5,
    call_baseline: float = 1.0,
) -> Path:
    if not call_penalties:
        raise ValueError("call_penalties must not be empty")
    detailed = pd.read_csv(detailed_csv)
    action_rows = detailed[~detailed["method"].isin(AGGREGATE_METHODS)].copy()
    actions = list(dict.fromkeys(action_rows["action"].astype(str).tolist()))
    feature_columns = [column for column in action_rows.columns if column.startswith("state_")]
    train_table = _action_table(action_rows[action_rows["split"] == "train"], actions, feature_columns)
    test_table = _action_table(action_rows[action_rows["split"] == split], actions, feature_columns)

    rows = []
    for penalty in call_penalties:
        train_utilities = {
            action: _utility(train_table[action], penalty, mrr_weight, call_baseline).tolist()
            for action in actions
        }
        train_best_action = max(actions, key=lambda action: (float(np.mean(train_utilities[action])), -actions.index(action)))
        policy = PerActionLinearUtilityPolicy(actions=actions, l2=l2)
        policy.fit(train_table["features"], train_utilities)
        selected_actions = policy.predict(test_table["features"])
        rows.append(
            _summary_row(
                dataset,
                float(penalty),
                test_table,
                actions,
                selected_actions,
                train_best_action,
                mrr_weight,
                call_baseline,
            )
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CONSTRAINED_SWEEP_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def export_constrained_policy_bootstrap(
    dataset: str,
    detailed_csv: Path,
    output_csv: Path,
    call_penalties: list[float],
    split: str = "test",
    l2: float = 1.0,
    mrr_weight: float = 0.5,
    call_baseline: float = 1.0,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> Path:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    detailed = pd.read_csv(detailed_csv)
    action_rows = detailed[~detailed["method"].isin(AGGREGATE_METHODS)].copy()
    actions = list(dict.fromkeys(action_rows["action"].astype(str).tolist()))
    feature_columns = [column for column in action_rows.columns if column.startswith("state_")]
    train_table = _action_table(action_rows[action_rows["split"] == "train"], actions, feature_columns)
    test_table = _action_table(action_rows[action_rows["split"] == split], actions, feature_columns)

    rng = np.random.default_rng(seed)
    rows = []
    for penalty in call_penalties:
        train_utilities = {
            action: _utility(train_table[action], penalty, mrr_weight, call_baseline).tolist()
            for action in actions
        }
        train_best_action = max(actions, key=lambda action: (float(np.mean(train_utilities[action])), -actions.index(action)))
        policy = PerActionLinearUtilityPolicy(actions=actions, l2=l2)
        policy.fit(train_table["features"], train_utilities)
        selected_actions = policy.predict(test_table["features"])
        policy_rows = _rows_for_actions(test_table, selected_actions)
        train_best_rows = test_table[train_best_action]
        policy_utility = _utility(policy_rows, penalty, mrr_weight, call_baseline).to_numpy(dtype=float)
        train_best_utility = _utility(train_best_rows, penalty, mrr_weight, call_baseline).to_numpy(dtype=float)
        policy_calls = policy_rows["retrieval_calls"].to_numpy(dtype=float)
        train_best_calls = train_best_rows["retrieval_calls"].to_numpy(dtype=float)
        utility_delta = policy_utility - train_best_utility
        call_delta = policy_calls - train_best_calls
        utility_ci = _paired_bootstrap_interval(utility_delta, rng, bootstrap_samples)
        call_ci = _paired_bootstrap_interval(call_delta, rng, bootstrap_samples)
        rows.append(
            {
                "dataset": dataset,
                "call_penalty": float(penalty),
                "n_queries": int(len(policy_rows)),
                "bootstrap_samples": int(bootstrap_samples),
                "policy_utility_mean": float(np.mean(policy_utility)),
                "train_best_utility_mean": float(np.mean(train_best_utility)),
                "utility_delta_mean": float(np.mean(utility_delta)),
                "utility_delta_ci_low": utility_ci[0],
                "utility_delta_ci_high": utility_ci[1],
                "policy_calls_mean": float(np.mean(policy_calls)),
                "train_best_calls_mean": float(np.mean(train_best_calls)),
                "call_delta_mean": float(np.mean(call_delta)),
                "call_delta_ci_low": call_ci[0],
                "call_delta_ci_high": call_ci[1],
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CONSTRAINED_BOOTSTRAP_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _action_table(rows: pd.DataFrame, actions: list[str], feature_columns: list[str]) -> dict[str, object]:
    if rows.empty:
        raise ValueError("action table split has no rows")
    qids = list(dict.fromkeys(rows["qid"].astype(str).tolist()))
    indexed = rows.assign(qid=rows["qid"].astype(str)).set_index(["qid", "action"], drop=False)
    table: dict[str, object] = {"qids": qids, "features": _feature_frame(rows, qids, feature_columns)}
    for action in actions:
        action_rows = []
        for qid in qids:
            key = (qid, action)
            if key not in indexed.index:
                raise ValueError(f"Missing action row for qid={qid!r}, action={action!r}")
            value = indexed.loc[key]
            if isinstance(value, pd.DataFrame):
                value = value.iloc[0]
            action_rows.append(value)
        table[action] = pd.DataFrame(action_rows).reset_index(drop=True)
    return table


def _feature_frame(rows: pd.DataFrame, qids: list[str], feature_columns: list[str]) -> pd.DataFrame:
    first_rows = rows.assign(qid=rows["qid"].astype(str)).drop_duplicates("qid").set_index("qid", drop=False)
    return first_rows.loc[qids, feature_columns].reset_index(drop=True)


def _utility(rows: pd.DataFrame, call_penalty: float, mrr_weight: float, call_baseline: float) -> pd.Series:
    extra_calls = np.maximum(rows["retrieval_calls"].to_numpy(dtype=float) - call_baseline, 0.0)
    return (
        rows["recall_at_5"].astype(float)
        + mrr_weight * rows["mrr"].astype(float)
        - rows["rewrite_cost"].astype(float)
        - call_penalty * extra_calls
    )


def _summary_row(
    dataset: str,
    call_penalty: float,
    table: dict[str, object],
    actions: list[str],
    selected_actions: list[str],
    train_best_action: str,
    mrr_weight: float,
    call_baseline: float,
) -> dict[str, object]:
    selected_rows = _rows_for_actions(table, selected_actions)
    utilities = {
        action: _utility(table[action], call_penalty, mrr_weight, call_baseline).to_numpy(dtype=float)
        for action in actions
    }
    oracle_actions = [
        max(actions, key=lambda action: (utilities[action][idx], -actions.index(action)))
        for idx in range(len(selected_actions))
    ]
    primary_action, primary_share = _primary_action(selected_actions)
    return {
        "dataset": dataset,
        "call_penalty": call_penalty,
        "policy_utility": float(np.mean(_utility(selected_rows, call_penalty, mrr_weight, call_baseline))),
        "policy_reward": float(selected_rows["reward"].mean()),
        "policy_recall_at_5": float(selected_rows["recall_at_5"].mean()),
        "policy_mrr": float(selected_rows["mrr"].mean()),
        "policy_ndcg_at_5": float(selected_rows["ndcg_at_5"].mean()),
        "policy_rewrite_cost": float(selected_rows["rewrite_cost"].mean()),
        "policy_retrieval_calls": float(selected_rows["retrieval_calls"].mean()),
        "policy_primary_action": primary_action,
        "policy_primary_action_share": primary_share,
        "train_best_action": train_best_action,
        "train_best_utility": float(np.mean(utilities[train_best_action])),
        "train_best_retrieval_calls": float(table[train_best_action]["retrieval_calls"].mean()),
        "oracle_utility": float(np.mean(_utility(_rows_for_actions(table, oracle_actions), call_penalty, mrr_weight, call_baseline))),
        "oracle_retrieval_calls": float(_rows_for_actions(table, oracle_actions)["retrieval_calls"].mean()),
    }


def _rows_for_actions(table: dict[str, object], actions: list[str]) -> pd.DataFrame:
    rows = [table[action].iloc[idx] for idx, action in enumerate(actions)]
    return pd.DataFrame(rows).reset_index(drop=True)


def _primary_action(actions: list[str]) -> tuple[str, float]:
    counts = pd.Series(actions).value_counts()
    action = str(counts.index[0])
    return action, float(counts.iloc[0] / len(actions)) if actions else 0.0


def _design_matrix(features: pd.DataFrame) -> np.ndarray:
    matrix = features.to_numpy(dtype=float) if len(features.columns) else np.empty((len(features), 0))
    return np.column_stack([np.ones(len(features), dtype=float), matrix])


def _paired_bootstrap_interval(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    estimates = []
    for _ in range(samples):
        idx = rng.integers(0, len(values), size=len(values))
        estimates.append(float(np.mean(values[idx])))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)
