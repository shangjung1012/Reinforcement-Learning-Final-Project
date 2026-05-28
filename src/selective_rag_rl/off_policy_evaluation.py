from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


AGGREGATE_METHODS = {
    "Train-best retrieval action",
    "Heuristic retrieval router",
    "Selective retrieval policy",
    "Confidence-gated retrieval policy",
    "Oracle retrieval action",
}
DEFAULT_TARGET_METHODS = [
    "Train-best retrieval action",
    "Heuristic retrieval router",
    "Selective retrieval policy",
    "Confidence-gated retrieval policy",
]
OPE_COLUMNS = [
    "dataset",
    "split",
    "logging_seed",
    "behavior_policy",
    "target_method",
    "estimator",
    "estimated_value",
    "true_value",
    "absolute_error",
    "n_queries",
    "match_rate",
    "effective_sample_size",
    "logged_reward",
    "mean_propensity",
]
OPE_STABILITY_COLUMNS = [
    "dataset",
    "split",
    "behavior_policy",
    "target_method",
    "estimator",
    "seed_count",
    "true_value",
    "mean_estimated_value",
    "std_estimated_value",
    "mean_absolute_error",
    "std_absolute_error",
    "ci95_absolute_error_low",
    "ci95_absolute_error_high",
    "mean_match_rate",
    "mean_effective_sample_size",
]


@dataclass(frozen=True)
class BehaviorPolicySpec:
    name: str
    kind: str
    epsilon: float = 0.2
    reference_method: str | None = None


@dataclass
class PerActionRidgeRewardModel:
    actions: list[str]
    feature_columns: list[str]
    l2: float = 1.0

    def fit(self, rows: pd.DataFrame) -> None:
        self.global_mean_ = float(rows["reward"].mean()) if len(rows) else 0.0
        self.weights_: dict[str, np.ndarray] = {}
        self.means_: dict[str, float] = {}
        for action in self.actions:
            action_rows = rows[rows["action"] == action]
            self.means_[action] = float(action_rows["reward"].mean()) if len(action_rows) else self.global_mean_
            if len(action_rows) < 2 or not self.feature_columns:
                continue
            x = _design_matrix(action_rows, self.feature_columns)
            y = action_rows["reward"].to_numpy(dtype=float)
            reg = self.l2 * np.eye(x.shape[1], dtype=float)
            reg[0, 0] = 0.0
            try:
                self.weights_[action] = np.linalg.solve(x.T @ x + reg, x.T @ y)
            except np.linalg.LinAlgError:
                self.weights_[action] = np.linalg.pinv(x.T @ x + reg) @ x.T @ y

    def predict(self, rows: pd.DataFrame, actions: list[str]) -> np.ndarray:
        values = []
        for (_, row), action in zip(rows.iterrows(), actions):
            if action in self.weights_:
                x = np.asarray([1.0, *[float(row[col]) for col in self.feature_columns]], dtype=float)
                values.append(float(x @ self.weights_[action]))
            else:
                values.append(float(self.means_.get(action, self.global_mean_)))
        return np.asarray(values, dtype=float)


def behavior_probabilities(
    actions: list[str],
    spec: BehaviorPolicySpec,
    reference_action: str | None = None,
) -> dict[str, float]:
    if not actions:
        raise ValueError("actions must not be empty")
    if spec.kind == "uniform":
        probability = 1.0 / len(actions)
        return {action: probability for action in actions}
    if spec.kind != "reference_epsilon":
        raise ValueError(f"Unknown behavior policy kind: {spec.kind}")
    if reference_action not in actions:
        raise ValueError("reference_action must be one of actions for reference_epsilon behavior")
    epsilon = float(spec.epsilon)
    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    explore = epsilon / len(actions)
    return {action: (1.0 - epsilon + explore if action == reference_action else explore) for action in actions}


def estimate_off_policy_value(
    logged: pd.DataFrame,
    target_actions: list[str],
    q_target: np.ndarray,
    q_logged: np.ndarray,
) -> dict[str, float]:
    if len(logged) != len(target_actions) or len(logged) != len(q_target) or len(logged) != len(q_logged):
        raise ValueError("logged, target_actions, q_target, and q_logged must have the same length")
    if (logged["propensity"] <= 0).any():
        raise ValueError("propensity values must be positive")

    logged_actions = logged["logged_action"].astype(str).to_numpy()
    target = np.asarray(target_actions, dtype=str)
    rewards = logged["logged_reward"].to_numpy(dtype=float)
    propensities = logged["propensity"].to_numpy(dtype=float)
    matches = logged_actions == target
    weights = np.where(matches, 1.0 / propensities, 0.0)
    weighted_rewards = weights * rewards
    weight_sum = float(np.sum(weights))
    ips = float(np.mean(weighted_rewards)) if len(logged) else np.nan
    snips = float(np.sum(weighted_rewards) / weight_sum) if weight_sum > 0.0 else np.nan
    doubly_robust = float(np.mean(q_target + weights * (rewards - q_logged))) if len(logged) else np.nan
    effective_sample_size = float(weight_sum**2 / np.sum(weights**2)) if np.sum(weights**2) > 0.0 else 0.0
    return {
        "direct_method": float(np.mean(q_target)) if len(q_target) else np.nan,
        "ips": ips,
        "snips": snips,
        "doubly_robust": doubly_robust,
        "match_rate": float(np.mean(matches)) if len(matches) else 0.0,
        "effective_sample_size": effective_sample_size,
        "logged_reward": float(np.mean(rewards)) if len(rewards) else np.nan,
        "mean_propensity": float(np.mean(propensities)) if len(propensities) else np.nan,
    }


def export_ope_diagnostics(
    dataset: str,
    detailed_csv: Path,
    output_csv: Path,
    split: str = "test",
    target_methods: list[str] | None = None,
    seed: int = 42,
    behavior_policies: list[BehaviorPolicySpec] | None = None,
    l2: float = 1.0,
) -> Path:
    rows = ope_diagnostics_frame(
        dataset=dataset,
        detailed_csv=detailed_csv,
        split=split,
        target_methods=target_methods,
        seed=seed,
        behavior_policies=behavior_policies,
        l2=l2,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    return output_csv


def export_ope_stability_diagnostics(
    dataset: str,
    detailed_csv: Path,
    output_csv: Path,
    seeds: list[int],
    split: str = "test",
    target_methods: list[str] | None = None,
    behavior_policies: list[BehaviorPolicySpec] | None = None,
    l2: float = 1.0,
) -> Path:
    if not seeds:
        raise ValueError("seeds must not be empty")
    frames = [
        ope_diagnostics_frame(
            dataset=dataset,
            detailed_csv=detailed_csv,
            split=split,
            target_methods=target_methods,
            seed=seed,
            behavior_policies=behavior_policies,
            l2=l2,
        )
        for seed in seeds
    ]
    diagnostics = pd.concat(frames, ignore_index=True)
    grouped = diagnostics.groupby(["dataset", "split", "behavior_policy", "target_method", "estimator"], sort=False)
    rows = []
    for keys, group in grouped:
        errors = group["absolute_error"].dropna().to_numpy(dtype=float)
        estimates = group["estimated_value"].dropna().to_numpy(dtype=float)
        true_values = group["true_value"].dropna().to_numpy(dtype=float)
        error_mean = float(np.mean(errors)) if len(errors) else np.nan
        error_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
        error_ci = 1.96 * error_std / np.sqrt(len(errors)) if len(errors) else np.nan
        estimate_mean = float(np.mean(estimates)) if len(estimates) else np.nan
        estimate_std = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
        rows.append(
            {
                "dataset": keys[0],
                "split": keys[1],
                "behavior_policy": keys[2],
                "target_method": keys[3],
                "estimator": keys[4],
                "seed_count": int(group["logging_seed"].nunique()),
                "true_value": float(np.mean(true_values)) if len(true_values) else np.nan,
                "mean_estimated_value": estimate_mean,
                "std_estimated_value": estimate_std,
                "mean_absolute_error": error_mean,
                "std_absolute_error": error_std,
                "ci95_absolute_error_low": error_mean - error_ci if not np.isnan(error_mean) else np.nan,
                "ci95_absolute_error_high": error_mean + error_ci if not np.isnan(error_mean) else np.nan,
                "mean_match_rate": float(group["match_rate"].mean()),
                "mean_effective_sample_size": float(group["effective_sample_size"].mean()),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OPE_STABILITY_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def ope_diagnostics_frame(
    dataset: str,
    detailed_csv: Path,
    split: str = "test",
    target_methods: list[str] | None = None,
    seed: int = 42,
    behavior_policies: list[BehaviorPolicySpec] | None = None,
    l2: float = 1.0,
) -> pd.DataFrame:
    detailed = pd.read_csv(detailed_csv)
    action_rows = _action_rows(detailed)
    actions = _actions_in_order(action_rows)
    feature_columns = _state_feature_columns(detailed)
    train_rows = action_rows[action_rows["split"] == "train"]
    model = PerActionRidgeRewardModel(actions=actions, feature_columns=feature_columns, l2=l2)
    model.fit(train_rows if len(train_rows) else action_rows)

    target_methods = target_methods or [method for method in DEFAULT_TARGET_METHODS if method in set(detailed["method"])]
    behavior_policies = behavior_policies or [
        BehaviorPolicySpec(name="uniform", kind="uniform"),
        BehaviorPolicySpec(
            name="train_best_eps_0.2",
            kind="reference_epsilon",
            epsilon=0.2,
            reference_method="Train-best retrieval action",
        ),
        BehaviorPolicySpec(
            name="heuristic_eps_0.2",
            kind="reference_epsilon",
            epsilon=0.2,
            reference_method="Heuristic retrieval router",
        ),
    ]
    split_actions = action_rows[action_rows["split"] == split]
    rows: list[dict[str, object]] = []
    for behavior_index, behavior in enumerate(behavior_policies):
        logged = _simulate_logged_feedback(
            detailed=detailed,
            action_rows=split_actions,
            actions=actions,
            behavior=behavior,
            split=split,
            seed=seed + behavior_index,
        )
        q_logged = model.predict(logged, logged["logged_action"].astype(str).tolist())
        for target_method in target_methods:
            target = _target_rows(detailed, split, target_method)
            target = _align_by_qid(target, logged["qid"].astype(str).tolist())
            target_actions = target["action"].astype(str).tolist()
            true_value = float(target["reward"].mean()) if len(target) else np.nan
            q_target = model.predict(target, target_actions)
            estimates = estimate_off_policy_value(logged, target_actions, q_target, q_logged)
            for estimator in ["direct_method", "ips", "snips", "doubly_robust"]:
                estimated = estimates[estimator]
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "logging_seed": seed,
                        "behavior_policy": behavior.name,
                        "target_method": target_method,
                        "estimator": estimator,
                        "estimated_value": estimated,
                        "true_value": true_value,
                        "absolute_error": abs(estimated - true_value)
                        if not np.isnan(estimated) and not np.isnan(true_value)
                        else np.nan,
                        "n_queries": int(len(target)),
                        "match_rate": estimates["match_rate"],
                        "effective_sample_size": estimates["effective_sample_size"],
                        "logged_reward": estimates["logged_reward"],
                        "mean_propensity": estimates["mean_propensity"],
                    }
                )
    return pd.DataFrame(rows, columns=OPE_COLUMNS)


def _simulate_logged_feedback(
    detailed: pd.DataFrame,
    action_rows: pd.DataFrame,
    actions: list[str],
    behavior: BehaviorPolicySpec,
    split: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    action_by_qid = {str(qid): group for qid, group in action_rows.groupby("qid", sort=False)}
    qids = list(action_by_qid)
    reference = None
    if behavior.reference_method is not None:
        reference = _target_rows(detailed, split, behavior.reference_method)
        reference = _align_by_qid(reference, qids).set_index("qid")
    rows = []
    for qid in qids:
        reference_action = str(reference.loc[qid, "action"]) if reference is not None else None
        probabilities = behavior_probabilities(actions, behavior, reference_action=reference_action)
        action = str(rng.choice(actions, p=[probabilities[item] for item in actions]))
        selected = action_by_qid[qid][action_by_qid[qid]["action"] == action]
        if selected.empty:
            raise ValueError(f"No reward row for qid={qid!r}, action={action!r}")
        row = selected.iloc[0].to_dict()
        row["logged_action"] = action
        row["logged_reward"] = float(row["reward"])
        row["propensity"] = float(probabilities[action])
        rows.append(row)
    return pd.DataFrame(rows)


def _action_rows(detailed: pd.DataFrame) -> pd.DataFrame:
    rows = detailed[~detailed["method"].isin(AGGREGATE_METHODS)].copy()
    required = {"split", "method", "action", "qid", "reward"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"detailed CSV is missing required column(s): {', '.join(sorted(missing))}")
    return rows


def _actions_in_order(rows: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(rows["action"].astype(str).tolist()))


def _state_feature_columns(detailed: pd.DataFrame) -> list[str]:
    return [column for column in detailed.columns if column.startswith("state_")]


def _target_rows(detailed: pd.DataFrame, split: str, target_method: str) -> pd.DataFrame:
    rows = detailed[(detailed["split"] == split) & (detailed["method"] == target_method)].copy()
    if rows.empty:
        raise ValueError(f"No rows found for split={split!r}, target_method={target_method!r}")
    return rows


def _align_by_qid(rows: pd.DataFrame, qids: list[str]) -> pd.DataFrame:
    indexed = rows.assign(qid=rows["qid"].astype(str)).set_index("qid", drop=False)
    missing = [qid for qid in qids if qid not in indexed.index]
    if missing:
        raise ValueError(f"Missing target rows for qid(s): {', '.join(missing[:5])}")
    return indexed.loc[qids].reset_index(drop=True)


def _design_matrix(rows: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    features = rows[feature_columns].to_numpy(dtype=float) if feature_columns else np.empty((len(rows), 0))
    return np.column_stack([np.ones(len(rows), dtype=float), features])
