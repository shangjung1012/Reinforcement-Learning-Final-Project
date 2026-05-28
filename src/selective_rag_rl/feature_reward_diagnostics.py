from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.feature_group_diagnostics import feature_group_slices

TARGET_COLUMNS = [
    "oracle_reward",
    "oracle_margin",
    "dense_advantage_vs_bm25",
    "hybrid_advantage_vs_bm25",
]


def reward_targets_frame(evals: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for action_eval in evals:
        actions = action_eval["actions"]
        rewards = {action: float(row["reward"]) for action, row in actions.items()}
        sorted_rewards = sorted(rewards.items(), key=lambda item: (-item[1], item[0]))
        oracle_action, oracle_reward = sorted_rewards[0]
        second_best = sorted_rewards[1][1] if len(sorted_rewards) > 1 else oracle_reward
        best_bm25 = _best_reward(rewards, ["bm25_keep", "bm25_keyword"])
        best_dense = _best_reward(rewards, ["dense_keep", "dense_keyword"])
        best_hybrid = _best_reward(rewards, ["hybrid_keep", "hybrid_keyword"])
        rows.append(
            {
                "oracle_action": oracle_action,
                "oracle_reward": oracle_reward,
                "oracle_margin": round(oracle_reward - second_best, 12),
                "dense_advantage_vs_bm25": round(best_dense - best_bm25, 12),
                "hybrid_advantage_vs_bm25": round(best_hybrid - best_bm25, 12),
            }
        )
    return pd.DataFrame(rows)


def export_feature_reward_diagnostics(
    features: np.ndarray,
    evals: list[dict[str, object]],
    output_csv: Path,
    dataset: str,
    split: str,
) -> Path:
    diagnostics = feature_reward_diagnostics_frame(features, evals, dataset=dataset, split=split)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_csv, index=False)
    return output_csv


def feature_reward_diagnostics_frame(
    features: np.ndarray,
    evals: list[dict[str, object]],
    dataset: str,
    split: str,
) -> pd.DataFrame:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    targets = reward_targets_frame(evals)
    if len(matrix) != len(targets):
        raise ValueError("Feature rows and reward targets must have the same length")

    rows = []
    for group, (start, end) in feature_group_slices(matrix.shape[1]).items():
        values = matrix[:, start:end]
        active_mask = np.any(np.abs(values) > 1e-12, axis=0) if values.size else np.asarray([], dtype=bool)
        for target in TARGET_COLUMNS:
            correlations = [_safe_corr(values[:, idx], targets[target].to_numpy(dtype=float)) for idx in range(values.shape[1])]
            abs_corr = np.abs(np.asarray(correlations, dtype=float)) if correlations else np.asarray([], dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "group": group,
                    "target": target,
                    "start": start,
                    "end": end,
                    "width": end - start,
                    "active_columns": int(np.sum(active_mask)),
                    "mean_abs_corr": float(np.mean(abs_corr)) if abs_corr.size else 0.0,
                    "max_abs_corr": float(np.max(abs_corr)) if abs_corr.size else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _best_reward(rewards: dict[str, float], actions: list[str]) -> float:
    values = [rewards[action] for action in actions if action in rewards]
    return max(values) if values else 0.0


def _safe_corr(feature: np.ndarray, target: np.ndarray) -> float:
    if feature.size < 2 or target.size < 2:
        return 0.0
    if float(np.std(feature)) <= 1e-12 or float(np.std(target)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(feature, target)[0, 1])
