from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METRICS = ["reward", "recall_at_5", "mrr", "ndcg_at_5"]


def export_paired_bootstrap_diagnostics(
    detailed_csv: Path,
    output_csv: Path,
    dataset: str,
    policy_method: str = "Selective retrieval policy",
    baseline_method: str = "Train-best retrieval action",
    metrics: list[str] | None = None,
    n_bootstrap: int = 5000,
    seed: int = 42,
    ci: float = 0.95,
) -> Path:
    detailed = pd.read_csv(detailed_csv)
    test_rows = detailed[detailed["split"] == "test"]
    selected_metrics = metrics or DEFAULT_METRICS
    missing = [metric for metric in selected_metrics if metric not in test_rows.columns]
    if missing:
        raise ValueError(f"Missing metric column(s): {', '.join(missing)}")

    paired = _paired_methods(test_rows, policy_method, baseline_method, selected_metrics)
    rows = [
        _bootstrap_metric_row(
            dataset=dataset,
            comparison=f"{policy_method} - {baseline_method}",
            metric=metric,
            policy_values=paired[f"{metric}__policy"].to_numpy(dtype=float),
            baseline_values=paired[f"{metric}__baseline"].to_numpy(dtype=float),
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci=ci,
        )
        for metric in selected_metrics
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def _paired_methods(
    test_rows: pd.DataFrame,
    policy_method: str,
    baseline_method: str,
    metrics: list[str],
) -> pd.DataFrame:
    keep_columns = ["qid", "method", *metrics]
    pivot = test_rows[keep_columns].pivot_table(index="qid", columns="method", values=metrics, aggfunc="first")
    required_methods = {policy_method, baseline_method}
    available_methods = set(pivot.columns.get_level_values("method"))
    missing_methods = sorted(required_methods - available_methods)
    if missing_methods:
        raise ValueError(f"Missing method row(s): {', '.join(missing_methods)}")

    data = {}
    for metric in metrics:
        data[f"{metric}__policy"] = pivot[(metric, policy_method)]
        data[f"{metric}__baseline"] = pivot[(metric, baseline_method)]
    paired = pd.DataFrame(data).dropna()
    if paired.empty:
        raise ValueError("No paired query rows found for bootstrap diagnostics")
    return paired


def _bootstrap_metric_row(
    dataset: str,
    comparison: str,
    metric: str,
    policy_values: np.ndarray,
    baseline_values: np.ndarray,
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, object]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be between 0 and 1")
    deltas = policy_values - baseline_values
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(deltas), size=(n_bootstrap, len(deltas)))
    means = deltas[sample_idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return {
        "dataset": dataset,
        "comparison": comparison,
        "metric": metric,
        "n_queries": int(len(deltas)),
        "mean_policy": float(np.mean(policy_values)),
        "mean_baseline": float(np.mean(baseline_values)),
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": float(np.quantile(means, alpha)),
        "ci_upper": float(np.quantile(means, 1.0 - alpha)),
        "prob_delta_gt_0": float(np.mean(means > 0.0)),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }
