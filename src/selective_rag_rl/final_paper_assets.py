from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = ["scifact", "nfcorpus"]
BANDIT_BASELINE_METHODS = [
    "LinUCB retrieval policy",
    "Epsilon-greedy retrieval policy",
    "Linear Thompson retrieval policy",
]
PAPER_TABLE_COLUMNS = [
    "dataset",
    "method_group",
    "method",
    "objective",
    "objective_value",
    "delta_vs_train_best",
    "ci_low",
    "ci_high",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "retrieval_calls",
    "evidence_artifact_id",
]


def export_final_paper_assets(
    *,
    root: Path,
    results_dir: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = build_main_results_table(root)
    main_results_csv = results_dir / "final_main_results_table.csv"
    main_results_tex = results_dir / "final_main_results_table.tex"
    table.to_csv(main_results_csv, index=False)
    _write_latex_table(table, main_results_tex)

    outputs = {
        "main_results_csv": main_results_csv,
        "main_results_tex": main_results_tex,
        "reward_delta_ci_png": figures_dir / "final_reward_delta_ci.png",
        "cost_reward_frontier_png": figures_dir / "final_cost_reward_frontier.png",
        "ope_estimator_error_png": figures_dir / "final_ope_estimator_error.png",
        "linucb_comparison_png": figures_dir / "final_linucb_comparison.png",
    }
    plot_reward_delta_ci(table, outputs["reward_delta_ci_png"])
    plot_cost_reward_frontier(root, outputs["cost_reward_frontier_png"])
    plot_ope_estimator_error(root, outputs["ope_estimator_error_png"])
    plot_linucb_comparison(root, outputs["linucb_comparison_png"])
    return outputs


def build_main_results_table(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        summary = pd.read_csv(root / "outputs" / "results" / f"{dataset}_retrieval_policy_summary.csv")
        linucb = pd.read_csv(root / "outputs" / "results" / f"{dataset}_linucb_baseline_summary.csv")
        bootstrap = pd.read_csv(root / "outputs" / "results" / f"{dataset}_bootstrap_diagnostics.csv")
        constrained_sweep = pd.read_csv(root / "outputs" / "results" / f"{dataset}_constrained_policy_sweep.csv")
        constrained_bootstrap = pd.read_csv(
            root / "outputs" / "results" / f"{dataset}_constrained_policy_bootstrap.csv"
        )
        train_best = _method(summary, "Train-best retrieval action")
        reward_ci = _metric_row(bootstrap, "reward")

        rows.extend(
            [
                _summary_row(
                    dataset=dataset,
                    method_group="fixed",
                    method="Train-best retrieval action",
                    row=train_best,
                    train_best_reward=train_best["reward"],
                    evidence_artifact_id=f"{dataset}_policy_summary",
                ),
                _summary_row(
                    dataset=dataset,
                    method_group="heuristic",
                    method="Heuristic retrieval router",
                    row=_method(summary, "Heuristic retrieval router"),
                    train_best_reward=train_best["reward"],
                    evidence_artifact_id=f"{dataset}_policy_summary",
                ),
                _best_bandit_row(dataset=dataset, linucb=linucb, train_best_reward=train_best["reward"]),
                _summary_row(
                    dataset=dataset,
                    method_group="learned",
                    method="Selective retrieval policy",
                    row=_method(summary, "Selective retrieval policy"),
                    train_best_reward=train_best["reward"],
                    ci_low=reward_ci["ci_lower"],
                    ci_high=reward_ci["ci_upper"],
                    evidence_artifact_id=f"{dataset}_bootstrap",
                ),
                _constrained_row(
                    dataset=dataset,
                    sweep=constrained_sweep,
                    bootstrap=constrained_bootstrap,
                    call_penalty=0.03,
                ),
                _summary_row(
                    dataset=dataset,
                    method_group="oracle",
                    method="Oracle retrieval action",
                    row=_method(summary, "Oracle retrieval action"),
                    train_best_reward=train_best["reward"],
                    evidence_artifact_id=f"{dataset}_policy_summary",
                ),
            ]
        )
    return pd.DataFrame(rows, columns=PAPER_TABLE_COLUMNS)


def plot_reward_delta_ci(table: pd.DataFrame, output_png: Path) -> None:
    policy = table[table["method"] == "Selective retrieval policy"].copy()
    x = range(len(policy))
    y = policy["delta_vs_train_best"].astype(float)
    low = y - policy["ci_low"].astype(float)
    high = policy["ci_high"].astype(float) - y

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.errorbar(x, y, yerr=[low, high], fmt="o", color="#1f77b4", capsize=5)
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_xticks(list(x), [str(dataset).upper() for dataset in policy["dataset"]])
    ax.set_ylabel("Reward delta vs train-best")
    ax.set_title("Selective policy reward gain with bootstrap CI")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def plot_cost_reward_frontier(root: Path, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for dataset in DATASETS:
        sweep = pd.read_csv(root / "outputs" / "results" / f"{dataset}_constrained_policy_sweep.csv")
        ax.plot(
            sweep["policy_retrieval_calls"],
            sweep["policy_utility"],
            marker="o",
            linewidth=1.8,
            label=dataset,
        )
        for _, row in sweep.iterrows():
            ax.annotate(f"{row['call_penalty']:.2g}", (row["policy_retrieval_calls"], row["policy_utility"]), fontsize=7)
    ax.set_xlabel("Policy retrieval calls")
    ax.set_ylabel("Lagrangian utility")
    ax.set_title("Constrained policy reward/call frontier")
    ax.grid(alpha=0.25)
    ax.legend(title="Dataset")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def plot_ope_estimator_error(root: Path, output_png: Path) -> None:
    frames = []
    for dataset in DATASETS:
        stability = pd.read_csv(root / "outputs" / "results" / f"{dataset}_ope_stability.csv")
        selected = stability[
            (stability["behavior_policy"] == "uniform")
            & (stability["target_method"] == "Selective retrieval policy")
        ].copy()
        selected["dataset"] = dataset
        frames.append(selected)
    frame = pd.concat(frames, ignore_index=True)

    estimators = ["direct_method", "ips", "snips", "doubly_robust"]
    width = 0.35
    x = list(range(len(estimators)))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for offset, dataset in [(-width / 2, "scifact"), (width / 2, "nfcorpus")]:
        data = frame[frame["dataset"] == dataset].set_index("estimator").loc[estimators]
        ax.bar(
            [pos + offset for pos in x],
            data["mean_absolute_error"].astype(float),
            width=width,
            label=dataset,
        )
    ax.set_xticks(x, ["DM", "IPS", "SNIPS", "DR"])
    ax.set_ylabel("Mean absolute OPE error")
    ax.set_title("Uniform-log OPE estimator error")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Dataset")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def plot_linucb_comparison(root: Path, output_png: Path) -> None:
    rows = []
    for dataset in DATASETS:
        summary = pd.read_csv(root / "outputs" / "results" / f"{dataset}_retrieval_policy_summary.csv")
        linucb = pd.read_csv(root / "outputs" / "results" / f"{dataset}_linucb_baseline_summary.csv")
        rows.extend(
            [
                {"dataset": dataset, "method": "Train-best", "reward": _method(summary, "Train-best retrieval action")["reward"]},
                {
                    "dataset": dataset,
                    "method": "Best selected-action",
                    "reward": _best_bandit_method(linucb)["reward"],
                },
                {"dataset": dataset, "method": "Selective", "reward": _method(summary, "Selective retrieval policy")["reward"]},
            ]
        )
    frame = pd.DataFrame(rows)
    methods = ["Train-best", "Best selected-action", "Selective"]
    width = 0.25
    x = list(range(len(DATASETS)))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for idx, method in enumerate(methods):
        if method not in set(frame["method"]):
            continue
        data = frame[frame["method"] == method].set_index("dataset").loc[DATASETS]
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar([pos + offset for pos in x], data["reward"].astype(float), width=width, label=method)
    ax.set_xticks(x, [dataset.upper() for dataset in DATASETS])
    ax.set_ylabel("Reward")
    ax.set_title("Best selected-action bandit baseline versus direct method")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _summary_row(
    *,
    dataset: str,
    method_group: str,
    method: str,
    row: pd.Series,
    train_best_reward: object,
    evidence_artifact_id: str,
    ci_low: object = pd.NA,
    ci_high: object = pd.NA,
) -> dict[str, object]:
    return _paper_row(
        dataset=dataset,
        method_group=method_group,
        method=method,
        objective="reward",
        objective_value=row["reward"],
        delta_vs_train_best=float(row["reward"]) - float(train_best_reward),
        ci_low=ci_low,
        ci_high=ci_high,
        recall_at_5=row.get("recall_at_5", pd.NA),
        mrr=row.get("mrr", pd.NA),
        ndcg_at_5=row.get("ndcg_at_5", pd.NA),
        retrieval_calls=row["retrieval_calls"],
        evidence_artifact_id=evidence_artifact_id,
    )


def _best_bandit_row(*, dataset: str, linucb: pd.DataFrame, train_best_reward: object) -> dict[str, object]:
    row = _best_bandit_method(linucb)
    return _paper_row(
        dataset=dataset,
        method_group="bandit",
        method="Best selected-action bandit baseline",
        objective="reward",
        objective_value=row["reward"],
        delta_vs_train_best=float(row["reward"]) - float(train_best_reward),
        ci_low=pd.NA,
        ci_high=pd.NA,
        recall_at_5=row.get("recall_at_5", pd.NA),
        mrr=row.get("mrr", pd.NA),
        ndcg_at_5=row.get("ndcg_at_5", pd.NA),
        retrieval_calls=row["retrieval_calls"],
        evidence_artifact_id=f"{dataset}_linucb_baseline_summary",
    )


def _best_bandit_method(linucb: pd.DataFrame) -> pd.Series:
    available = linucb[linucb["method"].isin(BANDIT_BASELINE_METHODS)]
    if available.empty:
        raise ValueError("Missing selected-action bandit baseline rows")
    return available.sort_values("reward", ascending=False).iloc[0]


def _constrained_row(
    *,
    dataset: str,
    sweep: pd.DataFrame,
    bootstrap: pd.DataFrame,
    call_penalty: float,
) -> dict[str, object]:
    row = _penalty_row(sweep, call_penalty)
    ci = _penalty_row(bootstrap, call_penalty)
    return _paper_row(
        dataset=dataset,
        method_group="constrained",
        method=f"Constrained policy lambda={call_penalty:.2f}",
        objective=f"utility_lambda_{call_penalty:.2f}",
        objective_value=row["policy_utility"],
        delta_vs_train_best=ci["utility_delta_mean"],
        ci_low=ci["utility_delta_ci_low"],
        ci_high=ci["utility_delta_ci_high"],
        recall_at_5=row.get("policy_recall_at_5", pd.NA),
        mrr=row.get("policy_mrr", pd.NA),
        ndcg_at_5=row.get("policy_ndcg_at_5", pd.NA),
        retrieval_calls=row["policy_retrieval_calls"],
        evidence_artifact_id=f"{dataset}_constrained_policy_bootstrap",
    )


def _paper_row(
    *,
    dataset: str,
    method_group: str,
    method: str,
    objective: str,
    objective_value: object,
    delta_vs_train_best: object,
    ci_low: object,
    ci_high: object,
    recall_at_5: object,
    mrr: object,
    ndcg_at_5: object,
    retrieval_calls: object,
    evidence_artifact_id: str,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method_group": method_group,
        "method": method,
        "objective": objective,
        "objective_value": _round(objective_value),
        "delta_vs_train_best": _round(delta_vs_train_best),
        "ci_low": _round(ci_low),
        "ci_high": _round(ci_high),
        "recall_at_5": _round(recall_at_5),
        "mrr": _round(mrr),
        "ndcg_at_5": _round(ndcg_at_5),
        "retrieval_calls": _round(retrieval_calls),
        "evidence_artifact_id": evidence_artifact_id,
    }


def _write_latex_table(table: pd.DataFrame, output_tex: Path) -> None:
    display = table[
        [
            "dataset",
            "method",
            "objective",
            "objective_value",
            "delta_vs_train_best",
            "ci_low",
            "ci_high",
            "retrieval_calls",
        ]
    ].copy()
    output_tex.write_text(display.to_latex(index=False, na_rep=""), encoding="utf-8")


def _method(frame: pd.DataFrame, method: str) -> pd.Series:
    matches = frame[frame["method"] == method]
    if matches.empty:
        raise ValueError(f"Missing method row: {method}")
    return matches.iloc[0]


def _metric_row(frame: pd.DataFrame, metric: str) -> pd.Series:
    matches = frame[frame["metric"] == metric]
    if matches.empty:
        raise ValueError(f"Missing metric row: {metric}")
    return matches.iloc[0]


def _penalty_row(frame: pd.DataFrame, call_penalty: float) -> pd.Series:
    matches = frame[frame["call_penalty"].round(8) == round(call_penalty, 8)]
    if matches.empty:
        raise ValueError(f"Missing call penalty row: {call_penalty}")
    return matches.iloc[0]


def _round(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return round(float(value), 6)
