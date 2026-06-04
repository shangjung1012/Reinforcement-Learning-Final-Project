from __future__ import annotations

from pathlib import Path

import pandas as pd


CLAIM_COLUMNS = [
    "claim_id",
    "claim_type",
    "claim",
    "primary_dataset",
    "metric",
    "value",
    "baseline",
    "baseline_value",
    "delta",
    "ci_low",
    "ci_high",
    "calls",
    "baseline_calls",
    "call_delta",
    "evidence_artifact_id",
    "evidence_path",
    "producer_command",
    "presentation_use",
]


def export_final_claims_matrix(*, root: Path, output_csv: Path) -> Path:
    rows = [
        _policy_gain_claim(root, dataset="scifact"),
        _policy_gain_claim(root, dataset="nfcorpus"),
        _selected_action_baseline_claim(root, dataset="scifact"),
        _selected_action_baseline_claim(root, dataset="nfcorpus"),
        _constrained_claim(root, dataset="scifact", call_penalty=0.03),
        _constrained_claim(root, dataset="nfcorpus", call_penalty=0.03),
        _ope_ips_coverage_claim(root, dataset="scifact"),
        _ope_dr_stability_claim(root, dataset="nfcorpus"),
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CLAIM_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _policy_gain_claim(root: Path, *, dataset: str) -> dict[str, object]:
    summary = pd.read_csv(root / "outputs" / "results" / f"{dataset}_retrieval_policy_summary.csv")
    bootstrap = pd.read_csv(root / "outputs" / "results" / f"{dataset}_bootstrap_diagnostics.csv")
    policy = _method(summary, "Selective retrieval policy")
    train_best = _method(summary, "Train-best retrieval action")
    reward_ci = _bootstrap_metric(bootstrap, "reward")
    call_delta = _round(policy["retrieval_calls"] - train_best["retrieval_calls"])
    return _row(
        claim_id=f"{dataset}_policy_reward_gain",
        claim_type="main_result",
        claim=(
            f"The learned retrieval policy improves held-out reward over the train-selected fixed "
            f"retrieval action on {dataset}."
        ),
        primary_dataset=dataset,
        metric="reward",
        value=reward_ci["mean_policy"],
        baseline="Train-best retrieval action",
        baseline_value=reward_ci["mean_baseline"],
        delta=reward_ci["mean_delta"],
        ci_low=reward_ci["ci_lower"],
        ci_high=reward_ci["ci_upper"],
        calls=policy["retrieval_calls"],
        baseline_calls=train_best["retrieval_calls"],
        call_delta=call_delta,
        evidence_artifact_id=f"{dataset}_bootstrap",
        evidence_path=f"outputs/results/{dataset}_bootstrap_diagnostics.csv[metric=reward]",
        producer_command=(
            f"uv run python scripts/run_statistical_diagnostics.py --dataset {dataset} "
            f"--detailed-csv outputs/results/{dataset}_retrieval_policy_detailed.csv "
            f"--output-csv outputs/results/{dataset}_bootstrap_diagnostics.csv"
        ),
        presentation_use="Main result slide: policy versus fixed retrieval-action baseline.",
    )


BANDIT_BASELINE_METHODS = [
    "LinUCB retrieval policy",
    "Epsilon-greedy retrieval policy",
    "Linear Thompson retrieval policy",
]


def _selected_action_baseline_claim(root: Path, *, dataset: str) -> dict[str, object]:
    summary = pd.read_csv(root / "outputs" / "results" / f"{dataset}_linucb_baseline_summary.csv")
    baseline = _best_bandit_method(summary)
    train_best = _method(summary, "Train-best retrieval action")
    return _row(
        claim_id=f"{dataset}_selected_action_baseline",
        claim_type="bandit_baseline",
        claim=(
            f"The best selected-action contextual-bandit baseline for {dataset} separates "
            "bandit feedback difficulty from the full-information direct-method policy."
        ),
        primary_dataset=dataset,
        metric="reward",
        value=baseline["reward"],
        baseline="Train-best retrieval action",
        baseline_value=train_best["reward"],
        delta=_round(baseline["reward"] - train_best["reward"]),
        calls=baseline["retrieval_calls"],
        baseline_calls=train_best["retrieval_calls"],
        call_delta=_round(baseline["retrieval_calls"] - train_best["retrieval_calls"]),
        evidence_artifact_id=f"{dataset}_linucb_baseline_summary",
        evidence_path=f"outputs/results/{dataset}_linucb_baseline_summary.csv[{baseline['method']}]",
        producer_command=(
            f"uv run python scripts/run_bandit_baselines.py --dataset {dataset} --num-train-examples 600 "
            "--num-test-examples 300 --seed 42 --full-corpus --alpha 1.0 --epsilon 0.1 --posterior-scale 1.0"
        ),
        presentation_use="RL framing slide: compact selected-action contextual bandit baseline.",
    )


def _constrained_claim(root: Path, *, dataset: str, call_penalty: float) -> dict[str, object]:
    bootstrap = pd.read_csv(root / "outputs" / "results" / f"{dataset}_constrained_policy_bootstrap.csv")
    row = _penalty_row(bootstrap, call_penalty)
    return _row(
        claim_id=f"{dataset}_constrained_bootstrap_gain",
        claim_type="constrained_bandit",
        claim=(
            f"The constrained policy keeps a positive paired-bootstrap utility gain on {dataset} "
            f"at retrieval-call penalty lambda={call_penalty:g}."
        ),
        primary_dataset=dataset,
        metric="utility_delta",
        value=row["utility_delta_mean"],
        baseline="Train-best retrieval action",
        baseline_value=row["train_best_utility_mean"],
        delta=row["utility_delta_mean"],
        ci_low=row["utility_delta_ci_low"],
        ci_high=row["utility_delta_ci_high"],
        calls=row["policy_calls_mean"],
        baseline_calls=row["train_best_calls_mean"],
        call_delta=row["call_delta_mean"],
        evidence_artifact_id=f"{dataset}_constrained_policy_bootstrap",
        evidence_path=f"outputs/results/{dataset}_constrained_policy_bootstrap.csv[call_penalty={call_penalty:g}]",
        producer_command=(
            f"uv run python scripts/run_constrained_policy_bootstrap.py --dataset {dataset} "
            f"--detailed-csv outputs/results/{dataset}_retrieval_policy_detailed.csv "
            f"--output-csv outputs/results/{dataset}_constrained_policy_bootstrap.csv "
            "--call-penalties 0,0.01,0.03,0.06,0.1,0.2 --bootstrap-samples 1000 --seed 42"
        ),
        presentation_use="Budget/constrained RL slide: utility gain with uncertainty interval.",
    )


def _ope_ips_coverage_claim(root: Path, *, dataset: str) -> dict[str, object]:
    stability = pd.read_csv(root / "outputs" / "results" / f"{dataset}_ope_stability.csv")
    ips_sparse = _ope_row(stability, behavior_policy="train_best_eps_0.2", estimator="ips")
    ips_uniform = _ope_row(stability, behavior_policy="uniform", estimator="ips")
    return _row(
        claim_id=f"{dataset}_ope_ips_coverage_warning",
        claim_type="off_policy_evaluation",
        claim=(
            f"IPS OPE becomes much less reliable on {dataset} when the behavior policy has poor "
            "coverage of the learned policy actions."
        ),
        primary_dataset=dataset,
        metric="mean_absolute_error",
        value=ips_sparse["mean_absolute_error"],
        baseline="Uniform behavior IPS",
        baseline_value=ips_uniform["mean_absolute_error"],
        delta=_round(ips_sparse["mean_absolute_error"] - ips_uniform["mean_absolute_error"]),
        calls=ips_sparse["mean_effective_sample_size"],
        baseline_calls=ips_uniform["mean_effective_sample_size"],
        call_delta=_round(ips_sparse["mean_effective_sample_size"] - ips_uniform["mean_effective_sample_size"]),
        evidence_artifact_id=f"{dataset}_ope_stability",
        evidence_path=f"outputs/results/{dataset}_ope_stability.csv[train_best_eps_0.2/Selective/IPS]",
        producer_command=(
            f"uv run python scripts/run_ope_stability.py --dataset {dataset} "
            f"--detailed-csv outputs/results/{dataset}_retrieval_policy_detailed.csv "
            f"--output-csv outputs/results/{dataset}_ope_stability.csv --seeds 1,2,3,4,5,6,7,8,9,10"
        ),
        presentation_use="OPE slide: why logging coverage matters in offline contextual bandits.",
    )


def _ope_dr_stability_claim(root: Path, *, dataset: str) -> dict[str, object]:
    stability = pd.read_csv(root / "outputs" / "results" / f"{dataset}_ope_stability.csv")
    dr = _ope_row(stability, behavior_policy="uniform", estimator="doubly_robust")
    dm = _ope_row(stability, behavior_policy="uniform", estimator="direct_method")
    return _row(
        claim_id=f"{dataset}_ope_dr_stability",
        claim_type="off_policy_evaluation",
        claim=(
            f"Doubly robust OPE is a useful diagnostic on {dataset} because it reduces uniform-log "
            "absolute error relative to the direct-method estimate."
        ),
        primary_dataset=dataset,
        metric="mean_absolute_error",
        value=dr["mean_absolute_error"],
        baseline="Direct method",
        baseline_value=dm["mean_absolute_error"],
        delta=_round(dr["mean_absolute_error"] - dm["mean_absolute_error"]),
        calls=dr["mean_effective_sample_size"],
        baseline_calls=dm["mean_effective_sample_size"],
        call_delta=_round(dr["mean_effective_sample_size"] - dm["mean_effective_sample_size"]),
        evidence_artifact_id=f"{dataset}_ope_stability",
        evidence_path=f"outputs/results/{dataset}_ope_stability.csv[uniform/Selective/doubly_robust]",
        producer_command=(
            f"uv run python scripts/run_ope_stability.py --dataset {dataset} "
            f"--detailed-csv outputs/results/{dataset}_retrieval_policy_detailed.csv "
            f"--output-csv outputs/results/{dataset}_ope_stability.csv --seeds 1,2,3,4,5,6,7,8,9,10"
        ),
        presentation_use="OPE slide: estimator comparison under repeated logging simulations.",
    )


def _method(frame: pd.DataFrame, method: str) -> pd.Series:
    matches = frame[frame["method"] == method]
    if matches.empty:
        raise ValueError(f"Missing method row: {method}")
    return matches.iloc[0]


def _best_bandit_method(frame: pd.DataFrame) -> pd.Series:
    matches = frame[frame["method"].isin(BANDIT_BASELINE_METHODS)]
    if matches.empty:
        raise ValueError("Missing selected-action bandit baseline rows")
    return matches.sort_values("reward", ascending=False).iloc[0]


def _penalty_row(frame: pd.DataFrame, call_penalty: float) -> pd.Series:
    matches = frame[frame["call_penalty"].round(8) == round(call_penalty, 8)]
    if matches.empty:
        raise ValueError(f"Missing constrained row for call_penalty={call_penalty}")
    return matches.iloc[0]


def _bootstrap_metric(frame: pd.DataFrame, metric: str) -> pd.Series:
    matches = frame[frame["metric"] == metric]
    if matches.empty:
        raise ValueError(f"Missing bootstrap metric row: {metric}")
    return matches.iloc[0]


def _ope_row(frame: pd.DataFrame, *, behavior_policy: str, estimator: str) -> pd.Series:
    matches = frame[
        (frame["behavior_policy"] == behavior_policy)
        & (frame["target_method"] == "Selective retrieval policy")
        & (frame["estimator"] == estimator)
    ]
    if matches.empty:
        raise ValueError(f"Missing OPE row: {behavior_policy}/Selective retrieval policy/{estimator}")
    return matches.iloc[0]


def _row(
    *,
    claim_id: str,
    claim_type: str,
    claim: str,
    primary_dataset: str,
    metric: str,
    value: object,
    baseline: str,
    baseline_value: object,
    delta: object,
    evidence_artifact_id: str,
    evidence_path: str,
    producer_command: str,
    presentation_use: str,
    ci_low: object = pd.NA,
    ci_high: object = pd.NA,
    calls: object = pd.NA,
    baseline_calls: object = pd.NA,
    call_delta: object = pd.NA,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "claim": claim,
        "primary_dataset": primary_dataset,
        "metric": metric,
        "value": _round(value),
        "baseline": baseline,
        "baseline_value": _round(baseline_value),
        "delta": _round(delta),
        "ci_low": _round(ci_low),
        "ci_high": _round(ci_high),
        "calls": _round(calls),
        "baseline_calls": _round(baseline_calls),
        "call_delta": _round(call_delta),
        "evidence_artifact_id": evidence_artifact_id,
        "evidence_path": evidence_path,
        "producer_command": producer_command,
        "presentation_use": presentation_use,
    }


def _round(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return round(float(value), 6)
