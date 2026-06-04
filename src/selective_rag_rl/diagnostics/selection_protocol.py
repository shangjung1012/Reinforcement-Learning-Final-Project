from __future__ import annotations

from pathlib import Path

import pandas as pd

PROTOCOL_COLUMNS = [
    "dataset",
    "selection_layer",
    "feature_set",
    "policy_model",
    "n_runs",
    "fallback_rate",
    "match_rate",
    "support_rate",
    "support_rate_ci_low",
    "support_rate_ci_high",
    "mean_reward_gap_vs_guardrail",
    "mean_validation_gap_vs_guardrail",
    "mean_call_gap_vs_guardrail",
    "mean_selection_gap",
    "effect_semantic_depth",
    "effect_baseline_semantic_depth",
    "mean_depth_effect_vs_baseline",
    "depth_effect_ci_low",
    "depth_effect_ci_high",
    "recommendation",
    "decision_reason",
    "evidence_strength",
    "evidence_caveat",
]

DEPLOYMENT_DECISION_COLUMNS = [
    "dataset",
    "recommended_runtime_policy",
    "learned_policy_status",
    "semantic_depth_strategy",
    "semantic_feature_status",
    "decision_confidence",
    "evidence_summary",
]


def export_selection_protocol_summary(
    *,
    policy_diagnostics_csv: Path,
    depth_selection_diagnostics_csv: Path,
    depth_stability_csv: Path | None = None,
    output_csv: Path,
    dataset: str,
) -> Path:
    depth_stability = pd.read_csv(depth_stability_csv) if depth_stability_csv is not None else None
    rows = [
        _policy_protocol_row(pd.read_csv(policy_diagnostics_csv), dataset=dataset),
        *_depth_protocol_rows(
            pd.read_csv(depth_selection_diagnostics_csv),
            dataset=dataset,
            depth_stability=depth_stability,
        ),
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PROTOCOL_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def export_deployment_decision(protocol_summary_csv: Path, output_csv: Path) -> Path:
    protocol = pd.read_csv(protocol_summary_csv)
    if protocol.empty:
        raise ValueError("Protocol summary CSV contains no rows")
    dataset = str(protocol["dataset"].iloc[0])
    policy_row = _required_layer(protocol, "policy_model")
    full_depth = protocol[
        (protocol["selection_layer"] == "semantic_depth")
        & (protocol["feature_set"] == "full")
    ]
    fallback_policy = str(policy_row["recommendation"]) == "fallback_to_train_best_fixed"
    unsafe_full_depth = bool((full_depth["recommendation"] != "validation_depth_selection_ok").any())
    decision_confidence = _minimum_evidence_strength(pd.concat([policy_row.to_frame().T, full_depth]))
    decision = {
        "dataset": dataset,
        "recommended_runtime_policy": (
            "train_best_fixed_retrieval_action" if fallback_policy else "validation_selected_policy"
        ),
        "learned_policy_status": "analysis_only_guardrailed" if fallback_policy else "deployable",
        "semantic_depth_strategy": (
            "do_not_select_full_depth_by_validation" if unsafe_full_depth else "validation_selected_full_depth"
        ),
        "semantic_feature_status": "analysis_only" if unsafe_full_depth else "candidate_runtime_feature",
        "decision_confidence": decision_confidence,
        "evidence_summary": _deployment_evidence_summary(policy_row, full_depth),
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([decision], columns=DEPLOYMENT_DECISION_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _policy_protocol_row(policy: pd.DataFrame, dataset: str) -> dict[str, object]:
    if policy.empty:
        raise ValueError("Policy diagnostics CSV contains no rows")
    fallback_rate = float((policy["guardrail_decision"] == "fallback_train_best_fixed").mean())
    mean_reward_gap = float(policy["validation_selected_reward_gap_vs_best_fixed"].mean())
    mean_validation_gap = float(policy["validation_selected_validation_gap_vs_best_fixed"].mean())
    mean_call_gap = float(policy["validation_selected_call_gap_vs_best_fixed"].mean())
    support_rate = 1.0 - fallback_rate
    n_runs = int(policy["seed"].nunique()) if "seed" in policy else int(len(policy))
    support_ci_low, support_ci_high = _wilson_interval(int(round(support_rate * n_runs)), n_runs)
    evidence_strength, evidence_caveat = _evidence_strength(n_runs, support_ci_low, support_ci_high)
    recommendation = (
        "fallback_to_train_best_fixed"
        if fallback_rate > 0 or mean_reward_gap <= 0
        else "keep_validation_selected_policy"
    )
    return {
        "dataset": dataset,
        "selection_layer": "policy_model",
        "feature_set": "",
        "policy_model": "",
        "n_runs": n_runs,
        "fallback_rate": fallback_rate,
        "match_rate": "",
        "support_rate": support_rate,
        "support_rate_ci_low": support_ci_low,
        "support_rate_ci_high": support_ci_high,
        "mean_reward_gap_vs_guardrail": round(mean_reward_gap, 12),
        "mean_validation_gap_vs_guardrail": round(mean_validation_gap, 12),
        "mean_call_gap_vs_guardrail": round(mean_call_gap, 12),
        "mean_selection_gap": "",
        "effect_semantic_depth": "",
        "effect_baseline_semantic_depth": "",
        "mean_depth_effect_vs_baseline": "",
        "depth_effect_ci_low": "",
        "depth_effect_ci_high": "",
        "recommendation": recommendation,
        "decision_reason": (
            f"fallback_rate={fallback_rate:.3f}; "
            f"mean_reward_gap={mean_reward_gap:.6f}; "
            f"mean_call_gap={mean_call_gap:.6f}"
        ),
        "evidence_strength": evidence_strength,
        "evidence_caveat": evidence_caveat,
    }


def _depth_protocol_rows(
    depth: pd.DataFrame,
    dataset: str,
    depth_stability: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    if depth.empty:
        raise ValueError("Depth-selection diagnostics CSV contains no rows")
    rows = []
    for (feature_set, policy_model), group in depth.groupby(["feature_set", "policy_model"], sort=True):
        match_rate = float((group["validation_best_depth"] == group["heldout_best_depth"]).mean())
        mean_selection_gap = float(group["depth_selection_reward_gap"].mean())
        support_rate = match_rate
        n_runs = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        support_ci_low, support_ci_high = _wilson_interval(int(round(support_rate * n_runs)), n_runs)
        evidence_strength, evidence_caveat = _evidence_strength(n_runs, support_ci_low, support_ci_high)
        effect = _depth_effect_row(depth_stability, feature_set, policy_model)
        recommendation = (
            "validation_depth_selection_ok"
            if match_rate >= 0.5 and mean_selection_gap <= 0
            else "do_not_select_depth_by_validation"
        )
        rows.append(
            {
                "dataset": dataset,
                "selection_layer": "semantic_depth",
                "feature_set": feature_set,
                "policy_model": policy_model,
                "n_runs": n_runs,
                "fallback_rate": "",
                "match_rate": match_rate,
                "support_rate": support_rate,
                "support_rate_ci_low": support_ci_low,
                "support_rate_ci_high": support_ci_high,
                "mean_reward_gap_vs_guardrail": "",
                "mean_validation_gap_vs_guardrail": "",
                "mean_call_gap_vs_guardrail": "",
                "mean_selection_gap": round(mean_selection_gap, 12),
                "effect_semantic_depth": effect["effect_semantic_depth"],
                "effect_baseline_semantic_depth": effect["effect_baseline_semantic_depth"],
                "mean_depth_effect_vs_baseline": effect["mean_depth_effect_vs_baseline"],
                "depth_effect_ci_low": effect["depth_effect_ci_low"],
                "depth_effect_ci_high": effect["depth_effect_ci_high"],
                "recommendation": recommendation,
                "decision_reason": f"match_rate={match_rate:.3f}; mean_selection_gap={mean_selection_gap:.6f}",
                "evidence_strength": evidence_strength,
                "evidence_caveat": evidence_caveat,
            }
        )
    return rows


def _depth_effect_row(
    depth_stability: pd.DataFrame | None,
    feature_set: str,
    policy_model: str,
) -> dict[str, object]:
    empty = {
        "effect_semantic_depth": "",
        "effect_baseline_semantic_depth": "",
        "mean_depth_effect_vs_baseline": "",
        "depth_effect_ci_low": "",
        "depth_effect_ci_high": "",
    }
    if depth_stability is None or depth_stability.empty:
        return empty
    matches = depth_stability[
        (depth_stability["feature_set"] == feature_set)
        & (depth_stability["policy_model"] == policy_model)
    ]
    if matches.empty:
        return empty
    row = matches.sort_values(["semantic_depth", "baseline_semantic_depth"]).iloc[0]
    return {
        "effect_semantic_depth": int(row["semantic_depth"]),
        "effect_baseline_semantic_depth": int(row["baseline_semantic_depth"]),
        "mean_depth_effect_vs_baseline": round(float(row["selective_reward_delta_across_seed_mean"]), 12),
        "depth_effect_ci_low": round(float(row["selective_reward_delta_ci_low"]), 12),
        "depth_effect_ci_high": round(float(row["selective_reward_delta_ci_high"]), 12),
    }


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / denominator
    return round(max(0.0, center - margin), 12), round(min(1.0, center + margin), 12)


def _evidence_strength(n_runs: int, ci_low: float, ci_high: float) -> tuple[str, str]:
    ci_width = ci_high - ci_low
    if n_runs < 5 or ci_width > 0.5:
        strength = "pilot_low_n"
    elif ci_width > 0.25:
        strength = "moderate"
    else:
        strength = "strong"
    return strength, f"n_runs={n_runs}; support_ci_width={ci_width:.3f}"


def _minimum_evidence_strength(rows: pd.DataFrame) -> str:
    order = {"pilot_low_n": 0, "moderate": 1, "strong": 2}
    values = [str(value) for value in rows.get("evidence_strength", pd.Series(dtype=object)).dropna()]
    if not values:
        return "unknown"
    return min(values, key=lambda value: order.get(value, -1))


def _required_layer(protocol: pd.DataFrame, selection_layer: str) -> pd.Series:
    rows = protocol[protocol["selection_layer"] == selection_layer]
    if rows.empty:
        raise ValueError(f"Protocol summary missing {selection_layer!r} row")
    return rows.iloc[0]


def _deployment_evidence_summary(policy_row: pd.Series, full_depth: pd.DataFrame) -> str:
    evidence = [f"policy: {policy_row['recommendation']} ({policy_row['decision_reason']})"]
    for _, row in full_depth.sort_values(["policy_model"]).iterrows():
        effect = _deployment_depth_effect_summary(row)
        evidence.append(
            f"full/{row['policy_model']}: {row['recommendation']} ({row['decision_reason']}{effect})"
        )
    return " | ".join(evidence)


def _deployment_depth_effect_summary(row: pd.Series) -> str:
    required = ["effect_semantic_depth", "effect_baseline_semantic_depth", "depth_effect_ci_low", "depth_effect_ci_high"]
    if any(column not in row or pd.isna(row[column]) or row[column] == "" for column in required):
        return ""
    return (
        f"; depth_effect_pair={int(row['effect_semantic_depth'])}_vs{int(row['effect_baseline_semantic_depth'])}"
        f"; depth_effect_ci=[{float(row['depth_effect_ci_low']):.6f},{float(row['depth_effect_ci_high']):.6f}]"
    )
