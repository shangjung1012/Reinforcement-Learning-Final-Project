from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.selection_protocol import export_deployment_decision, export_selection_protocol_summary


def test_export_selection_protocol_summary_recommends_guardrails(tmp_path: Path) -> None:
    policy_csv = tmp_path / "policy_diagnostics.csv"
    depth_csv = tmp_path / "depth_diagnostics.csv"
    depth_stability_csv = tmp_path / "depth_stability.csv"
    output_csv = tmp_path / "protocol.csv"
    pd.DataFrame(
        [
            _policy_row(1, reward_gap=-0.02, validation_gap=-0.01, call_gap=0.03, dominated=True),
            _policy_row(2, reward_gap=0.00, validation_gap=-0.02, call_gap=0.00, dominated=True),
            _policy_row(3, reward_gap=0.01, validation_gap=0.01, call_gap=0.10, dominated=False),
        ]
    ).to_csv(policy_csv, index=False)
    pd.DataFrame(
        [
            _depth_row(1, "full", "ridge", validation_depth=5, heldout_depth=8, reward_gap=0.02),
            _depth_row(2, "full", "ridge", validation_depth=8, heldout_depth=5, reward_gap=0.01),
            _depth_row(1, "no_semantic", "ridge", validation_depth=5, heldout_depth=5, reward_gap=0.00),
            _depth_row(2, "no_semantic", "ridge", validation_depth=5, heldout_depth=5, reward_gap=0.00),
        ]
    ).to_csv(depth_csv, index=False)
    pd.DataFrame(
        [
            {
                "feature_set": "full",
                "policy_model": "ridge",
                "semantic_depth": 8,
                "baseline_semantic_depth": 5,
                "selective_reward_delta_across_seed_mean": 0.015,
                "selective_reward_delta_ci_low": -0.01,
                "selective_reward_delta_ci_high": 0.04,
            },
            {
                "feature_set": "no_semantic",
                "policy_model": "ridge",
                "semantic_depth": 8,
                "baseline_semantic_depth": 5,
                "selective_reward_delta_across_seed_mean": 0.0,
                "selective_reward_delta_ci_low": 0.0,
                "selective_reward_delta_ci_high": 0.0,
            },
        ]
    ).to_csv(depth_stability_csv, index=False)

    exported = export_selection_protocol_summary(
        policy_diagnostics_csv=policy_csv,
        depth_selection_diagnostics_csv=depth_csv,
        depth_stability_csv=depth_stability_csv,
        output_csv=output_csv,
        dataset="toy",
    )
    summary = pd.read_csv(exported)
    policy = summary[summary["selection_layer"] == "policy_model"].iloc[0]
    full_depth = summary[
        (summary["selection_layer"] == "semantic_depth")
        & (summary["feature_set"] == "full")
        & (summary["policy_model"] == "ridge")
    ].iloc[0]
    no_semantic_depth = summary[
        (summary["selection_layer"] == "semantic_depth")
        & (summary["feature_set"] == "no_semantic")
        & (summary["policy_model"] == "ridge")
    ].iloc[0]

    assert policy["dataset"] == "toy"
    assert policy["n_runs"] == 3
    assert policy["fallback_rate"] == 2 / 3
    assert policy["support_rate"] == 1 / 3
    assert policy["support_rate_ci_low"] == 0.061490315276
    assert policy["support_rate_ci_high"] == 0.792345044874
    assert policy["mean_reward_gap_vs_guardrail"] == round((-0.02 + 0.00 + 0.01) / 3, 12)
    assert policy["recommendation"] == "fallback_to_train_best_fixed"
    assert policy["decision_reason"] == "fallback_rate=0.667; mean_reward_gap=-0.003333; mean_call_gap=0.043333"
    assert policy["evidence_strength"] == "pilot_low_n"
    assert policy["evidence_caveat"] == "n_runs=3; support_ci_width=0.731"
    assert full_depth["match_rate"] == 0.0
    assert full_depth["support_rate"] == 0.0
    assert full_depth["support_rate_ci_low"] == 0.0
    assert full_depth["support_rate_ci_high"] == 0.65762804711
    assert full_depth["mean_selection_gap"] == 0.015
    assert full_depth["recommendation"] == "do_not_select_depth_by_validation"
    assert full_depth["decision_reason"] == "match_rate=0.000; mean_selection_gap=0.015000"
    assert full_depth["effect_semantic_depth"] == 8
    assert full_depth["effect_baseline_semantic_depth"] == 5
    assert full_depth["mean_depth_effect_vs_baseline"] == 0.015
    assert full_depth["depth_effect_ci_low"] == -0.01
    assert full_depth["depth_effect_ci_high"] == 0.04
    assert full_depth["evidence_strength"] == "pilot_low_n"
    assert full_depth["evidence_caveat"] == "n_runs=2; support_ci_width=0.658"
    assert no_semantic_depth["match_rate"] == 1.0
    assert no_semantic_depth["support_rate"] == 1.0
    assert no_semantic_depth["support_rate_ci_low"] == 0.34237195289
    assert no_semantic_depth["support_rate_ci_high"] == 1.0
    assert no_semantic_depth["recommendation"] == "validation_depth_selection_ok"


def test_export_deployment_decision_records_final_policy_choice(tmp_path: Path) -> None:
    protocol_csv = tmp_path / "protocol.csv"
    output_csv = tmp_path / "deployment_decision.csv"
    pd.DataFrame(
        [
            {
                "dataset": "toy",
                "selection_layer": "policy_model",
                "feature_set": "",
                "policy_model": "",
                "n_runs": 3,
                "fallback_rate": 2 / 3,
                "match_rate": "",
                "support_rate": 1 / 3,
                "mean_reward_gap_vs_guardrail": -0.01,
                "mean_validation_gap_vs_guardrail": -0.02,
                "mean_call_gap_vs_guardrail": 0.05,
                "mean_selection_gap": "",
                "recommendation": "fallback_to_train_best_fixed",
                "decision_reason": "fallback_rate=0.667; mean_reward_gap=-0.010000; mean_call_gap=0.050000",
                "evidence_strength": "pilot_low_n",
                "evidence_caveat": "n_runs=3; support_ci_width=0.731",
            },
            {
                "dataset": "toy",
                "selection_layer": "semantic_depth",
                "feature_set": "full",
                "policy_model": "ridge",
                "n_runs": 3,
                "fallback_rate": "",
                "match_rate": 0.0,
                "support_rate": 0.0,
                "mean_reward_gap_vs_guardrail": "",
                "mean_validation_gap_vs_guardrail": "",
                "mean_call_gap_vs_guardrail": "",
                "mean_selection_gap": 0.02,
                "effect_semantic_depth": 8,
                "effect_baseline_semantic_depth": 5,
                "mean_depth_effect_vs_baseline": 0.005,
                "depth_effect_ci_low": -0.01,
                "depth_effect_ci_high": 0.02,
                "recommendation": "do_not_select_depth_by_validation",
                "decision_reason": "match_rate=0.000; mean_selection_gap=0.020000",
                "evidence_strength": "pilot_low_n",
                "evidence_caveat": "n_runs=3; support_ci_width=0.562",
            },
        ]
    ).to_csv(protocol_csv, index=False)

    exported = export_deployment_decision(protocol_csv, output_csv)
    decision = pd.read_csv(exported).iloc[0]

    assert decision["dataset"] == "toy"
    assert decision["recommended_runtime_policy"] == "train_best_fixed_retrieval_action"
    assert decision["learned_policy_status"] == "analysis_only_guardrailed"
    assert decision["semantic_depth_strategy"] == "do_not_select_full_depth_by_validation"
    assert decision["semantic_feature_status"] == "analysis_only"
    assert decision["decision_confidence"] == "pilot_low_n"
    assert "fallback_rate=0.667" in decision["evidence_summary"]
    assert "full/ridge" in decision["evidence_summary"]
    assert "depth_effect_ci=[-0.010000,0.020000]" in decision["evidence_summary"]


def _policy_row(seed: int, reward_gap: float, validation_gap: float, call_gap: float, dominated: bool) -> dict[str, object]:
    return {
        "seed": seed,
        "validation_selected_reward_gap_vs_best_fixed": reward_gap,
        "validation_selected_validation_gap_vs_best_fixed": validation_gap,
        "validation_selected_call_gap_vs_best_fixed": call_gap,
        "validation_selected_dominated_by_best_fixed": dominated,
        "guardrail_decision": "fallback_train_best_fixed" if dominated else "keep_validation_selected",
    }


def _depth_row(
    seed: int,
    feature_set: str,
    policy_model: str,
    validation_depth: int,
    heldout_depth: int,
    reward_gap: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "feature_set": feature_set,
        "policy_model": policy_model,
        "validation_best_depth": validation_depth,
        "heldout_best_depth": heldout_depth,
        "validation_selected_heldout_delta": -reward_gap,
        "heldout_best_delta": 0.0,
        "depth_selection_reward_gap": reward_gap,
    }
