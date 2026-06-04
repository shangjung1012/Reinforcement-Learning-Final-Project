from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.reports.evidence_consistency import export_evidence_consistency


def test_export_evidence_consistency_checks_report_against_machine_artifacts(tmp_path: Path) -> None:
    report = tmp_path / "FINAL_REPORT.md"
    protocol_csv = tmp_path / "protocol.csv"
    deployment_csv = tmp_path / "deployment.csv"
    artifact_index_csv = tmp_path / "artifact_index.csv"
    output_csv = tmp_path / "evidence_consistency.csv"

    report.write_text(
        "\n".join(
            [
                "The deployment record reports `decision_confidence=pilot_low_n`.",
                "`recommended_runtime_policy=train_best_fixed_retrieval_action` is used.",
                "The full/auto interval is [-0.019894, 0.008454].",
                "The full/ridge interval is [-0.008071, 0.010826].",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "selection_layer": "semantic_depth",
                "feature_set": "full",
                "policy_model": "auto",
                "depth_effect_ci_low": -0.019893583532,
                "depth_effect_ci_high": 0.008454170838,
            },
            {
                "selection_layer": "semantic_depth",
                "feature_set": "full",
                "policy_model": "ridge",
                "depth_effect_ci_low": -0.008070838562,
                "depth_effect_ci_high": 0.010826263105,
            },
        ]
    ).to_csv(protocol_csv, index=False)
    pd.DataFrame(
        [
            {
                "recommended_runtime_policy": "train_best_fixed_retrieval_action",
                "decision_confidence": "pilot_low_n",
                "evidence_summary": (
                    "full/auto: depth_effect_ci=[-0.019894,0.008454] | "
                    "full/ridge: depth_effect_ci=[-0.008071,0.010826]"
                ),
            }
        ]
    ).to_csv(deployment_csv, index=False)
    pd.DataFrame(
        [
            {"artifact_id": "nfcorpus_vertex_protocol_summary", "exists": True},
            {"artifact_id": "nfcorpus_vertex_deployment_decision", "exists": True},
        ]
    ).to_csv(artifact_index_csv, index=False)

    exported = export_evidence_consistency(
        final_report_md=report,
        protocol_summary_csv=protocol_csv,
        deployment_decision_csv=deployment_csv,
        artifact_index_csv=artifact_index_csv,
        output_csv=output_csv,
    )
    checks = pd.read_csv(exported)

    assert set(checks["status"]) == {"pass"}
    assert set(checks["claim_id"]) == {
        "deployment_confidence",
        "runtime_policy",
        "full_auto_depth_effect_ci",
        "full_ridge_depth_effect_ci",
        "protocol_artifact_indexed",
        "deployment_artifact_indexed",
    }


def test_export_evidence_consistency_flags_missing_report_claim(tmp_path: Path) -> None:
    report = tmp_path / "FINAL_REPORT.md"
    protocol_csv = tmp_path / "protocol.csv"
    deployment_csv = tmp_path / "deployment.csv"
    artifact_index_csv = tmp_path / "artifact_index.csv"
    output_csv = tmp_path / "evidence_consistency.csv"

    report.write_text("No deployment confidence is stated here.", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "selection_layer": "semantic_depth",
                "feature_set": "full",
                "policy_model": "auto",
                "depth_effect_ci_low": -0.019893583532,
                "depth_effect_ci_high": 0.008454170838,
            },
            {
                "selection_layer": "semantic_depth",
                "feature_set": "full",
                "policy_model": "ridge",
                "depth_effect_ci_low": -0.008070838562,
                "depth_effect_ci_high": 0.010826263105,
            },
        ]
    ).to_csv(protocol_csv, index=False)
    pd.DataFrame(
        [
            {
                "recommended_runtime_policy": "train_best_fixed_retrieval_action",
                "decision_confidence": "pilot_low_n",
                "evidence_summary": "",
            }
        ]
    ).to_csv(deployment_csv, index=False)
    pd.DataFrame(
        [
            {"artifact_id": "nfcorpus_vertex_protocol_summary", "exists": True},
            {"artifact_id": "nfcorpus_vertex_deployment_decision", "exists": True},
        ]
    ).to_csv(artifact_index_csv, index=False)

    exported = export_evidence_consistency(
        final_report_md=report,
        protocol_summary_csv=protocol_csv,
        deployment_decision_csv=deployment_csv,
        artifact_index_csv=artifact_index_csv,
        output_csv=output_csv,
    )
    checks = pd.read_csv(exported)

    confidence = checks[checks["claim_id"] == "deployment_confidence"].iloc[0]
    auto_ci = checks[checks["claim_id"] == "full_auto_depth_effect_ci"].iloc[0]
    assert confidence["status"] == "fail"
    assert confidence["observed_value"] == "missing_in_report"
    assert auto_ci["status"] == "fail"
