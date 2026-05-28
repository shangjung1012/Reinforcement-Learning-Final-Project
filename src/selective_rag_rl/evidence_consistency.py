from __future__ import annotations

from pathlib import Path

import pandas as pd


CONSISTENCY_COLUMNS = [
    "claim_id",
    "source_artifact",
    "expected_value",
    "observed_value",
    "status",
    "evidence_path",
]


def export_evidence_consistency(
    *,
    final_report_md: Path,
    protocol_summary_csv: Path,
    deployment_decision_csv: Path,
    artifact_index_csv: Path,
    output_csv: Path,
) -> Path:
    report_text = final_report_md.read_text(encoding="utf-8")
    protocol = pd.read_csv(protocol_summary_csv)
    deployment = pd.read_csv(deployment_decision_csv)
    artifact_index = pd.read_csv(artifact_index_csv)

    decision = deployment.iloc[0]
    rows = [
        _text_and_value_check(
            claim_id="deployment_confidence",
            source_artifact=deployment_decision_csv,
            expected=str(decision["decision_confidence"]),
            report_text=report_text,
            evidence_path="deployment_decision.decision_confidence",
        ),
        _text_and_value_check(
            claim_id="runtime_policy",
            source_artifact=deployment_decision_csv,
            expected=str(decision["recommended_runtime_policy"]),
            report_text=report_text,
            evidence_path="deployment_decision.recommended_runtime_policy",
        ),
        _depth_ci_check(
            claim_id="full_auto_depth_effect_ci",
            protocol=protocol,
            feature_set="full",
            policy_model="auto",
            report_text=report_text,
            deployment_summary=str(decision["evidence_summary"]),
            source_artifact=protocol_summary_csv,
        ),
        _depth_ci_check(
            claim_id="full_ridge_depth_effect_ci",
            protocol=protocol,
            feature_set="full",
            policy_model="ridge",
            report_text=report_text,
            deployment_summary=str(decision["evidence_summary"]),
            source_artifact=protocol_summary_csv,
        ),
        _artifact_index_check(
            claim_id="protocol_artifact_indexed",
            artifact_index=artifact_index,
            artifact_id="nfcorpus_vertex_protocol_summary",
            source_artifact=artifact_index_csv,
        ),
        _artifact_index_check(
            claim_id="deployment_artifact_indexed",
            artifact_index=artifact_index,
            artifact_id="nfcorpus_vertex_deployment_decision",
            source_artifact=artifact_index_csv,
        ),
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CONSISTENCY_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _text_and_value_check(
    *,
    claim_id: str,
    source_artifact: Path,
    expected: str,
    report_text: str,
    evidence_path: str,
) -> dict[str, str]:
    observed = expected if expected in report_text else "missing_in_report"
    return _row(
        claim_id=claim_id,
        source_artifact=source_artifact,
        expected=expected,
        observed=observed,
        evidence_path=evidence_path,
    )


def _depth_ci_check(
    *,
    claim_id: str,
    protocol: pd.DataFrame,
    feature_set: str,
    policy_model: str,
    report_text: str,
    deployment_summary: str,
    source_artifact: Path,
) -> dict[str, str]:
    matches = protocol[
        (protocol["selection_layer"] == "semantic_depth")
        & (protocol["feature_set"] == feature_set)
        & (protocol["policy_model"] == policy_model)
    ]
    if matches.empty:
        return _row(
            claim_id=claim_id,
            source_artifact=source_artifact,
            expected=f"{feature_set}/{policy_model}",
            observed="missing_in_protocol",
            evidence_path=f"protocol[{feature_set}/{policy_model}]",
        )

    protocol_row = matches.iloc[0]
    expected = _ci_text(protocol_row["depth_effect_ci_low"], protocol_row["depth_effect_ci_high"])
    compact_expected = expected.replace(" ", "")
    observed = expected if expected in report_text and compact_expected in deployment_summary else "missing"
    return _row(
        claim_id=claim_id,
        source_artifact=source_artifact,
        expected=expected,
        observed=observed,
        evidence_path=f"protocol[{feature_set}/{policy_model}].depth_effect_ci",
    )


def _artifact_index_check(
    *,
    claim_id: str,
    artifact_index: pd.DataFrame,
    artifact_id: str,
    source_artifact: Path,
) -> dict[str, str]:
    matches = artifact_index[artifact_index["artifact_id"] == artifact_id]
    exists = bool(matches.iloc[0]["exists"]) if not matches.empty else False
    observed = "indexed" if exists else "missing"
    return _row(
        claim_id=claim_id,
        source_artifact=source_artifact,
        expected="indexed",
        observed=observed,
        evidence_path=f"artifact_index[{artifact_id}]",
    )


def _row(
    *,
    claim_id: str,
    source_artifact: Path,
    expected: str,
    observed: str,
    evidence_path: str,
) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "source_artifact": str(source_artifact),
        "expected_value": expected,
        "observed_value": observed,
        "status": "pass" if observed == expected or (expected == "indexed" and observed == "indexed") else "fail",
        "evidence_path": evidence_path,
    }


def _ci_text(low: object, high: object) -> str:
    return f"[{float(low):.6f}, {float(high):.6f}]"
