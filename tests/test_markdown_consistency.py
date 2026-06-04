from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.reports.markdown_consistency import CONSISTENCY_COLUMNS, export_markdown_consistency


def test_export_markdown_consistency_checks_claim_values_and_artifact_paths(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    output_csv = tmp_path / "outputs" / "results" / "final_markdown_consistency.csv"
    exported = export_markdown_consistency(root=tmp_path, output_csv=output_csv)

    rows = pd.read_csv(exported)
    assert list(rows.columns) == CONSISTENCY_COLUMNS
    assert set(rows["status"]) == {"pass"}
    assert {"claim_value", "artifact_path"}.issubset(set(rows["check_type"]))

    scifact = rows[rows["check_id"] == "scifact_policy_reward_delta"].iloc[0]
    assert scifact["expected_value"] == "0.033711"
    assert scifact["observed_value"] == "present_in_all_required_docs"
    assert "FINAL_SLIDES.md" in scifact["checked_files"]

    path_check = rows[rows["check_id"] == "artifact_path_outputs/figures/final_reward_delta_ci.png"].iloc[0]
    assert path_check["expected_value"] == "exists"
    assert path_check["observed_value"] == "exists"
    assert path_check["evidence_path"] == "outputs/figures/final_reward_delta_ci.png"


def test_export_markdown_consistency_reports_missing_claim_value(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "FINAL_SLIDES.md").write_text("SciFact delta +0.999999\n", encoding="utf-8")

    exported = export_markdown_consistency(
        root=tmp_path,
        output_csv=tmp_path / "outputs" / "results" / "final_markdown_consistency.csv",
    )

    rows = pd.read_csv(exported)
    scifact = rows[rows["check_id"] == "scifact_policy_reward_delta"].iloc[0]
    assert scifact["status"] == "fail"
    assert "FINAL_SLIDES.md" in scifact["observed_value"]


def _write_fixture(root: Path) -> None:
    results = root / "outputs" / "results"
    figures = root / "outputs" / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "dataset": "scifact",
                "method": "Selective retrieval policy",
                "delta_vs_train_best": 0.033711,
                "ci_low": 0.008852,
                "ci_high": 0.058341,
            },
            {
                "dataset": "nfcorpus",
                "method": "Selective retrieval policy",
                "delta_vs_train_best": 0.029942,
                "ci_low": 0.005428,
                "ci_high": 0.054799,
            },
            {
                "dataset": "scifact",
                "method": "Constrained policy lambda=0.03",
                "delta_vs_train_best": 0.04735,
                "ci_low": 0.014315,
                "ci_high": 0.082247,
            },
            {
                "dataset": "nfcorpus",
                "method": "Constrained policy lambda=0.03",
                "delta_vs_train_best": 0.032656,
                "ci_low": 0.010587,
                "ci_high": 0.058431,
            },
        ]
    ).to_csv(results / "final_main_results_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "claim_id": "scifact_ope_ips_coverage_warning",
                "value": 0.267959,
                "baseline_value": 0.127173,
                "delta": 0.140786,
            },
            {
                "claim_id": "nfcorpus_ope_dr_stability",
                "value": 0.038419,
                "baseline_value": 0.05215,
                "delta": -0.013732,
            },
        ]
    ).to_csv(results / "final_claims_matrix.csv", index=False)

    for path in [
        results / "final_main_results_table.csv",
        results / "final_claims_matrix.csv",
        results / "final_artifact_index.csv",
        figures / "final_reward_delta_ci.png",
        figures / "final_cost_reward_frontier.png",
        figures / "final_ope_estimator_error.png",
        figures / "final_linucb_comparison.png",
    ]:
        path.write_text("artifact\n", encoding="utf-8") if not path.exists() else None

    text = """
SciFact +0.033711 CI [0.008852, 0.058341]
NFCorpus +0.029942 CI [0.005428, 0.054799]
SciFact constrained +0.047350 CI [0.014315, 0.082247]
NFCorpus constrained +0.032656 CI [0.010587, 0.058431]
OPE values 0.267959 0.127173 0.038419 0.052150
`outputs/results/final_main_results_table.csv`
`outputs/results/final_claims_matrix.csv`
`outputs/figures/final_reward_delta_ci.png`
`outputs/figures/final_cost_reward_frontier.png`
`outputs/figures/final_ope_estimator_error.png`
`outputs/figures/final_linucb_comparison.png`
"""
    for name in [
        "FINAL_SLIDES.md",
        "FINAL_DEFENSE_QA.md",
        "FINAL_RESULTS_SUMMARY.md",
        "FINAL_PRESENTATION_OUTLINE.md",
        "FINAL_REPORT.md",
    ]:
        (root / name).write_text(text, encoding="utf-8")
