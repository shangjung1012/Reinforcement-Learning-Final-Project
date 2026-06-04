from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.reports.artifact_index import (
    ArtifactSpec,
    export_artifact_index,
    final_project_artifact_specs,
)


def test_export_artifact_index_records_existing_and_missing_files(tmp_path: Path) -> None:
    existing = tmp_path / "outputs" / "results" / "summary.csv"
    existing.parent.mkdir(parents=True)
    payload = b"method,reward\npolicy,1.0\n"
    existing.write_bytes(payload)
    missing = tmp_path / "outputs" / "results" / "missing.csv"
    output_csv = tmp_path / "outputs" / "results" / "artifact_index.csv"

    exported = export_artifact_index(
        [
            ArtifactSpec(
                artifact_id="main_summary",
                category="result",
                path=existing,
                role="Primary held-out result table.",
                producer_command="uv run python scripts/run_example.py",
            ),
            ArtifactSpec(
                artifact_id="missing_optional",
                category="diagnostic",
                path=missing,
                role="Optional diagnostic.",
                producer_command="uv run python scripts/run_missing.py",
            ),
        ],
        output_csv=output_csv,
        root=tmp_path,
    )

    rows = pd.read_csv(exported)
    existing_row = rows[rows["artifact_id"] == "main_summary"].iloc[0]
    missing_row = rows[rows["artifact_id"] == "missing_optional"].iloc[0]

    assert existing_row["path"] == "outputs/results/summary.csv"
    assert bool(existing_row["exists"]) is True
    assert existing_row["size_bytes"] == len(payload)
    assert existing_row["sha256_12"] == "c39a45d5625c"
    assert existing_row["role"] == "Primary held-out result table."
    assert existing_row["producer_command"] == "uv run python scripts/run_example.py"

    assert missing_row["path"] == "outputs/results/missing.csv"
    assert bool(missing_row["exists"]) is False
    assert missing_row["size_bytes"] == 0
    assert pd.isna(missing_row["sha256_12"])


def test_final_project_artifact_specs_include_depth_stability_with_uncertainty() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    depth_stability = specs["nfcorpus_vertex_repeated_semantic_depth_stability"]
    depth_selection_stability = specs["nfcorpus_vertex_repeated_semantic_depth_selection_stability"]

    assert depth_stability.category == "selection_check"
    assert depth_stability.path.as_posix() == (
        "outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_stability.csv"
    )
    assert "confidence intervals" in depth_stability.role
    assert depth_selection_stability.path.as_posix() == (
        "outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_selection_stability.csv"
    )
    assert "Wilson" in depth_selection_stability.role


def test_final_project_artifact_specs_protocol_command_uses_depth_stability() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    protocol = specs["nfcorpus_vertex_protocol_summary"]

    assert "--depth-stability-csv" in protocol.producer_command
    assert "nfcorpus_vertex_repeated_semantic_depth_30x30x3_stability.csv" in protocol.producer_command


def test_final_project_artifact_specs_include_evidence_consistency() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    consistency = specs["final_evidence_consistency"]

    assert consistency.category == "consistency_check"
    assert consistency.path.as_posix() == "outputs/results/final_evidence_consistency.csv"
    assert "FINAL_REPORT" in consistency.role


def test_final_project_artifact_specs_include_confidence_gate_smoke() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    confidence_gate = specs["nfcorpus_confidence_gate_smoke"]

    assert confidence_gate.category == "diagnostic"
    assert confidence_gate.path.as_posix() == "outputs/confidence_gate_smoke/results/nfcorpus_retrieval_policy_summary.csv"
    assert "predicted action-score margin" in confidence_gate.role
    assert "--confidence-gate-margin" in confidence_gate.producer_command


def test_final_project_artifact_specs_include_confidence_gate_sweep() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    sweep = specs["nfcorpus_confidence_gate_sweep"]

    assert sweep.category == "diagnostic"
    assert sweep.path.as_posix() == "outputs/results/nfcorpus_confidence_gate_smoke_sweep.csv"
    assert "margin threshold sweep" in sweep.role
    assert "run_confidence_gate_sweep.py" in sweep.producer_command


def test_final_project_artifact_specs_include_main_confidence_gate_sweeps() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    nfcorpus = specs["nfcorpus_main_confidence_gate_sweep"]
    scifact = specs["scifact_main_confidence_gate_sweep"]

    assert nfcorpus.path.as_posix() == "outputs/results/nfcorpus_confidence_gate_sweep.csv"
    assert scifact.path.as_posix() == "outputs/results/scifact_confidence_gate_sweep.csv"
    assert "full-corpus" in nfcorpus.role
    assert "full-corpus" in scifact.role


def test_final_project_artifact_specs_include_bandit_baselines_and_budget_curves() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    scifact_linucb = specs["scifact_linucb_baseline_summary"]
    nfcorpus_linucb = specs["nfcorpus_linucb_baseline_summary"]
    scifact_history = specs["scifact_linucb_baseline_history"]
    nfcorpus_history = specs["nfcorpus_linucb_baseline_history"]
    scifact_budget = specs["scifact_budget_curve"]
    nfcorpus_budget = specs["nfcorpus_budget_curve"]

    assert scifact_linucb.category == "bandit_baseline"
    assert nfcorpus_linucb.category == "bandit_baseline"
    assert scifact_linucb.path.as_posix() == "outputs/results/scifact_linucb_baseline_summary.csv"
    assert nfcorpus_linucb.path.as_posix() == "outputs/results/nfcorpus_linucb_baseline_summary.csv"
    assert "LinUCB" in scifact_linucb.role
    assert "selected-action feedback" in nfcorpus_linucb.role
    assert "run_bandit_baselines.py" in scifact_linucb.producer_command
    assert scifact_history.category == "bandit_diagnostic"
    assert nfcorpus_history.category == "bandit_diagnostic"
    assert scifact_history.path.as_posix() == "outputs/results/scifact_linucb_baseline_history.csv"
    assert nfcorpus_history.path.as_posix() == "outputs/results/nfcorpus_linucb_baseline_history.csv"
    assert "regret" in scifact_history.role

    assert scifact_budget.category == "budget_curve"
    assert nfcorpus_budget.category == "budget_curve"
    assert scifact_budget.path.as_posix() == "outputs/results/scifact_budget_curve.csv"
    assert nfcorpus_budget.path.as_posix() == "outputs/results/nfcorpus_budget_curve.csv"
    assert "call budget" in scifact_budget.role
    assert "run_budget_curve.py" in nfcorpus_budget.producer_command


def test_final_project_artifact_specs_include_ope_diagnostics() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    scifact = specs["scifact_ope_diagnostics"]
    nfcorpus = specs["nfcorpus_ope_diagnostics"]
    scifact_stability = specs["scifact_ope_stability"]
    nfcorpus_stability = specs["nfcorpus_ope_stability"]

    assert scifact.category == "off_policy_evaluation"
    assert nfcorpus.category == "off_policy_evaluation"
    assert scifact.path.as_posix() == "outputs/results/scifact_ope_diagnostics.csv"
    assert nfcorpus.path.as_posix() == "outputs/results/nfcorpus_ope_diagnostics.csv"
    assert "doubly robust" in scifact.role
    assert "run_ope_diagnostics.py" in nfcorpus.producer_command
    assert scifact_stability.category == "off_policy_evaluation"
    assert nfcorpus_stability.category == "off_policy_evaluation"
    assert scifact_stability.path.as_posix() == "outputs/results/scifact_ope_stability.csv"
    assert nfcorpus_stability.path.as_posix() == "outputs/results/nfcorpus_ope_stability.csv"
    assert "repeated-seed" in scifact_stability.role
    assert "run_ope_stability.py" in nfcorpus_stability.producer_command


def test_final_project_artifact_specs_include_constrained_policy_sweeps() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    scifact = specs["scifact_constrained_policy_sweep"]
    nfcorpus = specs["nfcorpus_constrained_policy_sweep"]
    scifact_bootstrap = specs["scifact_constrained_policy_bootstrap"]
    nfcorpus_bootstrap = specs["nfcorpus_constrained_policy_bootstrap"]

    assert scifact.category == "constrained_bandit"
    assert nfcorpus.category == "constrained_bandit"
    assert scifact.path.as_posix() == "outputs/results/scifact_constrained_policy_sweep.csv"
    assert nfcorpus.path.as_posix() == "outputs/results/nfcorpus_constrained_policy_sweep.csv"
    assert "Lagrangian" in scifact.role
    assert "run_constrained_policy_sweep.py" in nfcorpus.producer_command
    assert scifact_bootstrap.category == "constrained_bandit"
    assert nfcorpus_bootstrap.category == "constrained_bandit"
    assert scifact_bootstrap.path.as_posix() == "outputs/results/scifact_constrained_policy_bootstrap.csv"
    assert nfcorpus_bootstrap.path.as_posix() == "outputs/results/nfcorpus_constrained_policy_bootstrap.csv"
    assert "confidence intervals" in scifact_bootstrap.role
    assert "run_constrained_policy_bootstrap.py" in nfcorpus_bootstrap.producer_command


def test_final_project_artifact_specs_include_final_claims_matrix() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    claims = specs["final_claims_matrix"]

    assert claims.category == "defense_artifact"
    assert claims.path.as_posix() == "outputs/results/final_claims_matrix.csv"
    assert "claim" in claims.role
    assert "run_final_claims_matrix.py" in claims.producer_command


def test_final_project_artifact_specs_include_presentation_outline() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    outline = specs["final_presentation_outline"]

    assert outline.category == "document"
    assert outline.path.as_posix() == "FINAL_PRESENTATION_OUTLINE.md"
    assert "defense" in outline.role


def test_final_project_artifact_specs_include_markdown_defense_package() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    expected = {
        "final_slides_markdown": "FINAL_SLIDES.md",
        "final_defense_qa": "FINAL_DEFENSE_QA.md",
        "final_results_summary": "FINAL_RESULTS_SUMMARY.md",
    }

    for artifact_id, path in expected.items():
        spec = specs[artifact_id]
        assert spec.category == "document"
        assert spec.path.as_posix() == path
        assert "Markdown" in spec.role


def test_final_project_artifact_specs_include_final_paper_assets() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    expected = {
        "final_main_results_table": "outputs/results/final_main_results_table.csv",
        "final_main_results_latex": "outputs/results/final_main_results_table.tex",
        "final_reward_delta_ci_figure": "outputs/figures/final_reward_delta_ci.png",
        "final_cost_reward_frontier_figure": "outputs/figures/final_cost_reward_frontier.png",
        "final_ope_estimator_error_figure": "outputs/figures/final_ope_estimator_error.png",
        "final_linucb_comparison_figure": "outputs/figures/final_linucb_comparison.png",
    }

    for artifact_id, path in expected.items():
        spec = specs[artifact_id]
        assert spec.path.as_posix() == path
        assert spec.category == "paper_asset"
        assert "run_final_paper_assets.py" in spec.producer_command


def test_final_project_artifact_specs_include_markdown_consistency_check() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    consistency = specs["final_markdown_consistency"]

    assert consistency.category == "consistency_check"
    assert consistency.path.as_posix() == "outputs/results/final_markdown_consistency.csv"
    assert "Markdown" in consistency.role
    assert "run_markdown_consistency.py" in consistency.producer_command


def test_final_project_artifact_specs_include_complexity_and_transfer_outputs() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    assert specs["scifact_complexity_buckets"].path.as_posix() == "outputs/results/scifact_complexity_buckets.csv"
    assert specs["nfcorpus_complexity_buckets"].path.as_posix() == "outputs/results/nfcorpus_complexity_buckets.csv"
    assert specs["cross_dataset_transfer_summary"].path.as_posix() == "outputs/results/cross_dataset_transfer_summary.csv"
    assert "difficulty" in specs["scifact_complexity_buckets"].role
    assert "transfer" in specs["cross_dataset_transfer_summary"].role
