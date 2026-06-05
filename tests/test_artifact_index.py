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


def test_final_project_artifact_specs_include_bandit_replay_diagnostics() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    scifact_summary = specs["scifact_bandit_replay_summary"]
    nfcorpus_history = specs["nfcorpus_bandit_replay_history"]
    scifact_figure = specs["scifact_bandit_replay_regret_figure"]

    assert scifact_summary.category == "bandit_diagnostic"
    assert scifact_summary.path.as_posix() == "outputs/results/scifact_bandit_replay_summary.csv"
    assert "selected-action" in scifact_summary.role
    assert "--detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv" in scifact_summary.producer_command

    assert nfcorpus_history.category == "bandit_diagnostic"
    assert nfcorpus_history.path.as_posix() == "outputs/results/nfcorpus_bandit_replay_history.csv"
    assert "cumulative regret" in nfcorpus_history.role

    assert scifact_figure.category == "paper_asset"
    assert scifact_figure.path.as_posix() == "outputs/figures/scifact_bandit_replay_regret.png"


def test_final_project_artifact_specs_include_multistep_fqi_extension() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    summary = specs["hotpot_multistep_fqi_summary"]
    detailed = specs["hotpot_multistep_fqi_detailed"]
    diagnostics = specs["hotpot_fqi_diagnostics_summary"]
    traces = specs["hotpot_fqi_trace_distribution"]
    action_figure = specs["hotpot_multistep_action_traces"]

    assert summary.category == "rl_extension"
    assert summary.path.as_posix() == "outputs/results/multistep_summary.csv"
    assert "two-step FQI" in summary.role
    assert "run_multistep_hotpot.py" in summary.producer_command

    assert detailed.category == "rl_extension"
    assert detailed.path.as_posix() == "outputs/results/multistep_detailed.csv"
    assert "trace" in detailed.role

    assert diagnostics.category == "rl_extension"
    assert diagnostics.path.as_posix() == "outputs/results/hotpot_fqi_diagnostics_summary.csv"
    assert "train-best fixed trace" in diagnostics.role
    assert "run_fqi_diagnostics.py" in diagnostics.producer_command

    assert traces.category == "rl_extension"
    assert traces.path.as_posix() == "outputs/results/hotpot_fqi_trace_distribution.csv"
    assert "action-trace distribution" in traces.role

    assert action_figure.category == "paper_asset"
    assert action_figure.path.as_posix() == "outputs/figures/multistep_action_traces.png"


def test_final_project_artifact_specs_include_reader_and_gemini_pilots() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    reader = specs["hotpot_reader_realdata_summary"]
    hotpot_reader_200 = specs["hotpot_reader_realdata_200_summary"]
    nq_reader = specs["nq_reader_realdata_summary"]
    gemini = specs["hotpot_gemini_pilot_summary"]
    repeated_gemini = specs["hotpot_gemini_repeated_pilot_summary"]

    assert reader.category == "reader_smoke"
    assert reader.path.as_posix() == "outputs/results/hotpot_reader_realdata_summary.csv"
    assert "not final QA benchmark" in reader.role
    assert "run_reader_comparison.py" in reader.producer_command
    assert hotpot_reader_200.category == "reader_smoke"
    assert hotpot_reader_200.path.as_posix() == "outputs/results/hotpot_reader_realdata_200_summary.csv"
    assert "answer-type" in hotpot_reader_200.role
    assert "--num-examples 200" in hotpot_reader_200.producer_command
    assert nq_reader.category == "reader_smoke"
    assert nq_reader.path.as_posix() == "outputs/results/nq_reader_realdata_summary.csv"
    assert "Natural Questions" in nq_reader.role
    assert "--dataset nq" in nq_reader.producer_command

    assert gemini.category == "api_pilot"
    assert gemini.path.as_posix() == "outputs/results/hotpot_gemini_pilot_summary.csv"
    assert "8 new Gemini calls" in gemini.role
    assert "--max-new-calls 8" in gemini.producer_command
    assert repeated_gemini.category == "api_pilot"
    assert repeated_gemini.path.as_posix() == "outputs/results/hotpot_gemini_repeated_pilot_summary.csv"
    assert "0 new calls" in repeated_gemini.role
    assert "run_repeated_gemini_baseline.py" in repeated_gemini.producer_command


def test_final_project_artifact_specs_include_vertex_tiny_pilot() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    diagnostics = specs["nfcorpus_vertex_repeated_10x10_diagnostics"]
    stability = specs["nfcorpus_vertex_repeated_10x10_stability"]

    assert diagnostics.category == "selection_check"
    assert diagnostics.path.as_posix() == "outputs/results/nfcorpus_vertex_repeated_10x10_diagnostics.csv"
    assert "208 new embedding texts" in diagnostics.role
    assert "--semantic-max-new-texts 90" in diagnostics.producer_command
    assert stability.category == "selection_check"
    assert stability.path.as_posix() == "outputs/results/nfcorpus_vertex_repeated_10x10_stability.csv"
    assert "guardrail" in stability.role


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


def test_final_project_artifact_specs_include_experiment_dashboard_and_poster_audit() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    dashboard = specs["experiment_dashboard"]
    poster_audit = specs["poster_claim_audit"]

    assert dashboard.category == "evidence_dashboard"
    assert dashboard.path.as_posix() == "outputs/results/experiment_dashboard.csv"
    assert "evidence level" in dashboard.role
    assert "run_experiment_dashboard.py" in dashboard.producer_command

    assert poster_audit.category == "document"
    assert poster_audit.path.as_posix() == "docs/POSTER_CLAIM_AUDIT.md"
    assert "poster" in poster_audit.role.lower()


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
