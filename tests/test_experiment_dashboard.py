from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.reports.artifact_index import ArtifactSpec
from selective_rag_rl.reports.experiment_dashboard import (
    DASHBOARD_COLUMNS,
    build_experiment_dashboard,
    write_experiment_dashboard_markdown,
)


def test_experiment_dashboard_separates_final_benchmark_from_smoke_and_api_pilot(tmp_path: Path) -> None:
    specs = [
        ArtifactSpec(
            artifact_id="scifact_policy_summary",
            category="main_result",
            path=Path("outputs/results/scifact_retrieval_policy_summary.csv"),
            role="BEIR SciFact full-corpus retrieval-action policy summary.",
            producer_command=(
                "uv run python scripts/run_retrieval_policy_scifact.py "
                "--num-train-examples 600 --num-test-examples 300 --seed 42 "
                "--full-corpus --policy-model auto"
            ),
        ),
        ArtifactSpec(
            artifact_id="nfcorpus_confidence_gate_smoke",
            category="diagnostic",
            path=Path("outputs/confidence_gate_smoke/results/nfcorpus_retrieval_policy_summary.csv"),
            role="No-API NFCorpus smoke test for confidence-gated policy fallback.",
            producer_command=(
                "uv run python scripts/run_retrieval_policy_nfcorpus.py "
                "--num-train-examples 12 --num-test-examples 8 --pool-size 20 "
                "--embedder fake --policy-model ridge"
            ),
        ),
        ArtifactSpec(
            artifact_id="nfcorpus_vertex_repeated_selection_diagnostics",
            category="selection_check",
            path=Path("outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_diagnostics.csv"),
            role="Cross-seed validation-selection guardrail diagnostics for Vertex semantic features.",
            producer_command=(
                "uv run python scripts/run_repeated_selection.py --dataset nfcorpus "
                "--seeds 41,42,43 --policy-models ridge,auto --feature-sets full,no_semantic "
                "--num-train-examples 30 --num-test-examples 30 --semantic-features vertex"
            ),
        ),
        ArtifactSpec(
            artifact_id="scifact_bootstrap",
            category="statistical_check",
            path=Path("outputs/results/scifact_bootstrap_diagnostics.csv"),
            role="Paired bootstrap intervals for SciFact policy-vs-baseline reward gaps.",
            producer_command=(
                "uv run python scripts/run_statistical_diagnostics.py --dataset scifact "
                "--detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv "
                "--output-csv outputs/results/scifact_bootstrap_diagnostics.csv"
            ),
        ),
        ArtifactSpec(
            artifact_id="readme",
            category="document",
            path=Path("README.md"),
            role="Reproduction commands, scope, and project inventory.",
            producer_command="manual documentation update",
        ),
        ArtifactSpec(
            artifact_id="hotpot_multistep_metrics_figure",
            category="paper_asset",
            path=Path("outputs/figures/multistep_metrics.png"),
            role="HotpotQA two-step FQI metric comparison figure.",
            producer_command="uv run python scripts/run_multistep_hotpot.py --num-examples 600 --seed 42",
        ),
        ArtifactSpec(
            artifact_id="hotpot_reader_realdata_summary",
            category="reader_smoke",
            path=Path("outputs/results/hotpot_reader_realdata_summary.csv"),
            role="Tiny HotpotQA real-data downstream reader comparison for lexical and span heuristic readers.",
            producer_command=(
                "uv run python scripts/run_reader_comparison.py --dataset hotpot "
                "--num-examples 50 --readers lexical,span"
            ),
        ),
    ]
    claims_csv = tmp_path / "outputs" / "results" / "final_claims_matrix.csv"
    claims_csv.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"claim_id": "scifact_policy_reward_gain", "evidence_artifact_id": "scifact_policy_summary"},
            {"claim_id": "scifact_bootstrap_gain", "evidence_artifact_id": "scifact_bootstrap"},
        ]
    ).to_csv(claims_csv, index=False)

    dashboard = build_experiment_dashboard(specs, root=tmp_path, claims_csv=claims_csv)
    rows = dashboard.set_index("artifact_id")

    assert list(dashboard.columns) == DASHBOARD_COLUMNS
    assert rows.loc["scifact_policy_summary", "evidence_level"] == "full_benchmark"
    assert bool(rows.loc["scifact_policy_summary", "claim_allowed"]) is True
    assert bool(rows.loc["scifact_policy_summary", "supports_final_claim"]) is True
    assert rows.loc["scifact_policy_summary", "num_train_examples"] == 600
    assert rows.loc["scifact_policy_summary", "num_test_examples"] == 300
    assert rows.loc["scifact_policy_summary", "seed"] == 42
    assert rows.loc["scifact_policy_summary", "policy_model"] == "auto"

    assert rows.loc["nfcorpus_confidence_gate_smoke", "evidence_level"] == "smoke_synthetic"
    assert bool(rows.loc["nfcorpus_confidence_gate_smoke", "claim_allowed"]) is False
    assert bool(rows.loc["nfcorpus_confidence_gate_smoke", "uses_external_api"]) is False

    assert rows.loc["nfcorpus_vertex_repeated_selection_diagnostics", "evidence_level"] == "api_pilot"
    assert bool(rows.loc["nfcorpus_vertex_repeated_selection_diagnostics", "claim_allowed"]) is False
    assert bool(rows.loc["nfcorpus_vertex_repeated_selection_diagnostics", "uses_external_api"]) is True
    assert rows.loc["nfcorpus_vertex_repeated_selection_diagnostics", "feature_set"] == "full,no_semantic"

    assert rows.loc["scifact_bootstrap", "evidence_level"] == "full_benchmark"
    assert bool(rows.loc["scifact_bootstrap", "claim_allowed"]) is True
    assert bool(rows.loc["scifact_bootstrap", "supports_final_claim"]) is True

    assert rows.loc["readme", "evidence_level"] == "final_claim"
    assert bool(rows.loc["readme", "claim_allowed"]) is True

    assert rows.loc["hotpot_multistep_metrics_figure", "evidence_level"] == "tiny_realdata"
    assert bool(rows.loc["hotpot_multistep_metrics_figure", "claim_allowed"]) is False

    assert rows.loc["hotpot_reader_realdata_summary", "evidence_level"] == "tiny_realdata"
    assert bool(rows.loc["hotpot_reader_realdata_summary", "claim_allowed"]) is False


def test_write_experiment_dashboard_markdown_summarizes_levels(tmp_path: Path) -> None:
    dashboard = pd.DataFrame(
        [
            {
                **{column: "" for column in DASHBOARD_COLUMNS},
                "artifact_id": "scifact_policy_summary",
                "artifact_path": "outputs/results/scifact_retrieval_policy_summary.csv",
                "dataset": "scifact",
                "experiment_type": "main_result",
                "evidence_level": "full_benchmark",
                "claim_allowed": True,
                "supports_final_claim": True,
                "notes": "final retrieval-stage benchmark",
            },
            {
                **{column: "" for column in DASHBOARD_COLUMNS},
                "artifact_id": "nfcorpus_vertex_pilot",
                "artifact_path": "outputs/results/nfcorpus_vertex.csv",
                "dataset": "nfcorpus",
                "experiment_type": "selection_check",
                "evidence_level": "api_pilot",
                "claim_allowed": False,
                "supports_final_claim": False,
                "notes": "pilot only",
            },
        ]
    )

    output_md = tmp_path / "docs" / "EXPERIMENT_DASHBOARD.md"
    write_experiment_dashboard_markdown(dashboard, output_md=output_md)

    text = output_md.read_text()
    assert "# Experiment Dashboard" in text
    assert "| full_benchmark | 1 |" in text
    assert "| api_pilot | 1 |" in text
    assert "pilot only" in text
