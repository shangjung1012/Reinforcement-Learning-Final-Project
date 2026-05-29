from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.experiment_dashboard import build_experiment_dashboard, write_dashboard


def test_experiment_dashboard_classifies_api_preflight_and_final_results(tmp_path: Path) -> None:
    results = tmp_path / "outputs/results"
    api = tmp_path / "outputs/codex_api_preflight"
    results.mkdir(parents=True)
    api.mkdir(parents=True)
    (results / "scifact_retrieval_policy_summary.csv").write_text("method,reward\nA,1.0\n", encoding="utf-8")
    (api / "api_preflight_summary.json").write_text(
        json.dumps({"providers": [{"provider": "gemini", "actual_calls_or_texts": 1}]}),
        encoding="utf-8",
    )

    rows = build_experiment_dashboard(tmp_path)
    by_path = {row["artifact_path"]: row for row in rows}

    assert by_path["outputs/results/scifact_retrieval_policy_summary.csv"]["evidence_level"] == "full_benchmark"
    assert by_path["outputs/codex_api_preflight/api_preflight_summary.json"]["evidence_level"] == "api_preflight"
    assert by_path["outputs/codex_api_preflight/api_preflight_summary.json"]["uses_external_api"] is True


def test_write_dashboard_outputs_csv_and_markdown(tmp_path: Path) -> None:
    results = tmp_path / "outputs/results"
    results.mkdir(parents=True)
    (results / "final_claims_matrix.csv").write_text("claim,evidence\nA,B\n", encoding="utf-8")

    output_csv = tmp_path / "outputs/results/experiment_dashboard.csv"
    output_md = tmp_path / "docs/EXPERIMENT_DASHBOARD.md"
    written = write_dashboard(tmp_path, output_csv=output_csv, output_md=output_md)

    assert output_csv.exists()
    assert output_md.exists()
    rows = pd.read_csv(output_csv)
    assert rows.loc[0, "evidence_level"] == "final_claim"
    assert "Experiment Dashboard" in output_md.read_text(encoding="utf-8")
