from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.experiment_dashboard import build_experiment_dashboard


def test_experiment_dashboard_classifies_smoke_manifest(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "outputs" / "codex_smoke"
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "smoke_manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "seed": 42,
                "num_examples": 12,
                "uses_raw_data": False,
                "uses_external_api": False,
                "uses_model_download": False,
            }
        ),
        encoding="utf-8",
    )

    metadata = build_experiment_dashboard([tmp_path / "outputs"], tmp_path / "dashboard.csv")

    dashboard = pd.read_csv(metadata["outputs"]["csv"])
    row = dashboard.iloc[0]
    assert row["evidence_level"] == "smoke_synthetic"
    assert bool(row["uses_raw_data"]) is False
    assert bool(row["claim_allowed"]) is False
    assert row["num_test_examples"] == 12


def test_experiment_dashboard_allows_final_scifact_benchmark_claim_boundary(tmp_path: Path) -> None:
    results_dir = tmp_path / "outputs" / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method": "Selective retrieval policy",
                "reward": 1.09,
                "recall_at_5": 0.77,
                "mrr": 0.66,
                "retrieval_calls": 1.4,
            }
        ]
    ).to_csv(results_dir / "scifact_retrieval_policy_summary.csv", index=False)

    metadata = build_experiment_dashboard([results_dir], tmp_path / "dashboard.csv")

    dashboard = pd.read_csv(metadata["outputs"]["csv"])
    row = dashboard.iloc[0]
    assert row["dataset"] == "scifact"
    assert row["experiment_type"] == "retrieval_policy_summary"
    assert row["evidence_level"] == "full_benchmark"
    assert bool(row["uses_raw_data"]) is True
    assert bool(row["claim_allowed"]) is True


def test_experiment_dashboard_marks_api_preflight_as_non_claim_evidence(tmp_path: Path) -> None:
    api_dir = tmp_path / "outputs" / "codex_api_preflight"
    api_dir.mkdir(parents=True)
    (api_dir / "api_preflight_summary.json").write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "provider": "gemini",
                        "actual_calls_or_texts": 0,
                        "result": "dry_run_no_api_call",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    metadata = build_experiment_dashboard([api_dir], tmp_path / "dashboard.csv")

    row = pd.read_csv(metadata["outputs"]["csv"]).iloc[0]
    assert row["evidence_level"] == "api_preflight"
    assert bool(row["uses_external_api"]) is False
    assert bool(row["claim_allowed"]) is False


def test_experiment_dashboard_preserves_tiny_realdata_counts(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "outputs" / "codex_realdata_smoke" / "scifact_fake" / "results"
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "scifact_retrieval_policy_metadata.json").write_text(
        json.dumps(
            {
                "dataset_key": "scifact",
                "seed": 42,
                "train_examples": 30,
                "test_examples": 30,
                "policy_model": "ridge",
                "feature_set": "full",
                "semantic_features": "none",
            }
        ),
        encoding="utf-8",
    )

    metadata = build_experiment_dashboard([tmp_path / "outputs"], tmp_path / "dashboard.csv")

    row = pd.read_csv(metadata["outputs"]["csv"]).iloc[0]
    assert row["dataset"] == "scifact"
    assert row["evidence_level"] == "tiny_realdata"
    assert row["num_train_examples"] == 30
    assert row["num_test_examples"] == 30
    assert row["policy_model"] == "ridge"
    assert bool(row["claim_allowed"]) is False


def test_experiment_dashboard_writes_markdown_summary(tmp_path: Path) -> None:
    results_dir = tmp_path / "outputs" / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame([{"method": "Selective retrieval policy", "reward": 0.4}]).to_csv(
        results_dir / "nfcorpus_retrieval_policy_summary.csv",
        index=False,
    )

    output_md = tmp_path / "dashboard.md"
    metadata = build_experiment_dashboard([results_dir], tmp_path / "dashboard.csv", output_md=output_md)

    assert Path(metadata["outputs"]["markdown"]) == output_md
    text = output_md.read_text(encoding="utf-8")
    assert "Experiment Dashboard" in text
    assert "nfcorpus" in text
