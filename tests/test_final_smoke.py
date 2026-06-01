from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd


def test_run_final_smoke_writes_raw_data_free_artifacts(tmp_path: Path) -> None:
    run_final_smoke = _load_run_final_smoke()
    output_dir = tmp_path / "smoke"

    manifest = run_final_smoke(
        output_dir=output_dir,
        num_examples=12,
        seed=7,
        pytest_mode="skip",
    )

    manifest_path = output_dir / "smoke_manifest.json"
    fixture_path = output_dir / "fixtures" / "synthetic_hotpot.json"
    summary_path = output_dir / "retrieval_policy_smoke" / "results" / "retrieval_policy_summary.csv"
    detailed_path = output_dir / "retrieval_policy_smoke" / "results" / "retrieval_policy_detailed.csv"
    ope_path = output_dir / "results" / "smoke_ope_diagnostics.csv"

    assert manifest_path.exists()
    assert fixture_path.exists()
    assert summary_path.exists()
    assert detailed_path.exists()
    assert ope_path.exists()

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    assert saved_manifest["status"] == "pass"
    assert saved_manifest["uses_raw_data"] is False
    assert saved_manifest["uses_external_api"] is False
    assert saved_manifest["pytest_mode"] == "skip"
    assert saved_manifest["outputs"]["synthetic_fixture"] == str(fixture_path)
    assert saved_manifest["outputs"]["ope_diagnostics_csv"] == str(ope_path)

    summary = pd.read_csv(summary_path)
    assert "Selective retrieval policy" in set(summary["method"])
    assert "Oracle retrieval action" in set(summary["method"])

    ope = pd.read_csv(ope_path)
    assert set(ope["estimator"]) == {"direct_method", "ips", "snips", "doubly_robust"}
    assert "Selective retrieval policy" in set(ope["target_method"])


def _load_run_final_smoke():
    script_path = Path("scripts/run_final_smoke.py")
    spec = importlib.util.spec_from_file_location("run_final_smoke", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_final_smoke
