from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

EVIDENCE_LEVELS = {
    "smoke_synthetic",
    "smoke_toy_reader",
    "tiny_realdata",
    "api_preflight",
    "api_pilot",
    "full_benchmark",
    "final_claim",
    "blocked_missing_data",
    "blocked_missing_credentials",
    "blocked_budget_exceeded",
}


def build_experiment_dashboard(project_root: Path) -> list[dict[str, object]]:
    roots = [project_root / "outputs" / "results", *sorted((project_root / "outputs").glob("codex_*"))]
    rows: list[dict[str, object]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted([*root.rglob("*.csv"), *root.rglob("*.json")]):
            rows.append(_artifact_row(project_root, path))
    return rows


def write_dashboard(project_root: Path, output_csv: Path, output_md: Path) -> dict[str, str]:
    rows = build_experiment_dashboard(project_root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_md.write_text(_markdown(rows), encoding="utf-8")
    return {"dashboard_csv": str(output_csv), "dashboard_md": str(output_md)}


def _artifact_row(project_root: Path, path: Path) -> dict[str, object]:
    rel = path.relative_to(project_root).as_posix()
    evidence_level = _evidence_level(rel)
    metadata = _json_metadata(path) if path.suffix == ".json" else {}
    return {
        "artifact_path": rel,
        "dataset": _dataset_from_path(rel, metadata),
        "experiment_type": _experiment_type(rel),
        "evidence_level": evidence_level,
        "uses_raw_data": evidence_level in {"full_benchmark", "final_claim", "tiny_realdata"},
        "uses_external_api": _uses_external_api(rel, metadata, evidence_level),
        "uses_model_download": False,
        "num_train_examples": _metadata_value(metadata, "train_examples", "num_train_examples", "train_size"),
        "num_test_examples": _metadata_value(metadata, "test_examples", "num_test_examples", "test_size"),
        "seed": _metadata_value(metadata, "seed"),
        "policy_model": _metadata_value(metadata, "policy_model", "selected_policy_model"),
        "feature_set": _metadata_value(metadata, "feature_set"),
        "claim_allowed": evidence_level in {"full_benchmark", "final_claim"},
        "notes": _notes(evidence_level),
    }


def _evidence_level(rel: str) -> str:
    lower = rel.lower()
    if "codex_api_preflight" in lower:
        return "api_preflight"
    if "codex_data_preflight" in lower:
        return "blocked_missing_data"
    if "codex_gemini_pilot" in lower or "codex_vertex_pilot" in lower:
        return "api_pilot"
    if "codex_reader" in lower:
        return "smoke_toy_reader"
    if "codex_smoke" in lower:
        return "smoke_synthetic"
    if "/final_" in lower or lower.startswith("outputs/results/final_"):
        return "final_claim"
    if lower.startswith("outputs/results/"):
        return "full_benchmark"
    return "smoke_synthetic"


def _dataset_from_path(rel: str, metadata: dict[str, Any]) -> str:
    dataset = metadata.get("dataset")
    if isinstance(dataset, str) and dataset:
        return dataset
    lower = rel.lower()
    for name in ["scifact", "nfcorpus", "hotpot", "natural_questions", "nq", "toy"]:
        if name in lower:
            return name
    return ""


def _experiment_type(rel: str) -> str:
    lower = rel.lower()
    if "guardrail" in lower:
        return "validation_guardrail"
    if "frontier" in lower or "budget" in lower:
        return "cost_frontier"
    if "api_preflight" in lower:
        return "api_preflight"
    if "data_preflight" in lower:
        return "data_preflight"
    if "reader_eval" in lower:
        return "reader_eval"
    if "gemini" in lower:
        return "gemini_rewrite"
    if "retrieval_policy" in lower:
        return "retrieval_policy"
    return "artifact"


def _uses_external_api(rel: str, metadata: dict[str, Any], evidence_level: str) -> bool:
    if evidence_level == "api_pilot":
        return True
    if evidence_level != "api_preflight":
        return False
    providers = metadata.get("providers")
    if not isinstance(providers, list):
        return False
    return any(int(provider.get("actual_calls_or_texts", 0)) > 0 for provider in providers if isinstance(provider, dict))


def _json_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _metadata_value(metadata: dict[str, Any], *keys: str) -> object:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return ""


def _notes(evidence_level: str) -> str:
    if evidence_level in {"smoke_synthetic", "smoke_toy_reader"}:
        return "smoke_only_not_benchmark"
    if evidence_level == "api_preflight":
        return "credential_or_sdk_check_only"
    if evidence_level == "api_pilot":
        return "api_pilot_not_final_claim"
    if evidence_level == "blocked_missing_data":
        return "blocked_until_raw_data_available"
    return ""


def _markdown(rows: list[dict[str, object]]) -> str:
    counts = pd.DataFrame(rows)["evidence_level"].value_counts().to_dict() if rows else {}
    lines = [
        "# Experiment Dashboard",
        "",
        "This dashboard classifies generated artifacts by evidence level so smoke",
        "and API pilots are not confused with final benchmark evidence.",
        "",
        "## Evidence Counts",
        "",
        "| Evidence level | Count |",
        "| --- | ---: |",
    ]
    for level in sorted(EVIDENCE_LEVELS):
        lines.append(f"| `{level}` | {int(counts.get(level, 0))} |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Only rows marked `full_benchmark` or `final_claim` may support final",
            "retrieval-stage claims. API pilot and smoke rows are integration evidence",
            "only.",
            "",
        ]
    )
    return "\n".join(lines)
