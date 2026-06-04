from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from selective_rag_rl.reports.artifact_index import ArtifactSpec, final_project_artifact_specs


DASHBOARD_COLUMNS = [
    "artifact_id",
    "artifact_path",
    "dataset",
    "experiment_type",
    "evidence_level",
    "exists",
    "uses_raw_data",
    "uses_external_api",
    "uses_model_download",
    "num_train_examples",
    "num_test_examples",
    "seed",
    "policy_model",
    "feature_set",
    "claim_allowed",
    "supports_final_claim",
    "notes",
]


def build_experiment_dashboard(
    specs: list[ArtifactSpec] | None = None,
    *,
    root: Path | None = None,
    claims_csv: Path | None = None,
) -> pd.DataFrame:
    root = (root or Path(".")).resolve()
    specs = specs if specs is not None else final_project_artifact_specs(Path("."))
    final_claim_artifacts = _final_claim_artifact_ids(claims_csv)
    rows = [
        _dashboard_row(spec, root=root, final_claim_artifacts=final_claim_artifacts)
        for spec in specs
    ]
    return pd.DataFrame(rows, columns=DASHBOARD_COLUMNS)


def export_experiment_dashboard(
    *,
    root: Path | None = None,
    output_csv: Path,
    output_md: Path | None = None,
    claims_csv: Path | None = None,
) -> tuple[Path, Path | None]:
    dashboard = build_experiment_dashboard(root=root, claims_csv=claims_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dashboard.to_csv(output_csv, index=False)
    if output_md is not None:
        write_experiment_dashboard_markdown(dashboard, output_md=output_md)
    return output_csv, output_md


def write_experiment_dashboard_markdown(frame: pd.DataFrame, *, output_md: Path) -> Path:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    level_counts = frame["evidence_level"].value_counts().sort_index().reset_index()
    level_counts.columns = ["evidence_level", "count"]
    rows = frame.sort_values(["evidence_level", "dataset", "artifact_id"])
    preview_columns = [
        "artifact_id",
        "dataset",
        "experiment_type",
        "evidence_level",
        "claim_allowed",
        "supports_final_claim",
        "notes",
    ]
    text = "\n".join(
        [
            "# Experiment Dashboard",
            "",
            "This dashboard separates final benchmark evidence from smoke checks, API pilots,",
            "and analysis-only artifacts. `claim_allowed=false` means the artifact can support",
            "engineering confidence or future work, but should not be used as a final project",
            "claim without stronger evidence.",
            "",
            "## Evidence Level Counts",
            "",
            _markdown_table(level_counts),
            "",
            "## Artifact Evidence Levels",
            "",
            _markdown_table(rows[preview_columns]),
            "",
        ]
    )
    output_md.write_text(text, encoding="utf-8")
    return output_md


def _dashboard_row(
    spec: ArtifactSpec,
    *,
    root: Path,
    final_claim_artifacts: set[str],
) -> dict[str, object]:
    text = " ".join(
        [
            spec.artifact_id,
            spec.category,
            spec.path.as_posix(),
            spec.role,
            spec.producer_command,
        ]
    ).lower()
    supports_final_claim = spec.artifact_id in final_claim_artifacts
    evidence_level = _evidence_level(spec, text=text, supports_final_claim=supports_final_claim)
    uses_external_api = _uses_external_api(text)
    claim_allowed = evidence_level in {"final_claim", "full_benchmark"} and not (
        uses_external_api and evidence_level != "final_claim"
    )
    path = spec.path if spec.path.is_absolute() else root / spec.path
    return {
        "artifact_id": spec.artifact_id,
        "artifact_path": spec.path.as_posix(),
        "dataset": _dataset(text),
        "experiment_type": spec.category,
        "evidence_level": evidence_level,
        "exists": path.exists(),
        "uses_raw_data": _uses_raw_data(text, evidence_level),
        "uses_external_api": uses_external_api,
        "uses_model_download": _uses_model_download(text),
        "num_train_examples": _int_flag(spec.producer_command, "--num-train-examples"),
        "num_test_examples": _int_flag(spec.producer_command, "--num-test-examples"),
        "seed": _int_flag(spec.producer_command, "--seed"),
        "policy_model": _string_flag(spec.producer_command, "--policy-model")
        or _string_flag(spec.producer_command, "--policy-models"),
        "feature_set": _string_flag(spec.producer_command, "--feature-set")
        or _string_flag(spec.producer_command, "--feature-sets")
        or _string_flag(spec.producer_command, "--semantic-features"),
        "claim_allowed": claim_allowed,
        "supports_final_claim": supports_final_claim,
        "notes": _notes(evidence_level, uses_external_api=uses_external_api),
    }


def _evidence_level(spec: ArtifactSpec, *, text: str, supports_final_claim: bool) -> str:
    if "api_preflight" in text or "run_api_preflight.py" in text:
        return "api_preflight"
    if spec.artifact_id.startswith("final_") or spec.category in {
        "document",
        "paper_asset",
        "defense_artifact",
        "consistency_check",
    }:
        return "final_claim"
    if "smoke" in text or "--embedder fake" in text:
        return "smoke_toy_reader" if "reader" in text else "smoke_synthetic"
    if _uses_external_api(text):
        return "api_pilot"
    if supports_final_claim:
        return "full_benchmark"
    if _is_supported_full_benchmark(text):
        return "full_benchmark"
    if _dataset(text) in {"scifact", "nfcorpus", "hotpot", "nq"}:
        return "tiny_realdata"
    return "blocked_missing_data" if "missing" in text else "smoke_synthetic"


def _is_supported_full_benchmark(text: str) -> bool:
    return (
        ("scifact" in text or "nfcorpus" in text)
        and "full-corpus" in text
        and "smoke" not in text
        and not _uses_external_api(text)
    )


def _uses_external_api(text: str) -> bool:
    api_terms = ["vertex", "gemini", "google_genai", "llm", "generated action"]
    return any(term in text for term in api_terms)


def _uses_raw_data(text: str, evidence_level: str) -> bool:
    if evidence_level in {"smoke_synthetic", "smoke_toy_reader", "api_preflight", "final_claim"}:
        return False
    return _dataset(text) in {"scifact", "nfcorpus", "hotpot", "nq"}


def _uses_model_download(text: str) -> bool:
    if "--embedder fake" in text:
        return False
    return "sentence-transformers" in text or "dense" in text and "fake" not in text


def _dataset(text: str) -> str:
    if "scifact" in text and "nfcorpus" in text:
        return "multiple"
    if "scifact" in text:
        return "scifact"
    if "nfcorpus" in text:
        return "nfcorpus"
    if "hotpot" in text:
        return "hotpot"
    if "natural questions" in text or "nq" in text:
        return "nq"
    if "cross-dataset" in text or "cross_dataset" in text:
        return "multiple"
    return "unknown"


def _notes(evidence_level: str, *, uses_external_api: bool) -> str:
    if evidence_level == "full_benchmark":
        return "final retrieval-stage benchmark evidence"
    if evidence_level == "final_claim":
        return "final report, figure, or claim artifact"
    if evidence_level == "api_pilot":
        return "API-backed pilot or semantic analysis only"
    if evidence_level == "api_preflight":
        return "credential/cache check only, not benchmark evidence"
    if evidence_level.startswith("smoke"):
        return "code-path smoke only, not benchmark evidence"
    if evidence_level == "tiny_realdata":
        return "small or non-final real-data run"
    if uses_external_api:
        return "external API artifact requires bounded cache-first review"
    return "analysis-only artifact"


def _final_claim_artifact_ids(claims_csv: Path | None) -> set[str]:
    if claims_csv is None or not claims_csv.exists():
        return set()
    frame = pd.read_csv(claims_csv)
    if "evidence_artifact_id" not in frame.columns:
        return set()
    return {str(value) for value in frame["evidence_artifact_id"].dropna()}


def _int_flag(command: str, flag: str) -> int | str:
    value = _string_flag(command, flag)
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        return ""


def _string_flag(command: str, flag: str) -> str:
    pattern = rf"{re.escape(flag)}\s+([^\s]+)"
    match = re.search(pattern, command)
    return match.group(1) if match else ""


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [_markdown_cell(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "/")
