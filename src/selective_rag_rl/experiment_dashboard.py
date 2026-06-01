from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DASHBOARD_COLUMNS = [
    "artifact_path",
    "dataset",
    "experiment_type",
    "evidence_level",
    "uses_raw_data",
    "uses_external_api",
    "uses_model_download",
    "num_train_examples",
    "num_test_examples",
    "seed",
    "policy_model",
    "feature_set",
    "claim_allowed",
    "notes",
]


@dataclass(frozen=True)
class ExperimentArtifact:
    artifact_path: str
    dataset: str
    experiment_type: str
    evidence_level: str
    uses_raw_data: bool
    uses_external_api: bool
    uses_model_download: bool
    num_train_examples: int | None
    num_test_examples: int | None
    seed: int | None
    policy_model: str
    feature_set: str
    claim_allowed: bool
    notes: str


def build_experiment_dashboard(
    input_paths: list[Path],
    output_csv: Path,
    output_md: Path | None = None,
) -> dict[str, object]:
    artifacts = [_summarize_artifact(path) for path in _discover_artifacts(input_paths)]
    artifacts.sort(key=lambda item: (item.evidence_level, item.dataset, item.artifact_path))
    rows = [asdict(artifact) for artifact in artifacts]
    dashboard = pd.DataFrame(rows, columns=DASHBOARD_COLUMNS)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dashboard.to_csv(output_csv, index=False)
    outputs: dict[str, str] = {"csv": str(output_csv)}
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown_dashboard(dashboard), encoding="utf-8")
        outputs["markdown"] = str(output_md)

    return {
        "artifact_count": int(len(dashboard)),
        "evidence_levels": dashboard["evidence_level"].value_counts().to_dict() if len(dashboard) else {},
        "claim_allowed_count": int(dashboard["claim_allowed"].sum()) if len(dashboard) else 0,
        "outputs": outputs,
    }


def _discover_artifacts(input_paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for input_path in input_paths:
        path = Path(input_path)
        if not path.exists():
            continue
        if path.is_file():
            candidates = [path]
        else:
            candidates = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        for candidate in candidates:
            if candidate.suffix.lower() not in {".csv", ".json"}:
                continue
            if _is_raw_data_path(candidate):
                continue
            discovered.append(candidate)
    return sorted(set(discovered))


def _summarize_artifact(path: Path) -> ExperimentArtifact:
    payload = _read_payload(path)
    dataset = _infer_dataset(path, payload)
    experiment_type = _infer_experiment_type(path, payload)
    evidence_level = _infer_evidence_level(path, payload, dataset, experiment_type)
    uses_external_api = _uses_external_api(path, payload, evidence_level)
    uses_raw_data = _uses_raw_data(path, payload, evidence_level)
    uses_model_download = _bool_value(payload.get("uses_model_download"), False)
    claim_allowed = evidence_level in {"full_benchmark", "final_claim"}
    return ExperimentArtifact(
        artifact_path=str(path),
        dataset=dataset,
        experiment_type=experiment_type,
        evidence_level=evidence_level,
        uses_raw_data=uses_raw_data,
        uses_external_api=uses_external_api,
        uses_model_download=uses_model_download,
        num_train_examples=_int_value(
            payload.get("train_examples"),
            payload.get("num_train_examples"),
        ),
        num_test_examples=_int_value(
            payload.get("test_examples"),
            payload.get("num_test_examples"),
            payload.get("num_examples"),
        ),
        seed=_int_value(payload.get("seed")),
        policy_model=str(payload.get("policy_model") or ""),
        feature_set=str(payload.get("feature_set") or ""),
        claim_allowed=claim_allowed,
        notes=_notes(evidence_level, experiment_type),
    )


def _read_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    try:
        frame = pd.read_csv(path, nrows=1)
    except Exception:
        return {}
    if frame.empty:
        return {}
    return {str(key): value for key, value in frame.iloc[0].to_dict().items() if not pd.isna(value)}


def _infer_dataset(path: Path, payload: dict[str, Any]) -> str:
    for key in ("dataset_key", "dataset"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return _normalize_dataset(value)
    lower = str(path).replace("\\", "/").lower()
    for dataset in ("scifact", "nfcorpus", "hotpot", "hotpotqa", "nq", "natural_questions", "toy"):
        if dataset in lower:
            return _normalize_dataset(dataset)
    return "unknown"


def _infer_experiment_type(path: Path, payload: dict[str, Any]) -> str:
    name = path.name.lower()
    if name == "smoke_manifest.json":
        return "final_smoke"
    if name == "api_preflight_summary.json":
        return "api_preflight"
    if "reader_eval_summary" in name:
        return "reader_eval"
    if "retrieval_policy_metadata" in name:
        return "retrieval_policy_metadata"
    if "retrieval_policy_summary" in name:
        return "retrieval_policy_summary"
    if "validation_guardrail" in name:
        return "validation_guardrail"
    if "cost_frontier" in name:
        return "cost_frontier"
    if "final_claims" in name:
        return "final_claims"
    if "artifact_index" in name:
        return "artifact_index"
    if "method" in payload and "reward" in payload:
        return "summary_csv"
    return path.stem


def _infer_evidence_level(
    path: Path,
    payload: dict[str, Any],
    dataset: str,
    experiment_type: str,
) -> str:
    lower = str(path).replace("\\", "/").lower()
    if "codex_api_preflight" in lower or experiment_type == "api_preflight":
        return "api_preflight"
    if "codex_gemini" in lower or "codex_vertex" in lower:
        return "api_pilot"
    if "codex_realdata_smoke" in lower:
        return "tiny_realdata"
    if "codex_reader_smoke" in lower or (experiment_type == "reader_eval" and dataset == "toy"):
        return "smoke_toy_reader"
    if "codex_smoke" in lower or experiment_type == "final_smoke":
        return "smoke_synthetic"
    if experiment_type in {"final_claims", "artifact_index"}:
        return "final_claim"
    if (
        "outputs/results" in lower
        and experiment_type == "retrieval_policy_summary"
        and dataset in {"scifact", "nfcorpus"}
    ):
        return "full_benchmark"
    return "smoke_synthetic" if "codex" in lower else "analysis"


def _uses_external_api(path: Path, payload: dict[str, Any], evidence_level: str) -> bool:
    if "uses_external_api" in payload:
        return _bool_value(payload.get("uses_external_api"), False)
    providers = payload.get("providers")
    if isinstance(providers, list):
        return any(_int_value(provider.get("actual_calls_or_texts")) or 0 for provider in providers if isinstance(provider, dict))
    if evidence_level == "api_pilot":
        return True
    return False


def _uses_raw_data(path: Path, payload: dict[str, Any], evidence_level: str) -> bool:
    if "uses_raw_data" in payload:
        return _bool_value(payload.get("uses_raw_data"), False)
    return evidence_level in {"tiny_realdata", "full_benchmark", "final_claim"}


def _notes(evidence_level: str, experiment_type: str) -> str:
    if evidence_level in {"smoke_synthetic", "smoke_toy_reader"}:
        return "Code-path smoke; not benchmark evidence."
    if evidence_level == "tiny_realdata":
        return "Small real-data run; useful for integration only."
    if evidence_level == "api_preflight":
        return "Configuration/API readiness check; not model-quality evidence."
    if evidence_level == "api_pilot":
        return "Bounded API pilot; do not promote without repeated validation."
    if evidence_level == "full_benchmark":
        return "Can support retrieval-stage claims if command and split are documented."
    if evidence_level == "final_claim":
        return "Claim/evidence index artifact."
    return f"Analysis artifact: {experiment_type}."


def _markdown_dashboard(dashboard: pd.DataFrame) -> str:
    lines = [
        "# Experiment Dashboard",
        "",
        "This dashboard classifies local artifacts by evidence level. It separates smoke and pilot evidence from artifacts that can support final claims.",
        "",
    ]
    if dashboard.empty:
        lines.append("No artifacts found.")
        return "\n".join(lines) + "\n"
    counts = dashboard["evidence_level"].value_counts().sort_index()
    lines.extend(["## Evidence Levels", ""])
    for level, count in counts.items():
        lines.append(f"- `{level}`: {int(count)}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Dataset | Evidence | Type | Claim Allowed | Path |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in dashboard.to_dict(orient="records"):
        lines.append(
            "| {dataset} | {evidence_level} | {experiment_type} | {claim_allowed} | `{artifact_path}` |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def _normalize_dataset(value: str) -> str:
    normalized = value.lower().replace(" ", "_")
    if normalized == "hotpotqa":
        return "hotpot"
    if normalized == "natural_questions":
        return "nq"
    return normalized


def _is_raw_data_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "data" in parts and "raw" in parts


def _int_value(*values: Any) -> int | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _bool_value(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
