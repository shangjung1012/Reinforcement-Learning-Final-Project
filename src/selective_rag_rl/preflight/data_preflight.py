from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

FileKind = Literal["json", "jsonl", "tsv", "parquet"]


@dataclass(frozen=True)
class DataPathSpec:
    dataset: str
    split: str
    path: str
    kind: FileKind
    required: bool = True


EXPECTED_DATA_PATHS = [
    DataPathSpec("hotpotqa", "dev_distractor", "data/raw/HotpotQA/hotpot_dev_distractor_v1.json", "json"),
    DataPathSpec("hotpotqa", "dev_fullwiki", "data/raw/HotpotQA/hotpot_dev_fullwiki_v1.json", "json", required=False),
    DataPathSpec("hotpotqa", "train", "data/raw/HotpotQA/hotpot_train_v1.1.json", "json", required=False),
    DataPathSpec("natural_questions", "validation", "data/raw/natural-questions/default/validation-00000-of-00007.parquet", "parquet"),
    DataPathSpec("scifact", "corpus", "data/raw/scifact/corpus.jsonl", "jsonl"),
    DataPathSpec("scifact", "queries", "data/raw/scifact/queries.jsonl", "jsonl"),
    DataPathSpec("scifact", "qrels_train", "data/raw/scifact/qrels/train.tsv", "tsv"),
    DataPathSpec("scifact", "qrels_test", "data/raw/scifact/qrels/test.tsv", "tsv"),
    DataPathSpec("nfcorpus", "corpus", "data/raw/nfcorpus/corpus.jsonl", "jsonl"),
    DataPathSpec("nfcorpus", "queries", "data/raw/nfcorpus/queries.jsonl", "jsonl"),
    DataPathSpec("nfcorpus", "qrels_train", "data/raw/nfcorpus/qrels/train.tsv", "tsv"),
    DataPathSpec("nfcorpus", "qrels_dev", "data/raw/nfcorpus/qrels/dev.tsv", "tsv"),
    DataPathSpec("nfcorpus", "qrels_test", "data/raw/nfcorpus/qrels/test.tsv", "tsv"),
]


def run_data_preflight(project_root: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_check_path(project_root, spec) for spec in EXPECTED_DATA_PATHS]
    required_rows = [row for row in rows if row["required"]]
    dataset_status = _dataset_status(rows)
    summary = {
        "all_required_available": all(row["exists"] for row in required_rows),
        "required_missing_count": sum(1 for row in required_rows if not row["exists"]),
        "datasets": dataset_status,
        "outputs": {
            "summary_json": str(output_dir / "data_preflight.json"),
            "summary_csv": str(output_dir / "data_preflight.csv"),
        },
        "paths": rows,
    }
    (output_dir / "data_preflight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "data_preflight.csv", index=False)
    return summary


def _check_path(project_root: Path, spec: DataPathSpec) -> dict[str, object]:
    full_path = project_root / spec.path
    row: dict[str, object] = {
        "dataset": spec.dataset,
        "split": spec.split,
        "path": spec.path,
        "kind": spec.kind,
        "required": spec.required,
        "exists": full_path.exists(),
        "row_count": None,
        "status": "missing",
        "error_type": "",
    }
    if not full_path.exists():
        return row
    try:
        row["row_count"] = _count_rows(full_path, spec.kind)
        row["status"] = "available"
    except Exception as exc:  # pragma: no cover - defensive for malformed local data
        row["status"] = "error"
        row["error_type"] = exc.__class__.__name__
    return row


def _count_rows(path: Path, kind: FileKind) -> int:
    if kind == "jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if kind == "tsv":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return 0
        first = lines[0].lower()
        has_header = "query" in first and ("corpus" in first or "doc" in first)
        return max(len(lines) - int(has_header), 0)
    if kind == "json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return len(value) if isinstance(value, list) else 1
    if kind == "parquet":
        return int(len(pd.read_parquet(path)))
    raise ValueError(f"Unsupported file kind: {kind}")


def _dataset_status(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    result = []
    for dataset in datasets:
        part = [row for row in rows if row["dataset"] == dataset and row["required"]]
        result.append(
            {
                "dataset": dataset,
                "required_paths": len(part),
                "available_required_paths": sum(1 for row in part if row["exists"]),
                "missing_required_paths": sum(1 for row in part if not row["exists"]),
                "status": "available" if part and all(row["exists"] for row in part) else "blocked_missing_data",
            }
        )
    return result
