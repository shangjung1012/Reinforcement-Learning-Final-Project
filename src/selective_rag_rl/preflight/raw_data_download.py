from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import pandas as pd


Fetcher = Callable[[str, Path], None]
HfFetcher = Callable[[str, str, str], Path]


@dataclass(frozen=True)
class RawDataDownloadSpec:
    dataset_key: str
    dataset: str
    split: str
    target_path: str
    url: str
    hf_repo_id: str = ""
    hf_filename: str = ""
    hf_repo_type: str = "dataset"
    hf_converter: str = ""


RAW_DATA_DOWNLOAD_SPECS = [
    RawDataDownloadSpec(
        dataset_key="hotpot-dev-distractor",
        dataset="hotpotqa",
        split="dev_distractor",
        target_path="data/raw/HotpotQA/hotpot_dev_distractor_v1.json",
        url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        hf_repo_id="hotpotqa/hotpot_qa",
        hf_filename="distractor/validation-00000-of-00001.parquet",
        hf_converter="hotpot_parquet_to_json",
    ),
    RawDataDownloadSpec(
        dataset_key="hotpot-dev-fullwiki",
        dataset="hotpotqa",
        split="dev_fullwiki",
        target_path="data/raw/HotpotQA/hotpot_dev_fullwiki_v1.json",
        url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
    ),
    RawDataDownloadSpec(
        dataset_key="hotpot-train",
        dataset="hotpotqa",
        split="train",
        target_path="data/raw/HotpotQA/hotpot_train_v1.1.json",
        url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    ),
    RawDataDownloadSpec(
        dataset_key="nq-validation-shard",
        dataset="natural_questions",
        split="validation",
        target_path="data/raw/natural-questions/default/validation-00000-of-00007.parquet",
        url=(
            "https://huggingface.co/datasets/google-research-datasets/natural_questions"
            "/resolve/main/default/validation-00000-of-00007.parquet"
        ),
    ),
]


def download_missing_raw_data(
    *,
    project_root: Path,
    dataset_keys: list[str],
    output_dir: Path,
    dry_run: bool = False,
    overwrite: bool = False,
    fetcher: Fetcher | None = None,
    hf_fetcher: HfFetcher | None = None,
    prefer_hf: bool = False,
) -> dict[str, object]:
    specs = _select_specs(dataset_keys)
    output_dir.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or _url_fetch
    rows = [
        _download_one(
            spec,
            project_root=project_root,
            dry_run=dry_run,
            overwrite=overwrite,
            fetcher=fetch,
            hf_fetcher=hf_fetcher,
            prefer_hf=prefer_hf,
        )
        for spec in specs
    ]
    summary = {
        "dry_run": dry_run,
        "overwrite": overwrite,
        "prefer_hf": prefer_hf,
        "requested_dataset_keys": dataset_keys,
        "downloaded_count": sum(1 for row in rows if str(row["status"]).startswith("downloaded")),
        "already_exists_count": sum(1 for row in rows if row["status"] == "already_exists"),
        "would_download_count": sum(1 for row in rows if row["status"] == "would_download"),
        "outputs": {
            "summary_json": str(output_dir / "raw_data_download.json"),
            "summary_csv": str(output_dir / "raw_data_download.csv"),
        },
        "rows": rows,
    }
    (output_dir / "raw_data_download.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "raw_data_download.csv", index=False)
    return summary


def available_dataset_keys() -> list[str]:
    return [spec.dataset_key for spec in RAW_DATA_DOWNLOAD_SPECS]


def _select_specs(dataset_keys: list[str]) -> list[RawDataDownloadSpec]:
    by_key = {spec.dataset_key: spec for spec in RAW_DATA_DOWNLOAD_SPECS}
    selected_keys = available_dataset_keys() if dataset_keys == ["all"] else dataset_keys
    unknown = sorted(set(selected_keys) - set(by_key))
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {', '.join(unknown)}")
    return [by_key[key] for key in selected_keys]


def _download_one(
    spec: RawDataDownloadSpec,
    *,
    project_root: Path,
    dry_run: bool,
    overwrite: bool,
    fetcher: Fetcher,
    hf_fetcher: HfFetcher | None,
    prefer_hf: bool,
) -> dict[str, object]:
    target = project_root / spec.target_path
    exists_before = target.exists()
    row: dict[str, object] = {
        "dataset_key": spec.dataset_key,
        "dataset": spec.dataset,
        "split": spec.split,
        "target_path": spec.target_path,
        "target_exists_before": exists_before,
        "overwrite": overwrite,
        "dry_run": dry_run,
        "status": "",
        "size_bytes": target.stat().st_size if exists_before else 0,
        "error_type": "",
        "fallback_used": "",
    }
    if exists_before and not overwrite:
        row["status"] = "already_exists"
        return row
    if dry_run:
        row["status"] = "would_download"
        return row
    if prefer_hf and spec.hf_repo_id and spec.hf_converter:
        _download_hf_fallback(spec, target, hf_fetcher=hf_fetcher)
        row["status"] = "downloaded_hf"
        row["size_bytes"] = target.stat().st_size
        row["fallback_used"] = spec.hf_repo_id
        return row

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()
    try:
        fetcher(spec.url, part)
        shutil.move(str(part), str(target))
        row["status"] = "downloaded"
        row["size_bytes"] = target.stat().st_size
    except Exception as exc:
        if part.exists():
            part.unlink()
        if fetcher is _url_fetch and spec.hf_repo_id and spec.hf_converter:
            _download_hf_fallback(spec, target, hf_fetcher=hf_fetcher)
            row["status"] = "downloaded_hf_fallback"
            row["size_bytes"] = target.stat().st_size
            row["error_type"] = exc.__class__.__name__
            row["fallback_used"] = spec.hf_repo_id
            return row
        row["status"] = "error"
        row["error_type"] = exc.__class__.__name__
        raise
    return row


def _url_fetch(url: str, target: Path) -> None:
    with urlopen(url, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _download_hf_fallback(
    spec: RawDataDownloadSpec,
    target: Path,
    *,
    hf_fetcher: HfFetcher | None = None,
) -> None:
    from huggingface_hub import hf_hub_download

    def default_hf_fetch(repo_id: str, filename: str, repo_type: str) -> Path:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type))

    fetch = hf_fetcher or default_hf_fetch
    cached = fetch(spec.hf_repo_id, spec.hf_filename, spec.hf_repo_type)
    if spec.hf_converter == "hotpot_parquet_to_json":
        convert_hotpot_parquet_to_json(cached, target)
        return
    shutil.copy2(cached, target)


def convert_hotpot_parquet_to_json(parquet_path: Path, output_json: Path) -> int:
    frame = pd.read_parquet(parquet_path)
    rows = [_hotpot_row_to_original_json(row) for row in frame.to_dict(orient="records")]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def _hotpot_row_to_original_json(row: dict[str, object]) -> dict[str, object]:
    supporting_facts = row.get("supporting_facts") or {}
    context = row.get("context") or {}
    return {
        "_id": str(row.get("id") or ""),
        "question": str(row.get("question") or ""),
        "answer": str(row.get("answer") or ""),
        "type": str(row.get("type") or "unknown"),
        "level": str(row.get("level") or "unknown"),
        "supporting_facts": _zip_hotpot_pairs(supporting_facts, "title", "sent_id"),
        "context": _zip_hotpot_pairs(context, "title", "sentences"),
    }


def _zip_hotpot_pairs(value: object, left_key: str, right_key: str) -> list[list[object]]:
    if not isinstance(value, dict):
        return []
    left_values = _as_list(value.get(left_key))
    right_values = _as_list(value.get(right_key))
    return [
        [_to_jsonable(left), _to_jsonable(right)]
        for left, right in zip(left_values, right_values, strict=False)
    ]


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _to_jsonable(value: object) -> object:
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value
