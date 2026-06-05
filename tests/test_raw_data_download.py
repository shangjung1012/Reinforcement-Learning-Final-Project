from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from selective_rag_rl.preflight.raw_data_download import convert_hotpot_parquet_to_json, download_missing_raw_data


def test_download_missing_raw_data_dry_run_reports_missing_without_fetch(tmp_path: Path) -> None:
    def fail_fetch(_url: str, _target: Path) -> None:
        raise AssertionError("dry-run should not fetch")

    summary = download_missing_raw_data(
        project_root=tmp_path,
        dataset_keys=["hotpot-dev-distractor"],
        output_dir=tmp_path / "outputs",
        dry_run=True,
        fetcher=fail_fetch,
    )

    row = summary["rows"][0]
    assert row["dataset_key"] == "hotpot-dev-distractor"
    assert row["status"] == "would_download"
    assert row["target_exists_before"] is False
    assert not (tmp_path / "data/raw/HotpotQA/hotpot_dev_distractor_v1.json").exists()
    assert Path(summary["outputs"]["summary_csv"]).exists()


def test_download_missing_raw_data_skips_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "data/raw/HotpotQA/hotpot_dev_distractor_v1.json"
    target.parent.mkdir(parents=True)
    target.write_text("[]", encoding="utf-8")

    def fail_fetch(_url: str, _target: Path) -> None:
        raise AssertionError("existing file should not fetch")

    summary = download_missing_raw_data(
        project_root=tmp_path,
        dataset_keys=["hotpot-dev-distractor"],
        output_dir=tmp_path / "outputs",
        dry_run=False,
        fetcher=fail_fetch,
    )

    row = summary["rows"][0]
    assert row["status"] == "already_exists"
    assert row["size_bytes"] == 2


def test_download_missing_raw_data_uses_injected_fetcher(tmp_path: Path) -> None:
    def fake_fetch(_url: str, target: Path) -> None:
        target.write_text('[{"id": "toy"}]', encoding="utf-8")

    summary = download_missing_raw_data(
        project_root=tmp_path,
        dataset_keys=["hotpot-dev-distractor"],
        output_dir=tmp_path / "outputs",
        dry_run=False,
        fetcher=fake_fetch,
    )

    target = tmp_path / "data/raw/HotpotQA/hotpot_dev_distractor_v1.json"
    row = summary["rows"][0]
    csv = pd.read_csv(summary["outputs"]["summary_csv"]).iloc[0]

    assert target.exists()
    assert row["status"] == "downloaded"
    assert row["size_bytes"] == target.stat().st_size
    assert csv["status"] == "downloaded"


def test_download_missing_raw_data_prefer_hf_uses_injected_hf_fetcher(tmp_path: Path) -> None:
    def fail_url_fetch(_url: str, _target: Path) -> None:
        raise AssertionError("prefer_hf should not call the official URL fetcher")

    def fake_hf_fetch(_repo_id: str, _filename: str, _repo_type: str) -> Path:
        parquet = tmp_path / "cached.parquet"
        pd.DataFrame(
            [
                {
                    "id": "q1",
                    "question": "Who wrote notes for the Analytical Engine?",
                    "answer": "Ada Lovelace",
                    "type": "bridge",
                    "level": "easy",
                    "supporting_facts": {"title": ["Ada Lovelace"], "sent_id": [0]},
                    "context": {
                        "title": ["Ada Lovelace"],
                        "sentences": [["Ada Lovelace wrote notes for the Analytical Engine."]],
                    },
                }
            ]
        ).to_parquet(parquet)
        return parquet

    summary = download_missing_raw_data(
        project_root=tmp_path,
        dataset_keys=["hotpot-dev-distractor"],
        output_dir=tmp_path / "outputs",
        dry_run=False,
        fetcher=fail_url_fetch,
        hf_fetcher=fake_hf_fetch,
        prefer_hf=True,
    )

    target = tmp_path / "data/raw/HotpotQA/hotpot_dev_distractor_v1.json"
    row = summary["rows"][0]
    assert target.exists()
    assert row["status"] == "downloaded_hf"
    assert row["fallback_used"] == "hotpotqa/hotpot_qa"


def test_download_missing_raw_data_unknown_dataset_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataset key"):
        download_missing_raw_data(
            project_root=tmp_path,
            dataset_keys=["unknown"],
            output_dir=tmp_path / "outputs",
            dry_run=True,
        )


def test_convert_hotpot_parquet_to_json_preserves_loader_schema(tmp_path: Path) -> None:
    parquet = tmp_path / "hotpot.parquet"
    output = tmp_path / "hotpot.json"
    pd.DataFrame(
        [
            {
                "id": "q1",
                "question": "Who wrote notes for the Analytical Engine?",
                "answer": "Ada Lovelace",
                "type": "bridge",
                "level": "easy",
                "supporting_facts": {"title": ["Ada Lovelace"], "sent_id": [0]},
                "context": {
                    "title": ["Ada Lovelace", "Other"],
                    "sentences": [
                        ["Ada Lovelace wrote notes for the Analytical Engine."],
                        ["Other evidence."],
                    ],
                },
            }
        ]
    ).to_parquet(parquet)

    count = convert_hotpot_parquet_to_json(parquet, output)

    value = output.read_text(encoding="utf-8")
    assert count == 1
    assert '"_id": "q1"' in value
    assert '"supporting_facts": [[' in value
    assert '"context": [[' in value
