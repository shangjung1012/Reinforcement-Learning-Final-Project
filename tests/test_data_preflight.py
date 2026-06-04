from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.preflight.data_preflight import run_data_preflight


def test_data_preflight_reports_missing_expected_paths(tmp_path: Path) -> None:
    summary = run_data_preflight(project_root=tmp_path, output_dir=tmp_path / "out")

    assert summary["all_required_available"] is False
    rows = pd.read_csv(tmp_path / "out" / "data_preflight.csv")
    assert "data/raw/scifact/corpus.jsonl" in set(rows["path"])
    assert set(rows["exists"]) == {False}
    assert set(rows["status"]) == {"missing"}


def test_data_preflight_counts_existing_json_jsonl_and_tsv_without_dumping_content(tmp_path: Path) -> None:
    hotpot = tmp_path / "data/raw/HotpotQA"
    scifact = tmp_path / "data/raw/scifact"
    qrels = scifact / "qrels"
    hotpot.mkdir(parents=True)
    qrels.mkdir(parents=True)
    (hotpot / "hotpot_dev_distractor_v1.json").write_text(
        json.dumps([{"_id": "a"}, {"_id": "b"}]),
        encoding="utf-8",
    )
    (scifact / "corpus.jsonl").write_text('{"_id": "d1"}\n{"_id": "d2"}\n', encoding="utf-8")
    (qrels / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")

    summary = run_data_preflight(project_root=tmp_path, output_dir=tmp_path / "out")

    rows = pd.read_csv(tmp_path / "out" / "data_preflight.csv")
    by_path = {row["path"]: row for _, row in rows.iterrows()}
    assert by_path["data/raw/HotpotQA/hotpot_dev_distractor_v1.json"]["row_count"] == 2
    assert by_path["data/raw/scifact/corpus.jsonl"]["row_count"] == 2
    assert by_path["data/raw/scifact/qrels/test.tsv"]["row_count"] == 1
    serialized = json.dumps(summary)
    assert '"_id": "a"' not in serialized
    assert "q1" not in serialized
