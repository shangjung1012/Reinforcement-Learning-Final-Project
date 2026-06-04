from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.core.data import Passage, QAExample
from selective_rag_rl.preflight.embedding_preflight import (
    estimate_combined_embedding_workload,
    estimate_embedding_workload,
    estimate_repeated_split_embedding_workloads,
    write_embedding_workload_report,
)


def test_estimate_embedding_workload_counts_cache_hits_and_misses(tmp_path: Path) -> None:
    cache_path = tmp_path / "vertex_embeddings.jsonl"
    cache_path.write_text(
        "\n".join(
            [
                json.dumps(_cache_row("gemini-embedding-001", "RETRIEVAL_QUERY", "What is aspirin?")),
                json.dumps(_cache_row("gemini-embedding-001", "RETRIEVAL_DOCUMENT", "Aspirin. Aspirin is a medicine.")),
            ]
        ),
        encoding="utf-8",
    )
    examples = [
        QAExample(
            qid="q1",
            question="What is aspirin?",
            answer="",
            passages=[
                Passage("d1", "Aspirin", "Aspirin. Aspirin is a medicine."),
                Passage("d2", "Orange", "Orange. Citrus fruit."),
            ],
            gold_doc_ids={"d1"},
            level="test",
            qtype="toy",
        ),
        QAExample(
            qid="q2",
            question="What is orange?",
            answer="",
            passages=[
                Passage("d1", "Aspirin", "Aspirin. Aspirin is a medicine."),
                Passage("d2", "Orange", "Orange. Citrus fruit."),
            ],
            gold_doc_ids={"d2"},
            level="test",
            qtype="toy",
        ),
    ]

    report = estimate_embedding_workload(examples, cache_path=cache_path, k=2)

    assert report["query_unique"] == 2
    assert report["query_cached"] == 1
    assert report["query_missing"] == 1
    assert report["document_unique"] == 2
    assert report["document_cached"] == 1
    assert report["document_missing"] == 1
    assert report["total_missing"] == 2


def test_write_embedding_workload_report_writes_csv(tmp_path: Path) -> None:
    output_csv = tmp_path / "report.csv"

    written = write_embedding_workload_report(
        output_csv,
        {
            "dataset": "toy",
            "split": "train",
            "query_unique": 2,
            "query_cached": 1,
            "query_missing": 1,
            "document_unique": 3,
            "document_cached": 2,
            "document_missing": 1,
            "total_missing": 2,
        },
    )

    rows = pd.read_csv(written)
    assert list(rows["dataset"]) == ["toy"]
    assert list(rows["total_missing"]) == [2]


def test_estimate_combined_embedding_workload_counts_union_once(tmp_path: Path) -> None:
    cache_path = tmp_path / "empty.jsonl"
    examples = [
        QAExample(
            qid="q1",
            question="Shared question?",
            answer="",
            passages=[Passage("d1", "Shared", "Shared. Same document.")],
            gold_doc_ids={"d1"},
            level="train",
            qtype="toy",
        ),
        QAExample(
            qid="q2",
            question="Shared question?",
            answer="",
            passages=[Passage("d1", "Shared", "Shared. Same document.")],
            gold_doc_ids={"d1"},
            level="test",
            qtype="toy",
        ),
    ]

    report = estimate_combined_embedding_workload(examples, cache_path=cache_path, k=1, dataset="toy")

    assert report["split"] == "combined"
    assert report["examples"] == 2
    assert report["query_unique"] == 1
    assert report["document_unique"] == 1
    assert report["total_missing"] == 2


def test_estimate_repeated_split_embedding_workloads_counts_cross_seed_union(tmp_path: Path) -> None:
    cache_path = tmp_path / "vertex_embeddings.jsonl"
    cache_path.write_text(
        "\n".join(
            [
                json.dumps(_cache_row("gemini-embedding-001", "RETRIEVAL_QUERY", "Shared question?")),
                json.dumps(_cache_row("gemini-embedding-001", "RETRIEVAL_DOCUMENT", "Shared. Same document.")),
            ]
        ),
        encoding="utf-8",
    )
    seed_splits = [
        (1, [_example("q1", "Shared question?", "Shared. Same document.")], [_example("q2", "Seed one?", "One. Document.")]),
        (2, [_example("q3", "Shared question?", "Shared. Same document.")], [_example("q4", "Seed two?", "Two. Document.")]),
    ]

    reports = estimate_repeated_split_embedding_workloads(seed_splits, cache_path=cache_path, k=1, dataset="toy")
    repeated = reports[-1]

    assert [report["split"] for report in reports] == [
        "seed_1_train",
        "seed_1_test",
        "seed_1_combined",
        "seed_2_train",
        "seed_2_test",
        "seed_2_combined",
        "repeated_combined",
    ]
    assert repeated["examples"] == 4
    assert repeated["query_unique"] == 3
    assert repeated["query_cached"] == 1
    assert repeated["document_unique"] == 3
    assert repeated["document_cached"] == 1
    assert repeated["total_missing"] == 4


def _example(qid: str, question: str, passage_text: str) -> QAExample:
    return QAExample(
        qid=qid,
        question=question,
        answer="",
        passages=[Passage(qid.replace("q", "d"), "Title", passage_text)],
        gold_doc_ids={qid.replace("q", "d")},
        level="test",
        qtype="toy",
    )


def _cache_row(model: str, task_type: str, text: str) -> dict[str, object]:
    return {
        "model": model,
        "task_type": task_type,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "embedding": [1.0, 0.0],
    }
