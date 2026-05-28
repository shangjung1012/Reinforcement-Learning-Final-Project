from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.data import QAExample
from selective_rag_rl.retriever import BM25Retriever

VERTEX_EMBEDDING_MODEL = "gemini-embedding-001"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"


def estimate_embedding_workload(
    examples: list[QAExample],
    cache_path: Path,
    k: int = 5,
    dataset: str = "",
    split: str = "",
    model: str = VERTEX_EMBEDDING_MODEL,
) -> dict[str, object]:
    cached = _load_cached_keys(cache_path)
    query_texts, document_texts = collect_embedding_texts(examples, k=k)
    return estimate_embedding_text_workload(
        query_texts,
        document_texts,
        cached,
        cache_path=cache_path,
        k=k,
        dataset=dataset,
        split=split,
        examples=len(examples),
        model=model,
    )


def estimate_combined_embedding_workload(
    examples: list[QAExample],
    cache_path: Path,
    k: int = 5,
    dataset: str = "",
    model: str = VERTEX_EMBEDDING_MODEL,
) -> dict[str, object]:
    return estimate_embedding_workload(
        examples,
        cache_path=cache_path,
        k=k,
        dataset=dataset,
        split="combined",
        model=model,
    )


def estimate_split_embedding_workloads(
    train_examples: list[QAExample],
    test_examples: list[QAExample],
    cache_path: Path,
    k: int = 5,
    dataset: str = "",
    model: str = VERTEX_EMBEDDING_MODEL,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    cached = _load_cached_keys(cache_path)
    train_queries, train_documents = collect_embedding_texts(train_examples, k=k)
    test_queries, test_documents = collect_embedding_texts(test_examples, k=k)
    train_report = estimate_embedding_text_workload(
        train_queries,
        train_documents,
        cached,
        cache_path=cache_path,
        k=k,
        dataset=dataset,
        split="train",
        examples=len(train_examples),
        model=model,
    )
    test_report = estimate_embedding_text_workload(
        test_queries,
        test_documents,
        cached,
        cache_path=cache_path,
        k=k,
        dataset=dataset,
        split="test",
        examples=len(test_examples),
        model=model,
    )
    combined_report = estimate_embedding_text_workload(
        train_queries | test_queries,
        train_documents | test_documents,
        cached,
        cache_path=cache_path,
        k=k,
        dataset=dataset,
        split="combined",
        examples=len(train_examples) + len(test_examples),
        model=model,
    )
    return train_report, test_report, combined_report


def estimate_repeated_split_embedding_workloads(
    seed_splits: list[tuple[int, list[QAExample], list[QAExample]]],
    cache_path: Path,
    k: int = 5,
    dataset: str = "",
    model: str = VERTEX_EMBEDDING_MODEL,
) -> list[dict[str, object]]:
    cached = _load_cached_keys(cache_path)
    reports: list[dict[str, object]] = []
    repeated_queries: set[str] = set()
    repeated_documents: set[str] = set()
    repeated_examples = 0

    for seed, train_examples, test_examples in seed_splits:
        train_queries, train_documents = collect_embedding_texts(train_examples, k=k)
        test_queries, test_documents = collect_embedding_texts(test_examples, k=k)
        repeated_queries |= train_queries | test_queries
        repeated_documents |= train_documents | test_documents
        repeated_examples += len(train_examples) + len(test_examples)
        reports.extend(
            [
                estimate_embedding_text_workload(
                    train_queries,
                    train_documents,
                    cached,
                    cache_path=cache_path,
                    k=k,
                    dataset=dataset,
                    split=f"seed_{seed}_train",
                    examples=len(train_examples),
                    model=model,
                ),
                estimate_embedding_text_workload(
                    test_queries,
                    test_documents,
                    cached,
                    cache_path=cache_path,
                    k=k,
                    dataset=dataset,
                    split=f"seed_{seed}_test",
                    examples=len(test_examples),
                    model=model,
                ),
                estimate_embedding_text_workload(
                    train_queries | test_queries,
                    train_documents | test_documents,
                    cached,
                    cache_path=cache_path,
                    k=k,
                    dataset=dataset,
                    split=f"seed_{seed}_combined",
                    examples=len(train_examples) + len(test_examples),
                    model=model,
                ),
            ]
        )

    reports.append(
        estimate_embedding_text_workload(
            repeated_queries,
            repeated_documents,
            cached,
            cache_path=cache_path,
            k=k,
            dataset=dataset,
            split="repeated_combined",
            examples=repeated_examples,
            model=model,
        )
    )
    return reports


def collect_embedding_texts(examples: list[QAExample], k: int = 5) -> tuple[set[str], set[str]]:
    query_texts = {ex.question.strip() for ex in examples if ex.question.strip()}
    document_texts = _initial_top_document_texts(examples, k=k)
    return query_texts, document_texts


def estimate_embedding_text_workload(
    query_texts: set[str],
    document_texts: set[str],
    cached: set[tuple[str, str, str]],
    cache_path: Path,
    k: int,
    dataset: str,
    split: str,
    examples: int,
    model: str = VERTEX_EMBEDDING_MODEL,
) -> dict[str, object]:
    query_cached = _cached_count(query_texts, cached, model, QUERY_TASK_TYPE)
    document_cached = _cached_count(document_texts, cached, model, DOCUMENT_TASK_TYPE)
    query_missing = len(query_texts) - query_cached
    document_missing = len(document_texts) - document_cached
    return {
        "dataset": dataset,
        "split": split,
        "examples": examples,
        "k": k,
        "model": model,
        "cache_path": str(cache_path),
        "query_unique": len(query_texts),
        "query_cached": query_cached,
        "query_missing": query_missing,
        "document_unique": len(document_texts),
        "document_cached": document_cached,
        "document_missing": document_missing,
        "total_unique": len(query_texts) + len(document_texts),
        "total_cached": query_cached + document_cached,
        "total_missing": query_missing + document_missing,
    }


def write_embedding_workload_report(output_csv: Path, *reports: dict[str, object]) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(reports)).to_csv(output_csv, index=False)
    return output_csv


def _initial_top_document_texts(examples: list[QAExample], k: int) -> set[str]:
    texts: set[str] = set()
    for ex in examples:
        passage_by_id = {passage.doc_id: passage for passage in ex.passages}
        bm25 = BM25Retriever(ex.passages)
        for result in bm25.search(ex.question, k=k):
            passage = passage_by_id.get(result.doc_id)
            if passage is not None and passage.text.strip():
                texts.add(passage.text.strip())
    return texts


def _load_cached_keys(cache_path: Path) -> set[tuple[str, str, str]]:
    if not cache_path.exists():
        return set()
    keys = set()
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((str(row["model"]), str(row["task_type"]), str(row["text_hash"])))
    return keys


def _cached_count(texts: set[str], cached: set[tuple[str, str, str]], model: str, task_type: str) -> int:
    return sum(1 for text in texts if (model, task_type, _text_hash(text)) in cached)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
