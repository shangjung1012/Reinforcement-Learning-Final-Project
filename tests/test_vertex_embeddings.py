from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from selective_rag_rl.integrations.vertex_embeddings import VertexEmbeddingBudgetError, VertexTextEmbeddingProvider


def test_vertex_provider_returns_cached_embedding_without_live_client(tmp_path: Path) -> None:
    cache_path = tmp_path / "vertex_embeddings.jsonl"
    _write_cache(cache_path, "Cached question?", "RETRIEVAL_QUERY", [3.0, 4.0])

    provider = VertexTextEmbeddingProvider(project_root=tmp_path, cache_path=cache_path)

    vector = provider.embed_text("Cached question?", task_type="RETRIEVAL_QUERY")

    assert np.allclose(vector, np.asarray([3.0, 4.0]))
    assert provider.cache_hits == 1
    assert provider.cache_misses == 0
    assert provider.actual_new_texts == 0


def test_vertex_provider_blocks_cache_miss_without_allow_api(tmp_path: Path) -> None:
    provider = VertexTextEmbeddingProvider(project_root=tmp_path, cache_path=tmp_path / "vertex_embeddings.jsonl")

    with pytest.raises(VertexEmbeddingBudgetError, match="allow_api=True"):
        provider.embed_text("New question?", task_type="RETRIEVAL_QUERY")

    assert provider.cache_hits == 0
    assert provider.cache_misses == 1
    assert provider.actual_new_texts == 0


def test_vertex_provider_blocks_cache_miss_above_budget(tmp_path: Path) -> None:
    provider = VertexTextEmbeddingProvider(
        project_root=tmp_path,
        cache_path=tmp_path / "vertex_embeddings.jsonl",
        allow_api=True,
        max_new_texts=1,
        fetcher=lambda texts, task_type: [np.ones(2) for _ in texts],
    )

    with pytest.raises(VertexEmbeddingBudgetError, match="exceed max_new_texts"):
        provider.embed_texts(["first", "second"], task_type="RETRIEVAL_QUERY")

    assert provider.cache_misses == 2
    assert provider.actual_new_texts == 0


def test_vertex_provider_fetches_and_caches_within_budget(tmp_path: Path) -> None:
    calls: list[tuple[list[str], str]] = []

    def fetcher(texts: list[str], task_type: str) -> list[np.ndarray]:
        calls.append((texts, task_type))
        return [np.asarray([float(index + 1), 0.0]) for index, _text in enumerate(texts)]

    cache_path = tmp_path / "vertex_embeddings.jsonl"
    provider = VertexTextEmbeddingProvider(
        project_root=tmp_path,
        cache_path=cache_path,
        allow_api=True,
        max_new_texts=2,
        fetcher=fetcher,
    )

    vectors = provider.embed_texts(["first", "second", "first"], task_type="RETRIEVAL_QUERY")

    assert [vector.tolist() for vector in vectors] == [[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]]
    assert calls == [(["first", "second"], "RETRIEVAL_QUERY")]
    assert provider.cache_hits == 1
    assert provider.cache_misses == 2
    assert provider.actual_new_texts == 2
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 2


def test_vertex_provider_dry_run_reports_misses_without_fetching(tmp_path: Path) -> None:
    calls: list[str] = []
    provider = VertexTextEmbeddingProvider(
        project_root=tmp_path,
        cache_path=tmp_path / "vertex_embeddings.jsonl",
        dry_run=True,
        allow_api=True,
        max_new_texts=3,
        fetcher=lambda texts, task_type: calls.append(task_type) or [np.ones(2) for _ in texts],
    )

    with pytest.raises(VertexEmbeddingBudgetError, match="dry_run_no_api_call") as exc_info:
        provider.embed_texts(["alpha", "beta"], task_type="RETRIEVAL_DOCUMENT")

    assert calls == []
    assert exc_info.value.missing == 2
    assert exc_info.value.allowed == 3
    assert exc_info.value.dry_run is True
    assert provider.actual_new_texts == 0


def _write_cache(cache_path: Path, text: str, task_type: str, embedding: list[float]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "model": "gemini-embedding-001",
                "task_type": task_type,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "embedding": embedding,
            }
        )
        + "\n",
        encoding="utf-8",
    )
