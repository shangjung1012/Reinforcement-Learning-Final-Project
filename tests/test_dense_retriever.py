from __future__ import annotations

import numpy as np

from selective_rag_rl.core.data import Passage
from selective_rag_rl.core.dense_retriever import DenseRetriever, hybrid_merge
from selective_rag_rl.core.retriever import RetrievalResult


class FakeEmbedder:
    def encode(self, texts: list[str], normalize_embeddings: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            if "apple" in lower:
                vectors.append([1.0, 0.0])
            elif "carrot" in lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.1, 0.1])
        arr = np.asarray(vectors, dtype=float)
        if normalize_embeddings:
            arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        return arr


def test_dense_retriever_ranks_by_embedding_similarity() -> None:
    passages = [
        Passage("a", "Apple", "Apple fruit"),
        Passage("b", "Carrot", "Carrot vegetable"),
    ]
    retriever = DenseRetriever(passages, FakeEmbedder())

    results = retriever.search("apple", k=2)

    assert [r.doc_id for r in results] == ["a", "b"]


def test_hybrid_merge_combines_bm25_and_dense_ranks() -> None:
    bm25 = [RetrievalResult("a", 10.0, 1), RetrievalResult("b", 1.0, 2)]
    dense = [RetrievalResult("b", 0.9, 1), RetrievalResult("a", 0.1, 2)]

    results = hybrid_merge(bm25, dense, k=2, dense_weight=0.75)

    assert results[0].doc_id == "b"
    assert {r.doc_id for r in results} == {"a", "b"}
