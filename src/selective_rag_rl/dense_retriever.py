from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from selective_rag_rl.data import Passage
from selective_rag_rl.retriever import RetrievalResult


class Embedder(Protocol):
    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        ...


@dataclass
class DenseRetriever:
    passages: list[Passage]
    embedder: Embedder

    def __post_init__(self) -> None:
        self.doc_embeddings = np.asarray(
            self.embedder.encode(
                [p.text for p in self.passages],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        if not self.passages:
            return []
        query_embedding = np.asarray(
            self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=float,
        )
        scores = self.doc_embeddings @ query_embedding
        top = np.argsort(-scores, kind="mergesort")[:k]
        return [
            RetrievalResult(doc_id=self.passages[int(i)].doc_id, score=float(scores[int(i)]), rank=rank + 1)
            for rank, i in enumerate(top)
        ]


def load_sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def hybrid_merge(
    bm25_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
    k: int,
    dense_weight: float = 0.5,
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    for result in bm25_results:
        scores[result.doc_id] = scores.get(result.doc_id, 0.0) + (1.0 - dense_weight) / result.rank
    for result in dense_results:
        scores[result.doc_id] = scores.get(result.doc_id, 0.0) + dense_weight / result.rank
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [RetrievalResult(doc_id=doc_id, score=score, rank=i + 1) for i, (doc_id, score) in enumerate(ranked)]
