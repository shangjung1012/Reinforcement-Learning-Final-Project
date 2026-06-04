from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from selective_rag_rl.core.data import Passage
from selective_rag_rl.core.text import tokenize


@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    score: float
    rank: int


class BM25Retriever:
    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        tokenized = [tokenize(p.text) for p in passages]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        if scores.size == 0:
            return []
        top = np.argsort(-scores, kind="mergesort")[:k]
        return [
            RetrievalResult(
                doc_id=self.passages[int(i)].doc_id,
                score=float(scores[int(i)]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top)
        ]

