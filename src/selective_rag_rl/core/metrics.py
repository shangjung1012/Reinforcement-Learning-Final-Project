from __future__ import annotations

import math

from selective_rag_rl.core.retriever import RetrievalResult


def recall_at_k(results: list[RetrievalResult], gold_doc_ids: set[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    retrieved = {r.doc_id for r in results[:k]}
    return len(retrieved & gold_doc_ids) / len(gold_doc_ids)


def mrr(results: list[RetrievalResult], gold_doc_ids: set[str]) -> float:
    for result in results:
        if result.doc_id in gold_doc_ids:
            return 1.0 / result.rank
    return 0.0


def ndcg_at_k(results: list[RetrievalResult], gold_doc_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, result in enumerate(results[:k], start=1):
        if result.doc_id in gold_doc_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

