from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.data import QAExample, load_hotpotqa, split_examples
from selective_rag_rl.dense_retriever import DenseRetriever, hybrid_merge, load_sentence_transformer
from selective_rag_rl.experiment import _latex_table
from selective_rag_rl.metrics import mrr, ndcg_at_k, recall_at_k
from selective_rag_rl.retriever import BM25Retriever, RetrievalResult
from selective_rag_rl.rewrites import rewrite, rewrite_cost


class FakeDenseEmbedder:
    def encode(self, texts: list[str], normalize_embeddings: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        vectors = []
        for text in texts:
            tokens = set(text.lower().split())
            vectors.append([float(len(tokens)), float(sum(len(t) for t in tokens)), 1.0])
        arr = np.asarray(vectors, dtype=float)
        if normalize_embeddings:
            arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        return arr


def run_dense_hotpot_experiment(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 300,
    seed: int = 42,
    k: int = 5,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
) -> dict[str, object]:
    examples = load_hotpotqa(data_path, num_examples=num_examples, seed=seed)
    _train, test = split_examples(examples)
    embedder = FakeDenseEmbedder() if embedder_name == "fake" else load_sentence_transformer(embedder_name)

    rows: list[dict[str, object]] = []
    for ex in tqdm(test, desc="dense baselines"):
        dense_retriever = DenseRetriever(ex.passages, embedder)
        bm25_retriever = BM25Retriever(ex.passages)
        rows.extend(_evaluate_example(ex, dense_retriever, bm25_retriever, k, dense_weight))

    df = pd.DataFrame(rows)
    summary = _summarize(df)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = results_dir / "dense_hotpot_detailed.csv"
    summary_csv = results_dir / "dense_hotpot_summary.csv"
    summary_json = results_dir / "dense_hotpot_summary.json"
    table_tex = results_dir / "dense_hotpot_table.tex"
    df.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    table_tex.write_text(_latex_table(summary), encoding="utf-8")
    metadata = {
        "dataset": "HotpotQA dense retrieval baselines",
        "num_examples": len(examples),
        "test_examples": len(test),
        "seed": seed,
        "k": k,
        "embedder": embedder_name,
        "dense_weight": dense_weight,
        "outputs": {
            "detailed_csv": str(detailed_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "table_tex": str(table_tex),
        },
    }
    (results_dir / "dense_hotpot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _evaluate_example(
    ex: QAExample,
    dense_retriever: DenseRetriever,
    bm25_retriever: BM25Retriever,
    k: int,
    dense_weight: float,
) -> list[dict[str, object]]:
    original = rewrite(ex.question, "keep")
    keyword = rewrite(ex.question, "keyword_compress")
    dense_original = dense_retriever.search(original.joined_query, k=k)
    dense_keyword = dense_retriever.search(keyword.joined_query, k=k)
    bm25_original = bm25_retriever.search(original.joined_query, k=k)
    bm25_keyword = bm25_retriever.search(keyword.joined_query, k=k)
    return [
        _row(ex, "Dense original", "keep", dense_original, 0.0, 1, original.joined_query, k),
        _row(ex, "Dense keyword", "keyword_compress", dense_keyword, rewrite_cost(ex.question, keyword), 1, keyword.joined_query, k),
        _row(
            ex,
            "Hybrid original",
            "keep",
            hybrid_merge(bm25_original, dense_original, k=k, dense_weight=dense_weight),
            0.0,
            2,
            original.joined_query,
            k,
        ),
        _row(
            ex,
            "Hybrid keyword",
            "keyword_compress",
            hybrid_merge(bm25_keyword, dense_keyword, k=k, dense_weight=dense_weight),
            rewrite_cost(ex.question, keyword),
            2,
            keyword.joined_query,
            k,
        ),
    ]


def _row(
    ex: QAExample,
    method: str,
    action: str,
    results: list[RetrievalResult],
    cost: float,
    calls: int,
    query: str,
    k: int,
) -> dict[str, object]:
    rec = recall_at_k(results, ex.gold_doc_ids, k)
    rr = mrr(results, ex.gold_doc_ids)
    return {
        "method": method,
        "action": action,
        "qid": ex.qid,
        "question": ex.question,
        "recall_at_5": rec,
        "mrr": rr,
        "ndcg_at_5": ndcg_at_k(results, ex.gold_doc_ids, k),
        "reward": rec + 0.5 * rr - cost,
        "rewrite_cost": cost,
        "retrieval_calls": calls,
        "queries": query,
        "top_docs": " | ".join(r.doc_id for r in results),
        "gold_docs": " | ".join(sorted(ex.gold_doc_ids)),
    }


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["Dense original", "Dense keyword", "Hybrid original", "Hybrid keyword"]:
        part = df[df["method"] == method]
        rows.append(
            {
                "method": method,
                "recall_at_5": float(part["recall_at_5"].mean()),
                "mrr": float(part["mrr"].mean()),
                "ndcg_at_5": float(part["ndcg_at_5"].mean()),
                "reward": float(part["reward"].mean()),
                "rewrite_cost": float(part["rewrite_cost"].mean()),
                "retrieval_calls": float(part["retrieval_calls"].mean()),
            }
        )
    return pd.DataFrame(rows)
