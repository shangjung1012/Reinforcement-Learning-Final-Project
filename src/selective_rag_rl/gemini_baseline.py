from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from selective_rag_rl.data import QAExample, load_hotpotqa, split_examples
from selective_rag_rl.experiment import _latex_table, _merge_results
from selective_rag_rl.metrics import mrr, ndcg_at_k, recall_at_k
from selective_rag_rl.retriever import BM25Retriever
from selective_rag_rl.text import content_tokens

RewriteProvider = Callable[[str, str], list[str]]
GEMINI_MODES = [("rewrite", "Gemini rewrite-all"), ("decompose", "Gemini decompose")]


class GeminiBudgetError(RuntimeError):
    pass


class GeminiCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[tuple[str, str], list[str]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows[(row["qid"], row["mode"])] = list(row["queries"])

    def get(self, qid: str, mode: str) -> list[str] | None:
        return self.rows.get((qid, mode))

    def set(self, qid: str, mode: str, queries: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows[(qid, mode)] = queries
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"qid": qid, "mode": mode, "queries": queries}, ensure_ascii=False) + "\n")


class VertexGeminiRewriter:
    def __init__(
        self,
        project_root: Path,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        load_dotenv(project_root / ".env")
        project = os.environ["GOOGLE_CLOUD_PROJECT"]
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        credentials = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).expanduser()
        if not credentials.is_absolute():
            credentials = project_root / credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def rewrite(self, question: str, mode: str) -> list[str]:
        prompt = _prompt(question, mode)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        return _parse_queries(response.text or "", mode)


def evaluate_gemini_rewrites(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 300,
    seed: int = 42,
    k: int = 5,
    cache_path: Path | None = None,
    rewrite_provider: RewriteProvider | None = None,
    allow_api: bool = False,
    max_new_calls: int = 0,
    dry_run: bool = False,
) -> dict[str, object]:
    examples = load_hotpotqa(data_path, num_examples=num_examples, seed=seed)
    _train, test = split_examples(examples)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache = GeminiCache(cache_path or output_dir / "cache" / "gemini_rewrites.jsonl")
    cache_stats = _cache_stats(test, cache)
    metadata = _metadata(
        examples=examples,
        test=test,
        seed=seed,
        k=k,
        cache=cache,
        results_dir=results_dir,
        cache_stats=cache_stats,
        allow_api=allow_api,
        max_new_calls=max_new_calls,
        dry_run=dry_run,
    )
    if dry_run:
        Path(metadata["outputs"]["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    if rewrite_provider is None:
        _check_live_budget(cache_stats["misses"], allow_api=allow_api, max_new_calls=max_new_calls)
    provider = rewrite_provider or VertexGeminiRewriter(Path.cwd()).rewrite

    rows: list[dict[str, object]] = []
    for ex in tqdm(test, desc="gemini baselines"):
        for mode, method in GEMINI_MODES:
            queries = cache.get(ex.qid, mode)
            if queries is None:
                queries = provider(ex.question, mode)
                cache.set(ex.qid, mode, queries)
            rows.append(_evaluate_queries(ex, method, mode, queries, k))

    df = pd.DataFrame(rows)
    summary = _summarize(df)
    detailed_csv = results_dir / "gemini_rewrite_detailed.csv"
    summary_csv = results_dir / "gemini_rewrite_summary.csv"
    summary_json = results_dir / "gemini_rewrite_summary.json"
    table_tex = results_dir / "gemini_rewrite_table.tex"
    df.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    table_tex.write_text(_latex_table(summary), encoding="utf-8")
    metadata["outputs"].update(
        {
            "detailed_csv": str(detailed_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "table_tex": str(table_tex),
        }
    )
    (results_dir / "gemini_rewrite_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _cache_stats(test_examples: list[QAExample], cache: GeminiCache) -> dict[str, int]:
    hits = 0
    misses = 0
    for ex in test_examples:
        for mode, _method in GEMINI_MODES:
            if cache.get(ex.qid, mode) is None:
                misses += 1
            else:
                hits += 1
    return {"hits": hits, "misses": misses}


def _check_live_budget(misses: int, allow_api: bool, max_new_calls: int) -> None:
    if misses == 0:
        return
    if not allow_api:
        raise GeminiBudgetError("Gemini cache misses require allow_api=True before live API calls")
    if misses > max_new_calls:
        raise GeminiBudgetError(f"Gemini cache misses ({misses}) exceed max_new_calls ({max_new_calls})")


def _metadata(
    examples: list[QAExample],
    test: list[QAExample],
    seed: int,
    k: int,
    cache: GeminiCache,
    results_dir: Path,
    cache_stats: dict[str, int],
    allow_api: bool,
    max_new_calls: int,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "dataset": "HotpotQA Gemini rewrite baselines",
        "num_examples": len(examples),
        "test_examples": len(test),
        "seed": seed,
        "k": k,
        "cache": str(cache.path),
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "allow_api": allow_api,
        "max_new_calls": max_new_calls,
        "dry_run": dry_run,
        "outputs": {
            "metadata_json": str(results_dir / "gemini_rewrite_metadata.json"),
        },
    }


def _evaluate_queries(ex: QAExample, method: str, mode: str, queries: list[str], k: int) -> dict[str, object]:
    retriever = BM25Retriever(ex.passages)
    merged = _merge_results([retriever.search(q, k=k) for q in queries], k=k)
    cost = 1.0 + 0.01 * sum(len(content_tokens(q)) for q in queries)
    rec = recall_at_k(merged, ex.gold_doc_ids, k)
    rr = mrr(merged, ex.gold_doc_ids)
    return {
        "method": method,
        "mode": mode,
        "qid": ex.qid,
        "question": ex.question,
        "recall_at_5": rec,
        "mrr": rr,
        "ndcg_at_5": ndcg_at_k(merged, ex.gold_doc_ids, k),
        "reward": rec + 0.5 * rr - cost,
        "rewrite_cost": cost,
        "retrieval_calls": len(queries),
        "rewrite_tokens": sum(len(q.split()) for q in queries),
        "queries": " || ".join(queries),
        "top_docs": " | ".join(r.doc_id for r in merged),
        "gold_docs": " | ".join(sorted(ex.gold_doc_ids)),
    }


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["Gemini rewrite-all", "Gemini decompose"]:
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


def _prompt(question: str, mode: str) -> str:
    if mode == "rewrite":
        return (
            "Rewrite the question into one concise Wikipedia search query for evidence retrieval. "
            "Preserve named entities and constraints. Do not answer the question. "
            "Return only the rewritten query.\n\n"
            f"Question: {question}"
        )
    if mode == "decompose":
        return (
            "Decompose the question into at most two concise Wikipedia search queries for evidence retrieval. "
            "Preserve named entities and constraints. Do not answer the question. "
            "Return one query per line and no other text.\n\n"
            f"Question: {question}"
        )
    if mode == "hyde":
        return (
            "Write one concise hypothetical evidence passage that would answer the question and contain terms "
            "likely to appear in relevant source documents. Preserve named entities and constraints. "
            "Do not include citations or analysis. Return only the hypothetical evidence passage.\n\n"
            f"Question: {question}"
        )
    if mode == "multi_query":
        return (
            "Generate up to three diverse concise search queries for evidence retrieval. "
            "Preserve named entities and constraints, vary wording, and do not answer the question. "
            "Return one query per line and no other text.\n\n"
            f"Question: {question}"
        )
    raise ValueError(f"Unknown Gemini rewrite mode: {mode}")


def _parse_queries(text: str, mode: str) -> list[str]:
    lines = [line.strip(" -0123456789.\t") for line in text.splitlines()]
    queries = [line for line in lines if line]
    if mode in {"rewrite", "hyde"}:
        return queries[:1] or [text.strip()]
    if mode == "multi_query":
        return queries[:3] or [text.strip()]
    return queries[:2] or [text.strip()]
