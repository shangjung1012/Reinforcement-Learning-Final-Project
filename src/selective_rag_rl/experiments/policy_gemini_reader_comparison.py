from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from selective_rag_rl.core.answer_metrics import exact_match, token_f1
from selective_rag_rl.core.data import Passage, QAExample, load_hotpotqa, load_natural_questions
from selective_rag_rl.experiments.gemini_reader import (
    AnswerProvider,
    GeminiReaderBudgetError,
    GeminiReaderCache,
    VertexGeminiAnswerReader,
)
from selective_rag_rl.experiments.policy_reader_comparison import (
    DEFAULT_RETRIEVAL_METHODS,
    parse_top_docs,
)

CLAIM_SCOPE = "api_pilot_policy_routed_gemini_reader"


@dataclass(frozen=True)
class _EvaluationItem:
    dataset: str
    qid: str
    example: QAExample
    retrieval_method: str
    retrieval_row: pd.Series
    passages: list[Passage]
    top_doc_ids: list[str]
    missing_doc_count: int
    context_hash: str


def run_policy_gemini_reader_comparison(
    *,
    dataset: str,
    detailed_csv: Path,
    output_dir: Path,
    retrieval_methods: list[str] | None = None,
    split: str = "test",
    num_examples: int | None = None,
    seed: int = 42,
    k: int = 5,
    data_path: Path | None = None,
    examples: list[QAExample] | None = None,
    source_num_examples: int = 1000,
    cache_path: Path | None = None,
    answer_provider: AnswerProvider | None = None,
    allow_api: bool = False,
    max_new_calls: int = 0,
    dry_run: bool = False,
    context_fingerprint: Callable[[str, list[Passage]], str] | None = None,
) -> dict[str, object]:
    selected_methods = retrieval_methods or DEFAULT_RETRIEVAL_METHODS
    items = _evaluation_items(
        dataset=dataset,
        detailed_csv=detailed_csv,
        retrieval_methods=selected_methods,
        split=split,
        num_examples=num_examples,
        seed=seed,
        k=k,
        data_path=data_path,
        examples=examples,
        source_num_examples=source_num_examples,
        context_fingerprint=context_fingerprint,
    )

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache = GeminiReaderCache(cache_path or output_dir / "cache" / "policy_gemini_reader_answers.jsonl")
    unique_keys = _unique_context_keys(items)
    cache_hits = sum(cache.get(qid, context_hash) is not None for qid, context_hash in unique_keys)
    cache_misses = len(unique_keys) - cache_hits
    metadata = {
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "k": int(k),
        "num_examples_requested": None if num_examples is None else int(num_examples),
        "num_examples_evaluated": len({item.qid for item in items}),
        "retrieval_methods": selected_methods,
        "reader": "gemini",
        "claim_scope": CLAIM_SCOPE,
        "uses_external_api": not dry_run,
        "cache": str(cache.path),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "estimated_new_calls": int(cache_misses),
        "actual_new_calls": 0,
        "allow_api": bool(allow_api),
        "max_new_calls": int(max_new_calls),
        "dry_run": bool(dry_run),
        "outputs": {"metadata_json": str(results_dir / "policy_gemini_reader_metadata.json")},
    }
    if dry_run:
        Path(metadata["outputs"]["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    if answer_provider is None:
        _check_budget(cache_misses, allow_api=allow_api, max_new_calls=max_new_calls)
    provider = answer_provider or VertexGeminiAnswerReader(Path.cwd()).answer

    rows: list[dict[str, object]] = []
    actual_new_calls = 0
    for item in tqdm(items, desc="policy gemini reader"):
        answer = cache.get(item.qid, item.context_hash)
        if answer is None:
            answer = provider(item.example.question, [(passage.doc_id, passage.text) for passage in item.passages])
            cache.set(item.qid, item.context_hash, answer)
            actual_new_calls += 1
        rows.append(_output_row(item, answer))

    detailed = pd.DataFrame(rows)
    summary = _summary(detailed)
    detailed_csv_out = results_dir / "policy_gemini_reader_detailed.csv"
    summary_csv = results_dir / "policy_gemini_reader_summary.csv"
    summary_json = results_dir / "policy_gemini_reader_summary.json"
    detailed.to_csv(detailed_csv_out, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    metadata["actual_new_calls"] = int(actual_new_calls)
    metadata["outputs"].update(
        {
            "detailed_csv": str(detailed_csv_out),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
        }
    )
    Path(metadata["outputs"]["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _evaluation_items(
    *,
    dataset: str,
    detailed_csv: Path,
    retrieval_methods: list[str],
    split: str,
    num_examples: int | None,
    seed: int,
    k: int,
    data_path: Path | None,
    examples: list[QAExample] | None,
    source_num_examples: int,
    context_fingerprint: Callable[[str, list[Passage]], str] | None,
) -> list[_EvaluationItem]:
    detailed = pd.read_csv(detailed_csv)
    required = {"split", "method", "qid", "top_docs", "recall_at_5", "mrr", "ndcg_at_5"}
    missing = required - set(detailed.columns)
    if missing:
        raise ValueError(f"detailed CSV is missing required columns: {sorted(missing)}")

    rows_for_split = detailed[
        (detailed["split"].astype(str) == split) & (detailed["method"].isin(retrieval_methods))
    ].copy()
    if rows_for_split.empty:
        raise ValueError(f"No rows found for split={split!r} and methods={retrieval_methods}")

    qids = _ordered_qids(rows_for_split)
    if num_examples is not None:
        qids = qids[:num_examples]
    examples_by_qid = _examples_by_qid(
        dataset=dataset,
        data_path=data_path,
        num_examples=max(source_num_examples, len(qids)),
        seed=seed,
        examples=examples,
    )

    fingerprint = context_fingerprint or _context_hash
    items: list[_EvaluationItem] = []
    for qid in qids:
        ex = examples_by_qid.get(str(qid))
        if ex is None:
            continue
        passage_lookup = _passage_lookup(ex.passages)
        for method in retrieval_methods:
            method_rows = rows_for_split[
                (rows_for_split["qid"].astype(str) == str(qid)) & (rows_for_split["method"] == method)
            ]
            if method_rows.empty:
                continue
            retrieval_row = method_rows.iloc[0]
            top_doc_ids = parse_top_docs(retrieval_row["top_docs"])
            passages, missing_count = _passages_for_top_docs(top_doc_ids[:k], passage_lookup)
            items.append(
                _EvaluationItem(
                    dataset=dataset,
                    qid=str(qid),
                    example=ex,
                    retrieval_method=method,
                    retrieval_row=retrieval_row,
                    passages=passages,
                    top_doc_ids=top_doc_ids,
                    missing_doc_count=missing_count,
                    context_hash=fingerprint(str(qid), passages),
                )
            )
    return items


def _check_budget(misses: int, *, allow_api: bool, max_new_calls: int) -> None:
    if misses == 0:
        return
    if not allow_api:
        raise GeminiReaderBudgetError(
            "Policy-routed Gemini reader cache misses require allow_api=True before live API calls"
        )
    if misses > max_new_calls:
        raise GeminiReaderBudgetError(
            f"Policy-routed Gemini reader cache misses ({misses}) exceed max_new_calls ({max_new_calls})"
        )


def _output_row(item: _EvaluationItem, answer: str) -> dict[str, object]:
    cleaned = _clean_answer(answer)
    return {
        "dataset": item.dataset,
        "split": str(item.retrieval_row["split"]),
        "qid": item.example.qid,
        "question": item.example.question,
        "gold_answer": item.example.answer,
        "retrieval_method": item.retrieval_method,
        "retrieval_action": str(item.retrieval_row.get("action", "")),
        "reader": "gemini",
        "predicted_answer": cleaned,
        "exact_match": exact_match(cleaned, item.example.answer),
        "f1": token_f1(cleaned, item.example.answer),
        "recall_at_5": float(item.retrieval_row["recall_at_5"]),
        "mrr": float(item.retrieval_row["mrr"]),
        "ndcg_at_5": float(item.retrieval_row["ndcg_at_5"]),
        "retrieval_calls": float(item.retrieval_row.get("retrieval_calls", 0.0)),
        "retrieval_reward": float(item.retrieval_row.get("reward", 0.0)),
        "top_docs": " | ".join(item.top_doc_ids),
        "gold_docs": str(item.retrieval_row.get("gold_docs", "")),
        "missing_doc_count": int(item.missing_doc_count),
        "claim_scope": CLAIM_SCOPE,
    }


def _summary(detailed: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "retrieval_method",
        "reader",
        "num_examples",
        "exact_match",
        "f1",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
        "retrieval_calls",
        "retrieval_reward",
        "missing_doc_count",
        "claim_scope",
    ]
    if detailed.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        detailed.groupby(["dataset", "retrieval_method", "reader"], sort=False)
        .agg(
            num_examples=("qid", "count"),
            exact_match=("exact_match", "mean"),
            f1=("f1", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            mrr=("mrr", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            retrieval_calls=("retrieval_calls", "mean"),
            retrieval_reward=("retrieval_reward", "mean"),
            missing_doc_count=("missing_doc_count", "sum"),
        )
        .reset_index()
    )
    grouped["claim_scope"] = CLAIM_SCOPE
    return grouped[columns]


def _ordered_qids(rows: pd.DataFrame) -> list[str]:
    qids: list[str] = []
    seen: set[str] = set()
    for qid in rows["qid"].astype(str):
        if qid not in seen:
            seen.add(qid)
            qids.append(qid)
    return qids


def _examples_by_qid(
    *,
    dataset: str,
    data_path: Path | None,
    num_examples: int,
    seed: int,
    examples: list[QAExample] | None,
) -> dict[str, QAExample]:
    loaded = examples if examples is not None else _load_examples(dataset, data_path, num_examples, seed)
    return {str(ex.qid): ex for ex in loaded}


def _load_examples(dataset: str, data_path: Path | None, num_examples: int, seed: int) -> list[QAExample]:
    if dataset == "hotpot":
        path = data_path or Path("data/raw/HotpotQA/hotpot_dev_distractor_v1.json")
        _require_path(path, dataset)
        return load_hotpotqa(path, num_examples=num_examples, seed=seed)
    if dataset == "nq":
        path = data_path or Path("data/raw/natural-questions/default/validation-00000-of-00007.parquet")
        _require_path(path, dataset)
        return load_natural_questions(path, num_examples=num_examples, seed=seed)
    if dataset == "toy":
        raise ValueError("toy examples must be passed directly to run_policy_gemini_reader_comparison")
    raise ValueError("dataset must be one of: hotpot, nq")


def _passage_lookup(passages: list[Passage]) -> dict[str, Passage]:
    lookup: dict[str, Passage] = {}
    for passage in passages:
        lookup[passage.doc_id] = passage
        lookup[html.unescape(passage.doc_id)] = passage
    return lookup


def _passages_for_top_docs(top_doc_ids: list[str], lookup: dict[str, Passage]) -> tuple[list[Passage], int]:
    passages: list[Passage] = []
    missing = 0
    for doc_id in top_doc_ids:
        passage = lookup.get(doc_id) or lookup.get(html.unescape(doc_id))
        if passage is None:
            missing += 1
            continue
        passages.append(passage)
    return passages, missing


def _unique_context_keys(items: list[_EvaluationItem]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.qid, item.context_hash)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _context_hash(qid: str, passages: list[Passage]) -> str:
    payload = {
        "qid": qid,
        "passages": [(passage.doc_id, passage.text) for passage in passages],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _clean_answer(answer: str) -> str:
    cleaned = answer.strip().strip('"').strip("'").strip()
    for prefix in ["Answer:", "answer:"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned.splitlines()[0].strip() if cleaned else ""


def _require_path(path: Path, dataset: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{dataset} data not found at {path}")
