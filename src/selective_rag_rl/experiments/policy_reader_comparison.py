from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.core.answer_metrics import exact_match, token_f1
from selective_rag_rl.core.data import Passage, QAExample, load_hotpotqa, load_natural_questions
from selective_rag_rl.core.reader import AnswerTypeHeuristicReader, LexicalOverlapReader, SpanHeuristicReader

DEFAULT_RETRIEVAL_METHODS = [
    "Vanilla BM25",
    "Train-best retrieval action",
    "Selective retrieval policy",
]
DEFAULT_READERS = ["lexical", "span", "answer_type"]
CLAIM_SCOPE = "tiny_realdata_policy_reader_diagnostic"


def run_policy_reader_comparison(
    *,
    dataset: str,
    detailed_csv: Path,
    output_dir: Path,
    readers: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    split: str = "test",
    num_examples: int | None = None,
    seed: int = 42,
    k: int = 5,
    data_path: Path | None = None,
    examples: list[QAExample] | None = None,
    source_num_examples: int = 1000,
) -> dict[str, object]:
    selected_readers = readers or DEFAULT_READERS
    selected_methods = retrieval_methods or DEFAULT_RETRIEVAL_METHODS
    _validate_readers(selected_readers)

    detailed = pd.read_csv(detailed_csv)
    required = {"split", "method", "qid", "top_docs", "recall_at_5", "mrr", "ndcg_at_5"}
    missing = required - set(detailed.columns)
    if missing:
        raise ValueError(f"detailed CSV is missing required columns: {sorted(missing)}")

    rows_for_split = detailed[
        (detailed["split"].astype(str) == split) & (detailed["method"].isin(selected_methods))
    ].copy()
    if rows_for_split.empty:
        raise ValueError(f"No rows found for split={split!r} and methods={selected_methods}")

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

    readers_by_name = {name: _reader(name) for name in selected_readers}
    output_rows: list[dict[str, object]] = []
    for qid in qids:
        ex = examples_by_qid.get(str(qid))
        if ex is None:
            continue
        passage_lookup = _passage_lookup(ex.passages)
        for method in selected_methods:
            method_rows = rows_for_split[
                (rows_for_split["qid"].astype(str) == str(qid)) & (rows_for_split["method"] == method)
            ]
            if method_rows.empty:
                continue
            retrieval_row = method_rows.iloc[0]
            top_doc_ids = parse_top_docs(retrieval_row["top_docs"])
            passages, missing_count = _passages_for_top_docs(top_doc_ids, passage_lookup)
            for reader_name, reader in readers_by_name.items():
                prediction = reader.predict(ex.question, passages)
                output_rows.append(
                    _output_row(
                        dataset=dataset,
                        split=split,
                        retrieval_row=retrieval_row,
                        reader_name=reader_name,
                        ex=ex,
                        prediction_answer=prediction.answer,
                        reader_score=prediction.score,
                        reader_passage_id=prediction.passage_id,
                        top_doc_ids=top_doc_ids,
                        missing_doc_count=missing_count,
                    )
                )

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed_out = pd.DataFrame(output_rows)
    summary = _summary(detailed_out)
    detailed_csv_out = results_dir / "policy_reader_comparison_detailed.csv"
    summary_csv = results_dir / "policy_reader_comparison_summary.csv"
    summary_json = results_dir / "policy_reader_comparison_summary.json"
    metadata_json = results_dir / "policy_reader_comparison_metadata.json"
    detailed_out.to_csv(detailed_csv_out, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    metadata = {
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "k": int(k),
        "num_examples_requested": None if num_examples is None else int(num_examples),
        "num_examples_evaluated": int(detailed_out["qid"].nunique()) if len(detailed_out) else 0,
        "retrieval_methods": selected_methods,
        "readers": selected_readers,
        "claim_scope": CLAIM_SCOPE,
        "uses_external_api": False,
        "outputs": {
            "detailed_csv": str(detailed_csv_out),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "metadata_json": str(metadata_json),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_top_docs(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [doc_id.strip() for doc_id in str(value).split("|") if doc_id.strip()]


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
        raise ValueError("toy examples must be passed directly to run_policy_reader_comparison")
    raise ValueError("dataset must be one of: hotpot, nq")


def _reader(reader: str) -> LexicalOverlapReader | SpanHeuristicReader | AnswerTypeHeuristicReader:
    if reader == "lexical":
        return LexicalOverlapReader()
    if reader == "span":
        return SpanHeuristicReader()
    if reader == "answer_type":
        return AnswerTypeHeuristicReader()
    raise ValueError("reader must be one of: lexical, span, answer_type")


def _validate_readers(readers: list[str]) -> None:
    unknown = sorted(set(readers) - set(DEFAULT_READERS))
    if unknown:
        raise ValueError(f"Unknown readers: {unknown}")


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


def _output_row(
    *,
    dataset: str,
    split: str,
    retrieval_row: pd.Series,
    reader_name: str,
    ex: QAExample,
    prediction_answer: str,
    reader_score: float,
    reader_passage_id: str | None,
    top_doc_ids: list[str],
    missing_doc_count: int,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "split": split,
        "qid": ex.qid,
        "question": ex.question,
        "gold_answer": ex.answer,
        "retrieval_method": str(retrieval_row["method"]),
        "retrieval_action": str(retrieval_row.get("action", "")),
        "reader": reader_name,
        "predicted_answer": prediction_answer,
        "reader_score": float(reader_score),
        "reader_passage_id": "" if reader_passage_id is None else reader_passage_id,
        "exact_match": exact_match(prediction_answer, ex.answer),
        "f1": token_f1(prediction_answer, ex.answer),
        "recall_at_5": float(retrieval_row["recall_at_5"]),
        "mrr": float(retrieval_row["mrr"]),
        "ndcg_at_5": float(retrieval_row["ndcg_at_5"]),
        "retrieval_calls": float(retrieval_row.get("retrieval_calls", 0.0)),
        "retrieval_reward": float(retrieval_row.get("reward", 0.0)),
        "top_docs": " | ".join(top_doc_ids),
        "gold_docs": str(retrieval_row.get("gold_docs", "")),
        "missing_doc_count": int(missing_doc_count),
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


def _require_path(path: Path, dataset: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{dataset} data not found at {path}")
