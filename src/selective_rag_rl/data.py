from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    qid: str
    question: str
    answer: str
    passages: list[Passage]
    gold_doc_ids: set[str]
    level: str
    qtype: str


def load_hotpotqa(path: Path, num_examples: int, seed: int) -> list[QAExample]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rng = random.Random(seed)
    indices = list(range(len(raw)))
    rng.shuffle(indices)
    examples: list[QAExample] = []

    for idx in indices:
        item = raw[idx]
        passages: list[Passage] = []
        for title, sentences in item["context"]:
            text = " ".join(s.strip() for s in sentences if s.strip())
            if text:
                passages.append(Passage(doc_id=title, title=title, text=f"{title}. {text}"))

        gold_titles = {title for title, _sent_id in item["supporting_facts"]}
        gold_doc_ids = {p.doc_id for p in passages if p.title in gold_titles}
        if not passages or not gold_doc_ids:
            continue

        examples.append(
            QAExample(
                qid=item["_id"],
                question=item["question"],
                answer=item["answer"],
                passages=passages,
                gold_doc_ids=gold_doc_ids,
                level=item.get("level", "unknown"),
                qtype=item.get("type", "unknown"),
            )
        )
        if len(examples) >= num_examples:
            break

    return examples


def load_natural_questions(
    path: Path,
    num_examples: int,
    seed: int,
    pool_size: int = 50,
    max_document_tokens: int = 220,
) -> list[QAExample]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "document", "question", "annotations"])
    rows = [_nq_row_to_passage(row, max_document_tokens) for row in table.to_pylist()]
    rows = [row for row in rows if row is not None]

    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = rows[:num_examples]
    if not selected:
        return []

    pool = rows[: max(pool_size, len(selected))]
    examples: list[QAExample] = []
    for item in selected:
        negatives = [row["passage"] for row in pool if row["passage"].doc_id != item["passage"].doc_id]
        rng.shuffle(negatives)
        passages = [item["passage"], *negatives[: max(0, pool_size - 1)]]
        rng.shuffle(passages)
        examples.append(
            QAExample(
                qid=item["qid"],
                question=item["question"],
                answer=item["answer"],
                passages=passages,
                gold_doc_ids={item["passage"].doc_id},
                level="single-hop",
                qtype="natural-questions",
            )
        )
    return examples


def load_beir_scifact(
    path: Path,
    num_examples: int,
    seed: int,
    split: str = "test",
    pool_size: int = 100,
    full_corpus: bool = False,
) -> list[QAExample]:
    return load_beir_dataset(
        path=path,
        num_examples=num_examples,
        seed=seed,
        split=split,
        pool_size=pool_size,
        full_corpus=full_corpus,
        qtype="beir-scifact",
    )


def load_beir_dataset(
    path: Path,
    num_examples: int,
    seed: int,
    split: str = "test",
    pool_size: int = 100,
    full_corpus: bool = False,
    qtype: str = "beir",
) -> list[QAExample]:
    corpus = _read_jsonl(path / "corpus.jsonl")
    queries = _read_jsonl(path / "queries.jsonl")
    qrels = _read_qrels(path / "qrels" / f"{split}.tsv")
    passages_by_id = {
        str(row["_id"]): Passage(
            doc_id=str(row["_id"]),
            title=str(row.get("title") or ""),
            text=f"{row.get('title') or ''}. {row.get('text') or ''}".strip(),
        )
        for row in corpus
    }
    queries_by_id = {str(row["_id"]): str(row.get("text") or "").strip() for row in queries}

    rng = random.Random(seed)
    available_doc_ids = set(passages_by_id)
    qids = [qid for qid in qrels if qid in queries_by_id and qrels[qid] <= available_doc_ids]
    rng.shuffle(qids)
    all_passages = list(passages_by_id.values())
    examples: list[QAExample] = []
    for qid in qids:
        gold_doc_ids = qrels[qid]
        if full_corpus:
            passages = all_passages
        else:
            gold = [passages_by_id[doc_id] for doc_id in sorted(gold_doc_ids)]
            negatives = [p for p in all_passages if p.doc_id not in gold_doc_ids]
            rng.shuffle(negatives)
            passages = [*gold, *negatives[: max(0, pool_size - len(gold))]]
            rng.shuffle(passages)
        examples.append(
            QAExample(
                qid=qid,
                question=queries_by_id[qid],
                answer="",
                passages=passages,
                gold_doc_ids=gold_doc_ids,
                level=split,
                qtype=qtype,
            )
        )
        if len(examples) >= num_examples:
            break
    return examples


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        next(f, "")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, doc_id, score = parts
            if qid == "query-id" or not qid or not doc_id:
                continue
            if int(score) <= 0:
                continue
            qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def _nq_row_to_passage(row: dict[str, Any], max_document_tokens: int) -> dict[str, Any] | None:
    question = ((row.get("question") or {}).get("text") or "").strip()
    document = row.get("document") or {}
    title = (document.get("title") or "").strip()
    qid = str(row.get("id") or "").strip()
    if not question or not title or not qid:
        return None

    tokens_block = document.get("tokens") or {}
    tokens = tokens_block.get("token") or []
    is_html = tokens_block.get("is_html") or [False] * len(tokens)
    visible_tokens = [
        token
        for token, html in zip(tokens, is_html, strict=False)
        if token and not html and not str(token).startswith("<")
    ][:max_document_tokens]
    if not visible_tokens:
        return None

    answer = _nq_answer_text(row.get("annotations") or {})
    passage = Passage(
        doc_id=f"nq:{qid}",
        title=title,
        text=f"{title}. {' '.join(visible_tokens)}",
    )
    return {"qid": qid, "question": question, "answer": answer, "passage": passage}


def _nq_answer_text(annotations: dict[str, Any]) -> str:
    short_answers = annotations.get("short_answers") or []
    for answer_group in short_answers:
        texts = answer_group.get("text") or []
        if texts:
            return str(texts[0])
    yes_no = annotations.get("yes_no_answer") or []
    if yes_no and yes_no[0] in {0, 1}:
        return "yes" if yes_no[0] == 1 else "no"
    return ""


def split_examples(
    examples: list[QAExample], train_fraction: float = 0.6
) -> tuple[list[QAExample], list[QAExample]]:
    split = int(len(examples) * train_fraction)
    return examples[:split], examples[split:]
