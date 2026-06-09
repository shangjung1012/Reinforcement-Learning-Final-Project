from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from selective_rag_rl.core.answer_metrics import exact_match, token_f1
from selective_rag_rl.core.data import Passage, QAExample, load_hotpotqa, load_natural_questions
from selective_rag_rl.core.metrics import mrr, ndcg_at_k, recall_at_k
from selective_rag_rl.core.reader import AnswerTypeHeuristicReader, LexicalOverlapReader, SpanHeuristicReader
from selective_rag_rl.core.retriever import BM25Retriever, RetrievalResult

AnswerProvider = Callable[[str, list[tuple[str, str]]], str]
ContextFingerprint = Callable[[str, list[Passage]], str]


class GeminiReaderBudgetError(RuntimeError):
    pass


class GeminiReaderCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[tuple[str, str], str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.rows[(row["qid"], row["context_hash"])] = str(row["answer"])

    def get(self, qid: str, context_hash: str) -> str | None:
        return self.rows.get((qid, context_hash))

    def set(self, qid: str, context_hash: str, answer: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows[(qid, context_hash)] = answer
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"qid": qid, "context_hash": context_hash, "answer": answer},
                    ensure_ascii=False,
                )
                + "\n"
            )


class VertexGeminiAnswerReader:
    def __init__(self, project_root: Path, model: str | None = None, temperature: float = 0.0) -> None:
        load_dotenv(project_root / ".env")
        project = os.environ["GOOGLE_CLOUD_PROJECT"]
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        credentials = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).expanduser()
        if not credentials.is_absolute():
            credentials = project_root / credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = float(temperature)
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def answer(self, question: str, passages: list[tuple[str, str]]) -> str:
        prompt = build_reader_prompt(question, passages)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        return _clean_answer(response.text or "")


def evaluate_gemini_reader(
    *,
    dataset: str,
    output_dir: Path,
    num_examples: int = 40,
    seed: int = 42,
    k: int = 5,
    data_path: Path | None = None,
    cache_path: Path | None = None,
    answer_provider: AnswerProvider | None = None,
    allow_api: bool = False,
    max_new_calls: int = 0,
    dry_run: bool = False,
    context_fingerprint: ContextFingerprint | None = None,
) -> dict[str, object]:
    examples = _load_examples(dataset, num_examples, seed, data_path)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache = GeminiReaderCache(cache_path or output_dir / "cache" / "gemini_reader_answers.jsonl")
    retrieved = [_retrieved_passages(ex, k) for ex in examples]
    fingerprint = context_fingerprint or _context_hash
    fingerprints = [fingerprint(ex.qid, passages) for ex, passages in zip(examples, retrieved)]
    cache_hits = sum(cache.get(ex.qid, context_hash) is not None for ex, context_hash in zip(examples, fingerprints))
    cache_misses = len(examples) - cache_hits
    metadata = {
        "dataset": dataset,
        "num_examples": len(examples),
        "seed": int(seed),
        "k": int(k),
        "cache": str(cache.path),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "allow_api": bool(allow_api),
        "max_new_calls": int(max_new_calls),
        "dry_run": bool(dry_run),
        "outputs": {"metadata_json": str(results_dir / "gemini_reader_metadata.json")},
    }
    if dry_run:
        Path(metadata["outputs"]["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    if answer_provider is None:
        _check_budget(cache_misses, allow_api=allow_api, max_new_calls=max_new_calls)
    provider = answer_provider or VertexGeminiAnswerReader(Path.cwd()).answer

    rows: list[dict[str, object]] = []
    for ex, passages, context_hash in tqdm(
        list(zip(examples, retrieved, fingerprints)),
        desc="gemini reader eval",
    ):
        rows.extend(_baseline_rows(dataset, ex, passages, k))
        answer = cache.get(ex.qid, context_hash)
        if answer is None:
            answer = provider(ex.question, [(passage.doc_id, passage.text) for passage in passages])
            cache.set(ex.qid, context_hash, answer)
        rows.append(_reader_row(dataset, "Gemini reader", "gemini", ex, answer, passages, k))

    detailed = pd.DataFrame(rows)
    summary = _summary(detailed)
    detailed_csv = results_dir / "gemini_reader_detailed.csv"
    summary_csv = results_dir / "gemini_reader_summary.csv"
    summary_json = results_dir / "gemini_reader_summary.json"
    detailed.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    metadata["outputs"].update(
        {
            "detailed_csv": str(detailed_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
        }
    )
    Path(metadata["outputs"]["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_reader_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    context_lines = []
    for idx, (doc_id, text) in enumerate(passages, start=1):
        clipped = " ".join(text.split())[:1600]
        context_lines.append(f"[{idx}] {doc_id}: {clipped}")
    context = "\n".join(context_lines)
    return (
        "Answer the question using only the retrieved passages below. "
        "Return a concise answer string, not an explanation. "
        "If the answer is not supported by the passages, return unknown.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved passages:\n{context}\n\n"
        "Answer:"
    )


def _load_examples(dataset: str, num_examples: int, seed: int, data_path: Path | None) -> list[QAExample]:
    if dataset == "toy":
        return _toy_examples(num_examples)
    if dataset == "hotpot":
        path = data_path or Path("data/raw/HotpotQA/hotpot_dev_distractor_v1.json")
        _require_path(path, dataset)
        return load_hotpotqa(path, num_examples=num_examples, seed=seed)
    if dataset == "nq":
        path = data_path or Path("data/raw/natural-questions/default/validation-00000-of-00007.parquet")
        _require_path(path, dataset)
        return load_natural_questions(path, num_examples=num_examples, seed=seed)
    raise ValueError("dataset must be one of: toy, hotpot, nq")


def _retrieved_passages(ex: QAExample, k: int) -> list[Passage]:
    retriever = BM25Retriever(ex.passages)
    results = retriever.search(ex.question, k=k)
    passages_by_id = {passage.doc_id: passage for passage in ex.passages}
    return [passages_by_id[result.doc_id] for result in results if result.doc_id in passages_by_id]


def _baseline_rows(dataset: str, ex: QAExample, passages: list[Passage], k: int) -> list[dict[str, object]]:
    readers = [
        ("Lexical reader", "lexical", LexicalOverlapReader()),
        ("Span reader", "span", SpanHeuristicReader()),
        ("Answer-type reader", "answer_type", AnswerTypeHeuristicReader()),
    ]
    rows = []
    for method, reader_name, reader in readers:
        prediction = reader.predict(ex.question, passages)
        rows.append(_reader_row(dataset, method, reader_name, ex, prediction.answer, passages, k))
    return rows


def _reader_row(
    dataset: str,
    method: str,
    reader: str,
    ex: QAExample,
    answer: str,
    passages: list[Passage],
    k: int,
) -> dict[str, object]:
    top_doc_ids = [passage.doc_id for passage in passages]
    result_like = [
        RetrievalResult(doc_id=doc_id, score=float(len(top_doc_ids) - idx), rank=idx + 1)
        for idx, doc_id in enumerate(top_doc_ids)
    ]
    return {
        "dataset": dataset,
        "method": method,
        "reader": reader,
        "qid": ex.qid,
        "question": ex.question,
        "gold_answer": ex.answer,
        "predicted_answer": _clean_answer(answer),
        "exact_match": exact_match(_clean_answer(answer), ex.answer),
        "f1": token_f1(_clean_answer(answer), ex.answer),
        "recall_at_5": recall_at_k(result_like, ex.gold_doc_ids, k),
        "mrr": mrr(result_like, ex.gold_doc_ids),
        "ndcg_at_5": ndcg_at_k(result_like, ex.gold_doc_ids, k),
        "top_docs": " | ".join(top_doc_ids),
        "gold_docs": " | ".join(sorted(ex.gold_doc_ids)),
    }


def _summary(detailed: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        detailed.groupby(["dataset", "reader", "method"], sort=False)
        .agg(
            num_examples=("qid", "count"),
            exact_match=("exact_match", "mean"),
            f1=("f1", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            mrr=("mrr", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
        )
        .reset_index()
    )
    grouped["claim_scope"] = "api_pilot_reader_check"
    return grouped


def _check_budget(misses: int, *, allow_api: bool, max_new_calls: int) -> None:
    if misses == 0:
        return
    if not allow_api:
        raise GeminiReaderBudgetError("Gemini reader cache misses require allow_api=True before live API calls")
    if misses > max_new_calls:
        raise GeminiReaderBudgetError(
            f"Gemini reader cache misses ({misses}) exceed max_new_calls ({max_new_calls})"
        )


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


def _toy_examples(num_examples: int) -> list[QAExample]:
    people = [
        ("ada", "Ada Lovelace", "wrote notes for the Analytical Engine"),
        ("grace", "Grace Hopper", "worked on COBOL compilers"),
        ("alan", "Alan Turing", "studied computation and code breaking"),
        ("katherine", "Katherine Johnson", "calculated orbital trajectories"),
    ]
    examples = []
    for idx in range(num_examples):
        key, answer, clue = people[idx % len(people)]
        qid = f"toy-{idx}"
        gold = f"{key}-{idx}"
        passages = [
            Passage(gold, answer, f"{answer}. {answer} {clue}."),
            Passage(f"noise-{idx}", "Noise", f"Noise. This unrelated passage mentions topic {idx}."),
        ]
        examples.append(
            QAExample(
                qid=qid,
                question=f"Who {clue}?",
                answer=answer,
                passages=passages,
                gold_doc_ids={gold},
                level="toy",
                qtype="gemini-reader-smoke",
            )
        )
    return examples
