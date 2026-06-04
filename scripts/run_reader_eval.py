from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.core.answer_metrics import exact_match, token_f1
from selective_rag_rl.core.data import Passage, QAExample, load_hotpotqa, load_natural_questions
from selective_rag_rl.core.metrics import mrr, ndcg_at_k, recall_at_k
from selective_rag_rl.core.reader import LexicalOverlapReader
from selective_rag_rl.core.retriever import BM25Retriever


def run_reader_eval(
    dataset: str,
    output_dir: Path,
    num_examples: int = 10,
    seed: int = 42,
    k: int = 5,
    data_path: Path | None = None,
    reader: str = "lexical",
    retriever: str = "bm25",
) -> dict[str, object]:
    if reader != "lexical":
        raise ValueError("Only the deterministic lexical reader is available by default")
    if retriever != "bm25":
        raise ValueError("Only BM25 retrieval is available in the lightweight reader eval")

    examples = _load_examples(dataset, num_examples, seed, data_path)
    lexical_reader = LexicalOverlapReader()
    rows = []
    for ex in examples:
        bm25 = BM25Retriever(ex.passages)
        results = bm25.search(ex.question, k=k)
        passages_by_id = {passage.doc_id: passage for passage in ex.passages}
        retrieved_passages = [passages_by_id[result.doc_id] for result in results if result.doc_id in passages_by_id]
        prediction = lexical_reader.predict(ex.question, retrieved_passages)
        rows.append(
            {
                "dataset": dataset,
                "qid": ex.qid,
                "question": ex.question,
                "gold_answer": ex.answer,
                "predicted_answer": prediction.answer,
                "reader_score": prediction.score,
                "reader_passage_id": prediction.passage_id,
                "exact_match": exact_match(prediction.answer, ex.answer),
                "f1": token_f1(prediction.answer, ex.answer),
                "recall_at_5": recall_at_k(results, ex.gold_doc_ids, k),
                "mrr": mrr(results, ex.gold_doc_ids),
                "ndcg_at_5": ndcg_at_k(results, ex.gold_doc_ids, k),
                "top_docs": " | ".join(result.doc_id for result in results),
                "gold_docs": " | ".join(sorted(ex.gold_doc_ids)),
            }
        )

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = results_dir / "reader_eval_detailed.csv"
    summary_csv = results_dir / "reader_eval_summary.csv"
    summary_json = results_dir / "reader_eval_summary.json"
    detailed = pd.DataFrame(rows)
    detailed.to_csv(detailed_csv, index=False)
    summary = _summary_row(dataset, reader, retriever, seed, k, detailed)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = {
        **summary,
        "outputs": {
            "detailed_csv": str(detailed_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
        },
    }
    return metadata


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
                qtype="reader-smoke",
            )
        )
    return examples


def _summary_row(
    dataset: str,
    reader: str,
    retriever: str,
    seed: int,
    k: int,
    detailed: pd.DataFrame,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "reader": reader,
        "retriever": retriever,
        "seed": seed,
        "k": k,
        "num_examples": int(len(detailed)),
        "exact_match": float(detailed["exact_match"].mean()) if len(detailed) else 0.0,
        "f1": float(detailed["f1"].mean()) if len(detailed) else 0.0,
        "recall_at_5": float(detailed["recall_at_5"].mean()) if len(detailed) else 0.0,
        "mrr": float(detailed["mrr"].mean()) if len(detailed) else 0.0,
        "ndcg_at_5": float(detailed["ndcg_at_5"].mean()) if len(detailed) else 0.0,
        "claim_scope": "smoke_downstream_reader_check",
    }


def _require_path(path: Path, dataset: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset} data not found at {path}. Use --dataset toy for raw-data-free smoke evaluation "
            "or download raw data into the README layout."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight reader EM/F1 evaluation.")
    parser.add_argument("--dataset", choices=["toy", "hotpot", "nq"], default="toy")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_reader_smoke"))
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    metadata = run_reader_eval(
        dataset=args.dataset,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        seed=args.seed,
        k=args.k,
    )
    print(json.dumps(metadata, indent=2))
    print(pd.read_csv(metadata["outputs"]["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
