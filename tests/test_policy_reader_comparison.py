from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.core.data import Passage, QAExample
from selective_rag_rl.experiments.policy_reader_comparison import (
    parse_top_docs,
    run_policy_reader_comparison,
)


def test_parse_top_docs_strips_empty_values() -> None:
    assert parse_top_docs(" doc-a |  | doc-b | ") == ["doc-a", "doc-b"]


def test_policy_reader_comparison_writes_summary_for_retrieval_sources(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "retrieval_policy_detailed.csv"
    pd.DataFrame(
        [
            _retrieval_row("Vanilla BM25", "q1", "gold | distractor", 1.0, 1.0),
            _retrieval_row("Train-best retrieval action", "q1", "distractor | gold", 1.0, 0.5),
            _retrieval_row("Selective retrieval policy", "q1", "gold", 1.0, 1.0),
            _retrieval_row("Vanilla BM25", "q2", "distractor", 0.0, 0.0),
            _retrieval_row("Train-best retrieval action", "q2", "gold2", 1.0, 1.0),
            _retrieval_row("Selective retrieval policy", "q2", "gold2", 1.0, 1.0),
        ]
    ).to_csv(detailed_csv, index=False)
    examples = [
        QAExample(
            qid="q1",
            question="Who wrote notes for the Analytical Engine?",
            answer="Ada Lovelace",
            passages=[
                Passage("gold", "Ada Lovelace", "Ada Lovelace. Ada Lovelace wrote notes for the Analytical Engine."),
                Passage("distractor", "Noise", "Noise. This passage is unrelated."),
            ],
            gold_doc_ids={"gold"},
            level="test",
            qtype="toy",
        ),
        QAExample(
            qid="q2",
            question="Who worked on COBOL compilers?",
            answer="Grace Hopper",
            passages=[
                Passage("gold2", "Grace Hopper", "Grace Hopper. Grace Hopper worked on COBOL compilers."),
                Passage("distractor", "Noise", "Noise. This passage is unrelated."),
            ],
            gold_doc_ids={"gold2"},
            level="test",
            qtype="toy",
        ),
    ]

    metadata = run_policy_reader_comparison(
        dataset="toy",
        detailed_csv=detailed_csv,
        output_dir=tmp_path / "out",
        readers=["span"],
        examples=examples,
        num_examples=2,
    )

    summary = pd.read_csv(metadata["outputs"]["summary_csv"])
    detailed = pd.read_csv(metadata["outputs"]["detailed_csv"])

    assert set(summary["retrieval_method"]) == {
        "Vanilla BM25",
        "Train-best retrieval action",
        "Selective retrieval policy",
    }
    assert set(summary["reader"]) == {"span"}
    assert set(summary["claim_scope"]) == {"tiny_realdata_policy_reader_diagnostic"}
    assert "missing_doc_count" in detailed.columns
    assert summary.loc[
        summary["retrieval_method"] == "Selective retrieval policy",
        "exact_match",
    ].iloc[0] == 1.0


def test_policy_reader_comparison_counts_missing_doc_ids(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "retrieval_policy_detailed.csv"
    pd.DataFrame(
        [_retrieval_row("Vanilla BM25", "q1", "missing | gold", 1.0, 0.5)]
    ).to_csv(detailed_csv, index=False)
    examples = [
        QAExample(
            qid="q1",
            question="Who wrote notes for the Analytical Engine?",
            answer="Ada Lovelace",
            passages=[
                Passage("gold", "Ada Lovelace", "Ada Lovelace. Ada Lovelace wrote notes for the Analytical Engine."),
            ],
            gold_doc_ids={"gold"},
            level="test",
            qtype="toy",
        )
    ]

    metadata = run_policy_reader_comparison(
        dataset="toy",
        detailed_csv=detailed_csv,
        output_dir=tmp_path / "out",
        readers=["lexical"],
        retrieval_methods=["Vanilla BM25"],
        examples=examples,
        num_examples=1,
    )

    detailed = pd.read_csv(metadata["outputs"]["detailed_csv"])

    assert detailed["missing_doc_count"].iloc[0] == 1


def _retrieval_row(method: str, qid: str, top_docs: str, recall: float, rr: float) -> dict[str, object]:
    return {
        "split": "test",
        "method": method,
        "action": method.lower().replace(" ", "_"),
        "qid": qid,
        "question": "question",
        "recall_at_5": recall,
        "mrr": rr,
        "ndcg_at_5": recall,
        "rewrite_cost": 0.0,
        "retrieval_calls": 1,
        "rewrite_tokens": 4,
        "reward": recall + 0.5 * rr,
        "queries": "question",
        "top_docs": top_docs,
        "gold_docs": "gold",
    }
