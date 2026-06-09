from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from selective_rag_rl.core.data import Passage, QAExample
from selective_rag_rl.experiments.gemini_reader import GeminiReaderBudgetError
from selective_rag_rl.experiments.policy_gemini_reader_comparison import (
    run_policy_gemini_reader_comparison,
)


def test_policy_gemini_reader_dry_run_estimates_unique_misses_without_provider(tmp_path: Path) -> None:
    calls: list[str] = []

    metadata = run_policy_gemini_reader_comparison(
        dataset="toy",
        detailed_csv=_detailed_csv(tmp_path, same_context=True),
        output_dir=tmp_path / "outputs",
        examples=_examples(),
        num_examples=2,
        cache_path=tmp_path / "cache.jsonl",
        answer_provider=lambda question, passages: calls.append(question) or "wrong",
        dry_run=True,
    )

    assert calls == []
    assert metadata["dry_run"] is True
    assert metadata["cache_misses"] == 2
    assert metadata["estimated_new_calls"] == 2
    assert Path(metadata["outputs"]["metadata_json"]).exists()


def test_policy_gemini_reader_blocks_when_budget_exceeded(tmp_path: Path) -> None:
    with pytest.raises(GeminiReaderBudgetError, match="exceed"):
        run_policy_gemini_reader_comparison(
            dataset="toy",
            detailed_csv=_detailed_csv(tmp_path, same_context=False),
            output_dir=tmp_path / "outputs",
            examples=_examples(),
            num_examples=2,
            cache_path=tmp_path / "cache.jsonl",
            allow_api=True,
            max_new_calls=0,
        )


def test_policy_gemini_reader_writes_summary_for_retrieval_methods(tmp_path: Path) -> None:
    provider_calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_provider(question: str, passages: list[tuple[str, str]]) -> str:
        provider_calls.append((question, tuple(doc_id for doc_id, _text in passages)))
        return passages[0][1].split(".", maxsplit=1)[0]

    metadata = run_policy_gemini_reader_comparison(
        dataset="toy",
        detailed_csv=_detailed_csv(tmp_path, same_context=False),
        output_dir=tmp_path / "outputs",
        examples=_examples(),
        num_examples=2,
        cache_path=tmp_path / "cache.jsonl",
        answer_provider=fake_provider,
        allow_api=True,
        max_new_calls=6,
    )

    summary = pd.read_csv(metadata["outputs"]["summary_csv"])
    detailed = pd.read_csv(metadata["outputs"]["detailed_csv"])

    assert metadata["actual_new_calls"] == 4
    assert len(provider_calls) == 4
    assert set(summary["retrieval_method"]) == {
        "Vanilla BM25",
        "Train-best retrieval action",
        "Selective retrieval policy",
    }
    assert set(summary["reader"]) == {"gemini"}
    assert summary["exact_match"].mean() == 1.0
    assert set(detailed["claim_scope"]) == {"api_pilot_policy_routed_gemini_reader"}


def test_policy_gemini_reader_reuses_cache_for_identical_contexts(tmp_path: Path) -> None:
    provider_calls: list[str] = []

    metadata = run_policy_gemini_reader_comparison(
        dataset="toy",
        detailed_csv=_detailed_csv(tmp_path, same_context=True),
        output_dir=tmp_path / "outputs",
        examples=_examples(),
        num_examples=2,
        cache_path=tmp_path / "cache.jsonl",
        answer_provider=lambda question, passages: provider_calls.append(question) or passages[0][1].split(".")[0],
        allow_api=True,
        max_new_calls=2,
    )

    assert metadata["actual_new_calls"] == 2
    assert len(provider_calls) == 2
    summary = pd.read_csv(metadata["outputs"]["summary_csv"])
    assert summary["exact_match"].mean() == 1.0


def _examples() -> list[QAExample]:
    return [
        QAExample(
            qid="q1",
            question="Who wrote notes for the Analytical Engine?",
            answer="Ada Lovelace",
            passages=[
                Passage("ada", "Ada", "Ada Lovelace. Ada wrote notes for the Analytical Engine."),
                Passage("ada-alt", "Ada alt", "Ada Lovelace. The answer is still Ada Lovelace."),
                Passage("noise", "Noise", "Wrong person."),
            ],
            gold_doc_ids={"ada"},
            level="toy",
            qtype="policy-gemini",
        ),
        QAExample(
            qid="q2",
            question="Who worked on COBOL compilers?",
            answer="Grace Hopper",
            passages=[
                Passage("grace", "Grace", "Grace Hopper. Grace worked on COBOL compilers."),
                Passage("grace-alt", "Grace alt", "Grace Hopper. The answer is still Grace Hopper."),
                Passage("noise2", "Noise", "Wrong person."),
            ],
            gold_doc_ids={"grace"},
            level="toy",
            qtype="policy-gemini",
        ),
    ]


def _detailed_csv(tmp_path: Path, *, same_context: bool) -> Path:
    rows = []
    method_docs = {
        "Vanilla BM25": {"q1": "ada | noise", "q2": "grace | noise2"},
        "Train-best retrieval action": {"q1": "ada | noise", "q2": "grace | noise2"}
        if same_context
        else {"q1": "ada-alt | noise", "q2": "grace-alt | noise2"},
        "Selective retrieval policy": {"q1": "ada | noise", "q2": "grace | noise2"},
    }
    for qid in ["q1", "q2"]:
        for method, docs_by_qid in method_docs.items():
            rows.append(
                {
                    "split": "test",
                    "method": method,
                    "action": "bm25_keep",
                    "qid": qid,
                    "top_docs": docs_by_qid[qid],
                    "gold_docs": "ada" if qid == "q1" else "grace",
                    "recall_at_5": 1.0,
                    "mrr": 1.0,
                    "ndcg_at_5": 1.0,
                    "retrieval_calls": 1.0,
                    "reward": 1.5,
                }
            )
    path = tmp_path / "detailed.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
