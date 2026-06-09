from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from selective_rag_rl.experiments.gemini_reader import (
    GeminiReaderBudgetError,
    GeminiReaderCache,
    evaluate_gemini_reader,
)


def test_gemini_reader_cache_round_trip(tmp_path: Path) -> None:
    cache = GeminiReaderCache(tmp_path / "answers.jsonl")
    cache.set("q1", "abc123", "Ada Lovelace")

    loaded = GeminiReaderCache(tmp_path / "answers.jsonl")

    assert loaded.get("q1", "abc123") == "Ada Lovelace"


def test_gemini_reader_dry_run_writes_metadata_without_calling_provider(tmp_path: Path) -> None:
    calls: list[str] = []

    metadata = evaluate_gemini_reader(
        dataset="toy",
        output_dir=tmp_path / "outputs",
        num_examples=4,
        cache_path=tmp_path / "cache.jsonl",
        answer_provider=lambda question, passages: calls.append(question) or "wrong",
        dry_run=True,
    )

    assert calls == []
    assert metadata["dry_run"] is True
    assert metadata["cache_misses"] == 4
    assert Path(metadata["outputs"]["metadata_json"]).exists()


def test_gemini_reader_blocks_when_budget_exceeded(tmp_path: Path) -> None:
    with pytest.raises(GeminiReaderBudgetError, match="cache misses"):
        evaluate_gemini_reader(
            dataset="toy",
            output_dir=tmp_path / "outputs",
            num_examples=4,
            cache_path=tmp_path / "cache.jsonl",
            allow_api=True,
            max_new_calls=0,
        )


def test_gemini_reader_writes_baseline_and_gemini_summary(tmp_path: Path) -> None:
    provider_calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_provider(question: str, passages: list[tuple[str, str]]) -> str:
        provider_calls.append((question, tuple(doc_id for doc_id, _text in passages)))
        return passages[0][1].split(".", maxsplit=1)[0]

    metadata = evaluate_gemini_reader(
        dataset="toy",
        output_dir=tmp_path / "outputs",
        num_examples=4,
        cache_path=tmp_path / "cache.jsonl",
        answer_provider=fake_provider,
        allow_api=True,
        max_new_calls=4,
    )

    summary = pd.read_csv(metadata["outputs"]["summary_csv"])
    detailed = pd.read_csv(metadata["outputs"]["detailed_csv"])

    assert set(summary["reader"]) == {"lexical", "span", "answer_type", "gemini"}
    assert "Gemini reader" in set(detailed["method"])
    assert summary.loc[summary["reader"] == "gemini", "exact_match"].iloc[0] == 1.0
    assert len(provider_calls) == 4
    assert all(len(call) == 2 for call in provider_calls)


def test_gemini_reader_reuses_cache_without_provider(tmp_path: Path) -> None:
    cache = GeminiReaderCache(tmp_path / "cache.jsonl")
    for idx, answer in enumerate(["Ada Lovelace", "Grace Hopper", "Alan Turing", "Katherine Johnson"]):
        cache.set(f"toy-{idx}", "cached", answer)

    metadata = evaluate_gemini_reader(
        dataset="toy",
        output_dir=tmp_path / "outputs",
        num_examples=4,
        cache_path=tmp_path / "cache.jsonl",
        context_fingerprint=lambda _qid, _passages: "cached",
    )

    summary = pd.read_csv(metadata["outputs"]["summary_csv"])

    assert metadata["cache_hits"] == 4
    assert metadata["cache_misses"] == 0
    assert summary.loc[summary["reader"] == "gemini", "exact_match"].iloc[0] == 1.0
