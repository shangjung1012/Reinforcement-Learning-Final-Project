from __future__ import annotations

import json
from pathlib import Path

import pytest

from selective_rag_rl.gemini_baseline import GeminiBudgetError, GeminiCache, _parse_queries, _prompt, evaluate_gemini_rewrites


def test_gemini_cache_round_trip(tmp_path: Path) -> None:
    cache = GeminiCache(tmp_path / "rewrites.jsonl")
    cache.set("q1", "rewrite", ["Ada Lovelace Analytical Engine"])

    loaded = GeminiCache(tmp_path / "rewrites.jsonl")

    assert loaded.get("q1", "rewrite") == ["Ada Lovelace Analytical Engine"]


def test_evaluate_gemini_rewrites_writes_summary(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(6)]), encoding="utf-8")

    metadata = evaluate_gemini_rewrites(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=6,
        seed=1,
        rewrite_provider=lambda question, mode: [question] if mode == "rewrite" else [question, question],
    )

    summary = Path(metadata["outputs"]["summary_csv"]).read_text(encoding="utf-8")
    assert "Gemini rewrite-all" in summary
    assert "Gemini decompose" in summary


def test_evaluate_gemini_rewrites_blocks_live_provider_when_budget_is_exceeded(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    data_path.write_text(json.dumps([_example(i) for i in range(6)]), encoding="utf-8")

    with pytest.raises(GeminiBudgetError, match="cache misses"):
        evaluate_gemini_rewrites(
            data_path=data_path,
            output_dir=tmp_path / "outputs",
            num_examples=6,
            seed=1,
            cache_path=tmp_path / "cache.jsonl",
            allow_api=True,
            max_new_calls=0,
        )


def test_evaluate_gemini_rewrites_dry_run_writes_metadata_without_calling_provider(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(6)]), encoding="utf-8")
    calls: list[str] = []

    metadata = evaluate_gemini_rewrites(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=6,
        seed=1,
        cache_path=tmp_path / "cache.jsonl",
        rewrite_provider=lambda question, mode: calls.append(mode) or [question],
        dry_run=True,
    )

    assert calls == []
    assert metadata["dry_run"] is True
    assert metadata["cache_misses"] == 6
    assert Path(metadata["outputs"]["metadata_json"]).exists()


def test_generated_query_modes_have_distinct_prompts_and_parse_limits() -> None:
    assert "hypothetical evidence passage" in _prompt("Who wrote Hamlet?", "hyde")
    assert "three diverse" in _prompt("Who wrote Hamlet?", "multi_query")

    text = "first query\nsecond query\nthird query\nfourth query"
    assert _parse_queries(text, "rewrite") == ["first query"]
    assert _parse_queries(text, "hyde") == ["first query"]
    assert _parse_queries(text, "decompose") == ["first query", "second query"]
    assert _parse_queries(text, "multi_query") == ["first query", "second query", "third query"]


def _example(i: int) -> dict[str, object]:
    return {
        "_id": f"q{i}",
        "answer": f"Entity{i}A",
        "question": f"Which author lived longer, Entity{i}A or Entity{i}B?",
        "supporting_facts": [[f"Entity{i}A", 0], [f"Entity{i}B", 0]],
        "context": [
            [f"Entity{i}A", [f"Entity{i}A was a writer."]],
            [f"Entity{i}B", [f"Entity{i}B was another writer."]],
            [f"Distractor{i}", ["Unrelated passage."]],
        ],
        "type": "comparison",
        "level": "easy",
    }
