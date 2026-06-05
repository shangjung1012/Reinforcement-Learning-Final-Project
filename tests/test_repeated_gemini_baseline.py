from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from selective_rag_rl.experiments.gemini_baseline import GeminiBudgetError


def test_repeated_gemini_dry_run_estimates_unique_cache_misses(tmp_path: Path) -> None:
    run_repeated_gemini_baseline = _load_run_repeated_gemini_baseline()
    data_path = tmp_path / "hotpot.json"
    data_path.write_text(json.dumps([_example(i) for i in range(8)]), encoding="utf-8")

    metadata = run_repeated_gemini_baseline(
        data_path=data_path,
        output_dir=tmp_path / "outputs",
        seeds=[1, 2],
        num_examples=8,
        cache_path=tmp_path / "cache.jsonl",
        dry_run=True,
        max_new_calls=999,
    )

    assert metadata["dry_run"] is True
    assert metadata["estimated_new_calls"] > 0
    assert Path(metadata["outputs"]["metadata_json"]).exists()


def test_repeated_gemini_blocks_when_budget_exceeded(tmp_path: Path) -> None:
    run_repeated_gemini_baseline = _load_run_repeated_gemini_baseline()
    data_path = tmp_path / "hotpot.json"
    data_path.write_text(json.dumps([_example(i) for i in range(8)]), encoding="utf-8")

    with pytest.raises(GeminiBudgetError, match="exceed"):
        run_repeated_gemini_baseline(
            data_path=data_path,
            output_dir=tmp_path / "outputs",
            seeds=[1, 2],
            num_examples=8,
            cache_path=tmp_path / "cache.jsonl",
            allow_api=True,
            max_new_calls=0,
        )


def test_repeated_gemini_writes_seed_and_aggregate_summaries(tmp_path: Path) -> None:
    run_repeated_gemini_baseline = _load_run_repeated_gemini_baseline()
    data_path = tmp_path / "hotpot.json"
    data_path.write_text(json.dumps([_example(i) for i in range(8)]), encoding="utf-8")

    metadata = run_repeated_gemini_baseline(
        data_path=data_path,
        output_dir=tmp_path / "outputs",
        seeds=[1, 2],
        num_examples=8,
        cache_path=tmp_path / "cache.jsonl",
        rewrite_provider=lambda question, mode: [question] if mode == "rewrite" else [question, question],
        allow_api=True,
        max_new_calls=999,
    )

    seed_summary = pd.read_csv(metadata["outputs"]["seed_summary_csv"])
    aggregate = pd.read_csv(metadata["outputs"]["summary_csv"])

    assert set(seed_summary["seed"]) == {1, 2}
    assert set(aggregate["method"]) == {"Gemini rewrite-all", "Gemini decompose"}
    assert {"reward_mean", "reward_std", "recall_at_5_mean", "mrr_mean"} <= set(aggregate.columns)


def _load_run_repeated_gemini_baseline():
    script_path = Path("scripts/run_repeated_gemini_baseline.py")
    spec = importlib.util.spec_from_file_location("run_repeated_gemini_baseline", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_repeated_gemini_baseline


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
