from __future__ import annotations

import json
from pathlib import Path

from selective_rag_rl.ablation import run_hotpot_ablation


def test_run_hotpot_ablation_writes_policy_variants(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(8)]), encoding="utf-8")

    metadata = run_hotpot_ablation(data_path, output_dir, num_examples=8, seed=5)

    summary = Path(metadata["outputs"]["summary_csv"]).read_text(encoding="utf-8")
    assert "Selective bandit" in summary
    assert "No keep action" in summary
    assert "No cost penalty" in summary
    assert "Retrieval-only reward" in summary


def _example(i: int) -> dict[str, object]:
    left = f"Entity{i}A"
    right = f"Entity{i}B"
    return {
        "_id": f"q{i}",
        "answer": left,
        "question": f"Which author lived longer, {left} or {right}?",
        "supporting_facts": [[left, 0], [right, 0]],
        "context": [
            [left, [f"{left} was a writer with a long biography."]],
            [right, [f"{right} was another writer with a long biography."]],
            [f"Distractor{i}", ["This paragraph is unrelated."]],
        ],
        "type": "comparison",
        "level": "easy",
    }
