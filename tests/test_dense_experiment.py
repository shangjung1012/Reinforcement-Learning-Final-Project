from __future__ import annotations

import json
from pathlib import Path

from selective_rag_rl.dense_experiment import run_dense_hotpot_experiment


def test_run_dense_hotpot_experiment_writes_summary(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(8)]), encoding="utf-8")

    metadata = run_dense_hotpot_experiment(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=8,
        seed=1,
        embedder_name="fake",
    )

    summary = Path(metadata["outputs"]["summary_csv"]).read_text(encoding="utf-8")
    assert "Dense original" in summary
    assert "Hybrid keyword" in summary


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
