from __future__ import annotations

from pathlib import Path

import pyarrow
import pyarrow.parquet as parquet

from selective_rag_rl.experiments.nq_experiment import run_nq_experiment
from selective_rag_rl.core.policy_io import load_checkpoint
from selective_rag_rl.experiments.retrieval_policy_experiment import run_nq_retrieval_policy_experiment


def test_run_nq_experiment_writes_summary(tmp_path: Path) -> None:
    rows = [
        _row("q1", "Ada Lovelace", "who wrote notes for the analytical engine", ["Ada", "Lovelace", "Analytical", "Engine"]),
        _row("q2", "Grace Hopper", "who worked on cobol", ["Grace", "Hopper", "COBOL"]),
        _row("q3", "Alan Turing", "who studied computation", ["Alan", "Turing", "computation"]),
        _row("q4", "Katherine Johnson", "who calculated nasa trajectories", ["Katherine", "Johnson", "NASA", "trajectories"]),
        _row("q5", "Barbara Liskov", "who designed clust", ["Barbara", "Liskov", "CLU"]),
    ]
    data_path = tmp_path / "nq.parquet"
    output_dir = tmp_path / "outputs"
    parquet.write_table(pyarrow.Table.from_pylist(rows), data_path)

    metadata = run_nq_experiment(data_path, output_dir, num_examples=5, seed=3, pool_size=5)

    summary_path = Path(metadata["outputs"]["summary_csv"])
    assert summary_path.exists()
    assert "Natural Questions" in metadata["dataset"]
    assert "Selective bandit" in summary_path.read_text(encoding="utf-8")


def test_run_nq_retrieval_policy_experiment_writes_summary(tmp_path: Path) -> None:
    rows = [
        _row("q1", "Ada Lovelace", "who wrote notes for the analytical engine", ["Ada", "Lovelace", "Analytical", "Engine"]),
        _row("q2", "Grace Hopper", "who worked on cobol", ["Grace", "Hopper", "COBOL"]),
        _row("q3", "Alan Turing", "who studied computation", ["Alan", "Turing", "computation"]),
        _row("q4", "Katherine Johnson", "who calculated nasa trajectories", ["Katherine", "Johnson", "NASA", "trajectories"]),
        _row("q5", "Barbara Liskov", "who designed clust", ["Barbara", "Liskov", "CLU"]),
        _row("q6", "Edsger Dijkstra", "who wrote about structured programming", ["Edsger", "Dijkstra", "structured"]),
    ]
    data_path = tmp_path / "nq.parquet"
    output_dir = tmp_path / "outputs"
    parquet.write_table(pyarrow.Table.from_pylist(rows), data_path)

    metadata = run_nq_retrieval_policy_experiment(
        data_path=data_path,
        output_dir=output_dir,
        num_examples=6,
        seed=3,
        pool_size=5,
        embedder_name="fake",
        knn_k_candidates=[1, 3],
        tuning_folds=2,
    )

    summary = Path(metadata["outputs"]["summary_csv"]).read_text(encoding="utf-8")
    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert metadata["pool_size"] == 5
    assert checkpoint["metadata"]["dataset"] == "Natural Questions retrieval-action policy"
    assert Path(metadata["outputs"]["summary_csv"]).name == "nq_retrieval_policy_summary.csv"
    assert "Dense original" in summary
    assert "Hybrid keyword" in summary
    assert "Selective retrieval policy" in summary


def _row(qid: str, title: str, question: str, tokens: list[str]) -> dict[str, object]:
    return {
        "id": qid,
        "document": {
            "title": title,
            "tokens": {
                "token": tokens,
                "is_html": [False] * len(tokens),
            },
        },
        "question": {"text": question},
        "annotations": {
            "short_answers": [{"text": [title]}],
            "yes_no_answer": [-1],
        },
    }
