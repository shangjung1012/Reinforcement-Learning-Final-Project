from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def test_run_reader_eval_toy_mode_writes_em_f1_outputs(tmp_path: Path) -> None:
    run_reader_eval = _load_run_reader_eval()
    output_dir = tmp_path / "reader_eval"

    metadata = run_reader_eval(
        dataset="toy",
        output_dir=output_dir,
        num_examples=4,
        seed=3,
        k=2,
    )

    detailed_csv = output_dir / "results" / "reader_eval_detailed.csv"
    summary_csv = output_dir / "results" / "reader_eval_summary.csv"
    summary_json = output_dir / "results" / "reader_eval_summary.json"

    assert metadata["dataset"] == "toy"
    assert metadata["reader"] == "lexical"
    assert detailed_csv.exists()
    assert summary_csv.exists()
    assert summary_json.exists()

    detailed = pd.read_csv(detailed_csv)
    assert {"qid", "question", "gold_answer", "predicted_answer", "exact_match", "f1"} <= set(detailed.columns)
    assert detailed["f1"].between(0.0, 1.0).all()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["num_examples"] == 4
    assert 0.0 <= summary["exact_match"] <= 1.0
    assert 0.0 <= summary["f1"] <= 1.0


def test_run_reader_eval_toy_mode_supports_span_reader(tmp_path: Path) -> None:
    run_reader_eval = _load_run_reader_eval()
    output_dir = tmp_path / "span_reader_eval"

    metadata = run_reader_eval(
        dataset="toy",
        output_dir=output_dir,
        num_examples=4,
        seed=3,
        k=2,
        reader="span",
    )

    detailed = pd.read_csv(output_dir / "results" / "reader_eval_detailed.csv")

    assert metadata["reader"] == "span"
    assert metadata["exact_match"] == 1.0
    assert metadata["f1"] == 1.0
    assert set(detailed["predicted_answer"]) == {
        "Ada Lovelace",
        "Grace Hopper",
        "Alan Turing",
        "Katherine Johnson",
    }


def _load_run_reader_eval():
    script_path = Path("scripts/run_reader_eval.py")
    spec = importlib.util.spec_from_file_location("run_reader_eval", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_reader_eval
