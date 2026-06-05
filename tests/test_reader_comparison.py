from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def test_run_reader_comparison_writes_per_reader_summary(tmp_path: Path) -> None:
    run_reader_comparison = _load_run_reader_comparison()
    output_dir = tmp_path / "reader_comparison"

    metadata = run_reader_comparison(
        dataset="toy",
        output_dir=output_dir,
        readers=["lexical", "span"],
        num_examples=4,
        seed=7,
        k=2,
    )

    summary_csv = output_dir / "results" / "reader_comparison_summary.csv"
    detailed_csv = output_dir / "results" / "reader_comparison_detailed.csv"
    summary = pd.read_csv(summary_csv).set_index("reader")
    detailed = pd.read_csv(detailed_csv)

    assert metadata["dataset"] == "toy"
    assert metadata["readers"] == ["lexical", "span"]
    assert set(summary.index) == {"lexical", "span"}
    assert summary.loc["span", "exact_match"] == 1.0
    assert summary.loc["span", "f1"] >= summary.loc["lexical", "f1"]
    assert {"reader", "qid", "predicted_answer", "exact_match", "f1"} <= set(detailed.columns)


def _load_run_reader_comparison():
    script_path = Path("scripts/run_reader_comparison.py")
    spec = importlib.util.spec_from_file_location("run_reader_comparison", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_reader_comparison
