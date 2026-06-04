from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from selective_rag_rl.policies.off_policy_evaluation import export_ope_diagnostics
from selective_rag_rl.experiments.retrieval_policy_experiment import run_retrieval_policy_experiment

PYTEST_TARGETS = [
    "tests/test_core.py",
    "tests/test_dense_retriever.py",
    "tests/test_off_policy_evaluation.py",
    "tests/test_retrieval_policy_experiment.py::test_run_retrieval_policy_experiment_writes_summary_and_checkpoint",
]


def run_final_smoke(
    output_dir: Path,
    num_examples: int = 12,
    seed: int = 42,
    pytest_mode: str = "targeted",
) -> dict[str, object]:
    if num_examples < 4:
        raise ValueError("num_examples must be at least 4 so train/test smoke splits are non-empty")
    if pytest_mode not in {"targeted", "full", "skip"}:
        raise ValueError("pytest_mode must be one of: targeted, full, skip")

    output_dir = output_dir.resolve()
    fixture_path = output_dir / "fixtures" / "synthetic_hotpot.json"
    retrieval_output_dir = output_dir / "retrieval_policy_smoke"
    ope_output_csv = output_dir / "results" / "smoke_ope_diagnostics.csv"
    manifest_path = output_dir / "smoke_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_synthetic_hotpot_fixture(fixture_path, num_examples)

    pytest_command = _pytest_command(pytest_mode)
    if pytest_command:
        subprocess.run(pytest_command, check=True)

    metadata = run_retrieval_policy_experiment(
        data_path=fixture_path,
        output_dir=retrieval_output_dir,
        num_examples=num_examples,
        seed=seed,
        k=3,
        embedder_name="fake",
        retrieval_call_cost=0.03,
        policy_model="ridge",
        knn_k_candidates=[1],
        tuning_folds=2,
    )
    export_ope_diagnostics(
        dataset="synthetic_hotpot_smoke",
        detailed_csv=Path(metadata["outputs"]["detailed_csv"]),
        output_csv=ope_output_csv,
        split="test",
        target_methods=[
            "Train-best retrieval action",
            "Heuristic retrieval router",
            "Selective retrieval policy",
        ],
        seed=seed,
    )

    manifest = {
        "status": "pass",
        "seed": seed,
        "num_examples": num_examples,
        "pytest_mode": pytest_mode,
        "pytest_command": " ".join(pytest_command) if pytest_command else None,
        "uses_raw_data": False,
        "uses_external_api": False,
        "uses_model_download": False,
        "outputs": {
            "synthetic_fixture": str(fixture_path),
            "retrieval_detailed_csv": str(metadata["outputs"]["detailed_csv"]),
            "retrieval_summary_csv": str(metadata["outputs"]["summary_csv"]),
            "retrieval_metadata_json": str(metadata["outputs"]["metadata_json"]),
            "retrieval_checkpoint": str(metadata["outputs"]["checkpoint"]),
            "ope_diagnostics_csv": str(ope_output_csv),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _pytest_command(pytest_mode: str) -> list[str]:
    if pytest_mode == "skip":
        return []
    if pytest_mode == "full":
        return [sys.executable, "-m", "pytest", "-q"]
    return [sys.executable, "-m", "pytest", "-q", *PYTEST_TARGETS]


def _write_synthetic_hotpot_fixture(path: Path, num_examples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(num_examples):
        left = f"Alpha Evidence {idx}"
        right = f"Beta Evidence {idx}"
        distractor = f"Noise Document {idx}"
        rows.append(
            {
                "_id": f"synthetic-{idx}",
                "question": f"Which evidence connects {left} and {right}?",
                "answer": f"connection {idx}",
                "level": "synthetic",
                "type": "bridge",
                "context": [
                    [left, [f"{left} describes the first supporting fact for connection {idx}."]],
                    [right, [f"{right} describes the second supporting fact for connection {idx}."]],
                    [distractor, [f"{distractor} is unrelated background text for example {idx}."]],
                ],
                "supporting_facts": [[left, 0], [right, 0]],
            }
        )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a raw-data-free final project smoke reproduction.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_smoke"))
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pytest-mode", choices=["targeted", "full", "skip"], default="targeted")
    args = parser.parse_args()

    manifest = run_final_smoke(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        seed=args.seed,
        pytest_mode=args.pytest_mode,
    )
    print(json.dumps(manifest, indent=2))

    summary_csv = Path(manifest["outputs"]["retrieval_summary_csv"])
    if summary_csv.exists():
        print(pd.read_csv(summary_csv).to_string(index=False))


if __name__ == "__main__":
    main()
