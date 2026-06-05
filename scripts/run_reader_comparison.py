from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


def run_reader_comparison(
    *,
    dataset: str,
    output_dir: Path,
    readers: list[str],
    num_examples: int = 10,
    seed: int = 42,
    k: int = 5,
    data_path: Path | None = None,
) -> dict[str, object]:
    run_reader_eval = _load_run_reader_eval()
    summary_frames = []
    detailed_frames = []
    for reader in readers:
        reader_output = output_dir / "runs" / reader
        metadata = run_reader_eval(
            dataset=dataset,
            data_path=data_path,
            output_dir=reader_output,
            num_examples=num_examples,
            seed=seed,
            k=k,
            reader=reader,
        )
        summary = pd.read_csv(metadata["outputs"]["summary_csv"])
        detailed = pd.read_csv(metadata["outputs"]["detailed_csv"])
        summary["reader"] = reader
        detailed["reader"] = reader
        summary_frames.append(summary)
        detailed_frames.append(detailed)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "reader_comparison_summary.csv"
    detailed_csv = results_dir / "reader_comparison_detailed.csv"
    summary_json = results_dir / "reader_comparison_summary.json"
    summary_frame = pd.concat(summary_frames, ignore_index=True)
    detailed_frame = pd.concat(detailed_frames, ignore_index=True)
    summary_frame.to_csv(summary_csv, index=False)
    detailed_frame.to_csv(detailed_csv, index=False)
    summary_payload = {
        "dataset": dataset,
        "readers": readers,
        "num_examples": num_examples,
        "seed": seed,
        "k": k,
        "outputs": {
            "summary_csv": str(summary_csv),
            "detailed_csv": str(detailed_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload


def _load_run_reader_eval():
    script_path = Path(__file__).with_name("run_reader_eval.py")
    spec = importlib.util.spec_from_file_location("run_reader_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_reader_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare lightweight reader EM/F1 baselines.")
    parser.add_argument("--dataset", choices=["toy", "hotpot", "nq"], default="toy")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codex_reader_comparison"))
    parser.add_argument("--readers", default="lexical,span")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    readers = [reader.strip() for reader in args.readers.split(",") if reader.strip()]
    metadata = run_reader_comparison(
        dataset=args.dataset,
        data_path=args.data_path,
        output_dir=args.output_dir,
        readers=readers,
        num_examples=args.num_examples,
        seed=args.seed,
        k=args.k,
    )
    print(json.dumps(metadata, indent=2))
    print(pd.read_csv(metadata["outputs"]["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
