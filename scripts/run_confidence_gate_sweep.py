from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.confidence_gate_sweep import export_confidence_gate_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--detailed-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--margins", default="0,0.001,0.005,0.01,0.02,0.05,0.1")
    args = parser.parse_args()

    margins = [float(value) for value in args.margins.split(",") if value.strip()]
    output_csv = args.output_csv or Path("outputs/results") / f"{args.dataset}_confidence_gate_sweep.csv"
    csv_path = export_confidence_gate_sweep(
        detailed_csv=args.detailed_csv,
        output_csv=output_csv,
        dataset=args.dataset,
        margins=margins,
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
