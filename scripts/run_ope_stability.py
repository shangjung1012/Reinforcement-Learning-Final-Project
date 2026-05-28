from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.off_policy_evaluation import DEFAULT_TARGET_METHODS, export_ope_stability_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detailed-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-methods", default=",".join(DEFAULT_TARGET_METHODS))
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    target_methods = [method.strip() for method in args.target_methods.split(",") if method.strip()]
    output_csv = args.output_csv or Path("outputs/results") / f"{args.dataset}_ope_stability.csv"
    csv_path = export_ope_stability_diagnostics(
        dataset=args.dataset,
        detailed_csv=args.detailed_csv,
        output_csv=output_csv,
        seeds=seeds,
        split=args.split,
        target_methods=target_methods,
        l2=args.l2,
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
