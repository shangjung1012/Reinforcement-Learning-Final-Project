from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.policies.constrained_policy import export_constrained_policy_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detailed-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--call-penalties", default="0,0.01,0.03,0.06,0.1,0.2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--mrr-weight", type=float, default=0.5)
    parser.add_argument("--call-baseline", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    call_penalties = [float(value) for value in args.call_penalties.split(",") if value.strip()]
    output_csv = args.output_csv or Path("outputs/results") / f"{args.dataset}_constrained_policy_bootstrap.csv"
    csv_path = export_constrained_policy_bootstrap(
        dataset=args.dataset,
        detailed_csv=args.detailed_csv,
        output_csv=output_csv,
        call_penalties=call_penalties,
        split=args.split,
        l2=args.l2,
        mrr_weight=args.mrr_weight,
        call_baseline=args.call_baseline,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
