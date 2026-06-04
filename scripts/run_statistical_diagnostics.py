from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.diagnostics.statistical_diagnostics import DEFAULT_METRICS, export_paired_bootstrap_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detailed-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--policy-method", default="Selective retrieval policy")
    parser.add_argument("--baseline-method", default="Train-best retrieval action")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args()

    detailed_csv = args.detailed_csv or Path("outputs") / "results" / f"{args.dataset}_retrieval_policy_detailed.csv"
    output_csv = args.output_csv or Path("outputs") / "results" / f"{args.dataset}_bootstrap_diagnostics.csv"
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    csv_path = export_paired_bootstrap_diagnostics(
        detailed_csv=detailed_csv,
        output_csv=output_csv,
        dataset=args.dataset,
        policy_method=args.policy_method,
        baseline_method=args.baseline_method,
        metrics=metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        ci=args.ci,
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
