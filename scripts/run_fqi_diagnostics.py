from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.diagnostics.fqi_diagnostics import export_fqi_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export post-hoc diagnostics for the two-step FQI extension.")
    parser.add_argument("--dataset", default="hotpot")
    parser.add_argument("--detailed-csv", type=Path, default=Path("outputs/results/multistep_detailed.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/results/hotpot_fqi_diagnostics_summary.csv"))
    parser.add_argument("--trace-csv", type=Path, default=Path("outputs/results/hotpot_fqi_trace_distribution.csv"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    outputs = export_fqi_diagnostics(
        detailed_csv=args.detailed_csv,
        summary_csv=args.summary_csv,
        trace_csv=args.trace_csv,
        dataset=args.dataset,
        split=args.split,
    )
    print(pd.read_csv(outputs["summary_csv"]).to_string(index=False))
    print(pd.read_csv(outputs["trace_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
