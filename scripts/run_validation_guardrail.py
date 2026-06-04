from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selective_rag_rl.policies.validation_guardrail import (
    DEFAULT_CANDIDATE_METHOD,
    DEFAULT_TRAIN_BEST_METHOD,
    run_validation_guardrail,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply conservative validation guardrails to policy outputs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detailed-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--grid-csv", type=Path, action="append", default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--candidate-method", default=DEFAULT_CANDIDATE_METHOD)
    parser.add_argument("--train-best-method", default=DEFAULT_TRAIN_BEST_METHOD)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    summary = run_validation_guardrail(
        dataset=args.dataset,
        detailed_csv=args.detailed_csv,
        summary_csv=args.summary_csv,
        grid_csvs=args.grid_csv,
        output_csv=args.output_csv,
        candidate_method=args.candidate_method,
        train_best_method=args.train_best_method,
        heldout_split=args.split,
    )
    print(json.dumps(summary["aggregate"], indent=2))
    print(pd.read_csv(summary["outputs"]["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
