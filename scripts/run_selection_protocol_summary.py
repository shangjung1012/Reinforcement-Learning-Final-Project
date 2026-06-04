from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.diagnostics.selection_protocol import export_deployment_decision, export_selection_protocol_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--policy-diagnostics-csv", type=Path, required=True)
    parser.add_argument("--depth-selection-diagnostics-csv", type=Path, required=True)
    parser.add_argument("--depth-stability-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--deployment-decision-csv", type=Path, default=None)
    args = parser.parse_args()

    csv_path = export_selection_protocol_summary(
        policy_diagnostics_csv=args.policy_diagnostics_csv,
        depth_selection_diagnostics_csv=args.depth_selection_diagnostics_csv,
        depth_stability_csv=args.depth_stability_csv,
        output_csv=args.output_csv,
        dataset=args.dataset,
    )
    print(pd.read_csv(csv_path).to_string(index=False))
    if args.deployment_decision_csv is not None:
        decision_csv = export_deployment_decision(csv_path, args.deployment_decision_csv)
        print(pd.read_csv(decision_csv).to_string(index=False))


if __name__ == "__main__":
    main()
