from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.reports.evidence_consistency import export_evidence_consistency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-report-md", type=Path, default=Path("FINAL_REPORT.md"))
    parser.add_argument(
        "--protocol-summary-csv",
        type=Path,
        default=Path("outputs/results/nfcorpus_vertex_selection_protocol_summary.csv"),
    )
    parser.add_argument(
        "--deployment-decision-csv",
        type=Path,
        default=Path("outputs/results/nfcorpus_vertex_deployment_decision.csv"),
    )
    parser.add_argument(
        "--artifact-index-csv",
        type=Path,
        default=Path("outputs/results/final_artifact_index.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/results/final_evidence_consistency.csv"),
    )
    args = parser.parse_args()

    csv_path = export_evidence_consistency(
        final_report_md=args.final_report_md,
        protocol_summary_csv=args.protocol_summary_csv,
        deployment_decision_csv=args.deployment_decision_csv,
        artifact_index_csv=args.artifact_index_csv,
        output_csv=args.output_csv,
    )
    print(pd.read_csv(csv_path).to_string(index=False))


if __name__ == "__main__":
    main()
