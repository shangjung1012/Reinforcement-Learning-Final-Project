from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


MARKDOWN_FILES = [
    "FINAL_SLIDES.md",
    "FINAL_DEFENSE_QA.md",
    "FINAL_RESULTS_SUMMARY.md",
    "FINAL_PRESENTATION_OUTLINE.md",
    "FINAL_REPORT.md",
]
CONSISTENCY_COLUMNS = [
    "check_id",
    "check_type",
    "expected_value",
    "observed_value",
    "status",
    "checked_files",
    "evidence_path",
]


def export_markdown_consistency(*, root: Path, output_csv: Path) -> Path:
    markdown_text = _read_markdown_files(root)
    rows = [
        *_claim_value_rows(root, markdown_text),
        *_artifact_path_rows(root, markdown_text),
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CONSISTENCY_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _claim_value_rows(root: Path, markdown_text: dict[str, str]) -> list[dict[str, str]]:
    main = pd.read_csv(root / "outputs" / "results" / "final_main_results_table.csv")
    claims = pd.read_csv(root / "outputs" / "results" / "final_claims_matrix.csv")

    return [
        _value_check(
            check_id="scifact_policy_reward_delta",
            expected=_main_value(main, "scifact", "Selective retrieval policy", "delta_vs_train_best"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_main_results_table.csv[scifact/Selective retrieval policy/delta]",
        ),
        _value_check(
            check_id="scifact_policy_reward_ci_low",
            expected=_main_value(main, "scifact", "Selective retrieval policy", "ci_low"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_main_results_table.csv[scifact/Selective retrieval policy/ci_low]",
        ),
        _value_check(
            check_id="scifact_policy_reward_ci_high",
            expected=_main_value(main, "scifact", "Selective retrieval policy", "ci_high"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_main_results_table.csv[scifact/Selective retrieval policy/ci_high]",
        ),
        _value_check(
            check_id="nfcorpus_policy_reward_delta",
            expected=_main_value(main, "nfcorpus", "Selective retrieval policy", "delta_vs_train_best"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_main_results_table.csv[nfcorpus/Selective retrieval policy/delta]",
        ),
        _value_check(
            check_id="nfcorpus_policy_reward_ci_low",
            expected=_main_value(main, "nfcorpus", "Selective retrieval policy", "ci_low"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_main_results_table.csv[nfcorpus/Selective retrieval policy/ci_low]",
        ),
        _value_check(
            check_id="nfcorpus_policy_reward_ci_high",
            expected=_main_value(main, "nfcorpus", "Selective retrieval policy", "ci_high"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_main_results_table.csv[nfcorpus/Selective retrieval policy/ci_high]",
        ),
        _value_check(
            check_id="scifact_constrained_delta",
            expected=_main_value(main, "scifact", "Constrained policy lambda=0.03", "delta_vs_train_best"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_main_results_table.csv[scifact/constrained/delta]",
        ),
        _value_check(
            check_id="nfcorpus_constrained_delta",
            expected=_main_value(main, "nfcorpus", "Constrained policy lambda=0.03", "delta_vs_train_best"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_main_results_table.csv[nfcorpus/constrained/delta]",
        ),
        _value_check(
            check_id="scifact_ope_ips_sparse_error",
            expected=_claim_value(claims, "scifact_ope_ips_coverage_warning", "value"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_claims_matrix.csv[scifact_ope_ips_coverage_warning/value]",
        ),
        _value_check(
            check_id="scifact_ope_ips_uniform_error",
            expected=_claim_value(claims, "scifact_ope_ips_coverage_warning", "baseline_value"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_claims_matrix.csv[scifact_ope_ips_coverage_warning/baseline_value]",
        ),
        _value_check(
            check_id="nfcorpus_ope_dr_error",
            expected=_claim_value(claims, "nfcorpus_ope_dr_stability", "value"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_claims_matrix.csv[nfcorpus_ope_dr_stability/value]",
        ),
        _value_check(
            check_id="nfcorpus_ope_dm_error",
            expected=_claim_value(claims, "nfcorpus_ope_dr_stability", "baseline_value"),
            markdown_text=markdown_text,
            evidence_path="outputs/results/final_claims_matrix.csv[nfcorpus_ope_dr_stability/baseline_value]",
        ),
    ]


def _artifact_path_rows(root: Path, markdown_text: dict[str, str]) -> list[dict[str, str]]:
    paths = sorted(
        {
            match
            for text in markdown_text.values()
            for match in re.findall(r"outputs/(?:results|figures|checkpoints)/[A-Za-z0-9_./-]+", text)
        }
    )
    return [_path_check(root, path, markdown_text) for path in paths]


def _value_check(
    *,
    check_id: str,
    expected: object,
    markdown_text: dict[str, str],
    evidence_path: str,
) -> dict[str, str]:
    expected_text = _format_float(expected)
    missing = [name for name, text in markdown_text.items() if expected_text not in text]
    return {
        "check_id": check_id,
        "check_type": "claim_value",
        "expected_value": expected_text,
        "observed_value": "present_in_all_required_docs" if not missing else "missing_in:" + ",".join(missing),
        "status": "pass" if not missing else "fail",
        "checked_files": ",".join(markdown_text),
        "evidence_path": evidence_path,
    }


def _path_check(root: Path, path: str, markdown_text: dict[str, str]) -> dict[str, str]:
    exists = (root / path).exists()
    mentioned_in = [name for name, text in markdown_text.items() if path in text]
    return {
        "check_id": f"artifact_path_{path}",
        "check_type": "artifact_path",
        "expected_value": "exists",
        "observed_value": "exists" if exists else "missing",
        "status": "pass" if exists else "fail",
        "checked_files": ",".join(mentioned_in),
        "evidence_path": path,
    }


def _read_markdown_files(root: Path) -> dict[str, str]:
    texts = {}
    for name in MARKDOWN_FILES:
        path = root / name
        if not path.exists():
            texts[name] = ""
        else:
            texts[name] = path.read_text(encoding="utf-8")
    return texts


def _main_value(frame: pd.DataFrame, dataset: str, method: str, column: str) -> object:
    matches = frame[(frame["dataset"] == dataset) & (frame["method"] == method)]
    if matches.empty:
        raise ValueError(f"Missing main result row: {dataset}/{method}")
    return matches.iloc[0][column]


def _claim_value(frame: pd.DataFrame, claim_id: str, column: str) -> object:
    matches = frame[frame["claim_id"] == claim_id]
    if matches.empty:
        raise ValueError(f"Missing final claim row: {claim_id}")
    return matches.iloc[0][column]


def _format_float(value: object) -> str:
    return f"{float(value):.6f}"
