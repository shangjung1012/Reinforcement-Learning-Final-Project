from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.experiments.retrieval_policy_experiment import (
    RETRIEVAL_CONTRAST_FEATURE_WIDTH,
    SEMANTIC_FEATURE_START,
    semantic_feature_group_slices,
)


def feature_group_slices(width: int) -> dict[str, tuple[int, int]]:
    groups = {
        "bias": (0, 1),
        "query_form": (1, 3),
        "retrieval_confidence": (3, 6),
        "wh_word": (6, 14),
    }
    contrast_start = _contrast_start(width)
    if width > SEMANTIC_FEATURE_START and contrast_start > SEMANTIC_FEATURE_START:
        groups.update(semantic_feature_group_slices(width))
    if contrast_start < width:
        groups["retrieval_contrast"] = (contrast_start, width)
    return {name: (start, end) for name, (start, end) in groups.items() if start < end <= width}


def export_feature_group_diagnostics(
    features: np.ndarray,
    output_csv: Path,
    dataset: str,
    split: str,
) -> Path:
    diagnostics = feature_group_diagnostics_frame(features, dataset=dataset, split=split)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_csv, index=False)
    return output_csv


def feature_group_diagnostics_frame(features: np.ndarray, dataset: str, split: str) -> pd.DataFrame:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("Feature diagnostics require a vector or matrix")

    rows = []
    for name, (start, end) in feature_group_slices(matrix.shape[1]).items():
        values = matrix[:, start:end]
        column_std = values.std(axis=0)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "group": name,
                "start": start,
                "end": end,
                "width": end - start,
                "mean_abs": float(np.mean(np.abs(values))) if values.size else 0.0,
                "std_mean": float(np.mean(column_std)) if column_std.size else 0.0,
                "nonzero_rate": float(np.mean(np.abs(values) > 1e-12)) if values.size else 0.0,
                "active_columns": int(np.sum(np.any(np.abs(values) > 1e-12, axis=0))) if values.size else 0,
                "low_variance_columns": int(np.sum(column_std <= 1e-12)) if column_std.size else 0,
            }
        )

    return pd.DataFrame(rows)


def _contrast_start(width: int) -> int:
    if width >= 64:
        return width - RETRIEVAL_CONTRAST_FEATURE_WIDTH
    if width == SEMANTIC_FEATURE_START + RETRIEVAL_CONTRAST_FEATURE_WIDTH:
        return SEMANTIC_FEATURE_START
    return width
