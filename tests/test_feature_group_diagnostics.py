from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.diagnostics.feature_group_diagnostics import export_feature_group_diagnostics, feature_group_slices


def test_feature_group_slices_detects_base_and_contrast_groups() -> None:
    groups = feature_group_slices(22)

    assert groups["bias"] == (0, 1)
    assert groups["query_form"] == (1, 3)
    assert groups["retrieval_confidence"] == (3, 6)
    assert groups["wh_word"] == (6, 14)
    assert groups["retrieval_contrast"] == (14, 22)


def test_feature_group_slices_infers_deeper_semantic_profile_groups() -> None:
    groups = feature_group_slices(62)

    assert groups["semantic_summary"] == (14, 23)
    assert groups["semantic_rank_profile"] == (23, 34)
    assert groups["semantic_score_shape"] == (34, 38)
    assert groups["semantic_rank_agreement"] == (38, 44)
    assert groups["embedding_projection"] == (44, 56)
    assert groups["lexical_semantic_interactions"] == (56, 62)


def test_feature_group_slices_keeps_deeper_semantic_contrast_separate() -> None:
    groups = feature_group_slices(70)

    assert groups["retrieval_contrast"] == (62, 70)
    assert groups["lexical_semantic_interactions"] == (56, 62)


def test_export_feature_group_diagnostics_writes_group_statistics(tmp_path: Path) -> None:
    features = np.asarray(
        [
            [1.0, 0.0, 2.0, 0.5, 0.2, 0.1, *([0.0] * 8), 0.0, 0.5, 0.5, 0.0, 0.25, 0.25, 0.5, 0.5],
            [1.0, 0.0, 4.0, 0.7, 0.1, 0.1, *([0.0] * 8), 0.0, 0.5, 0.7, 0.0, 0.25, 0.5, 0.5, 0.75],
        ],
        dtype=float,
    )
    output_csv = tmp_path / "feature_groups.csv"

    exported = export_feature_group_diagnostics(
        features,
        output_csv,
        dataset="toy",
        split="train",
    )
    diagnostics = pd.read_csv(exported)
    contrast = diagnostics[diagnostics["group"] == "retrieval_contrast"].iloc[0]
    query = diagnostics[diagnostics["group"] == "query_form"].iloc[0]

    assert set(diagnostics["group"]) == {"bias", "query_form", "retrieval_confidence", "wh_word", "retrieval_contrast"}
    assert contrast["width"] == 8
    assert contrast["nonzero_rate"] > 0.0
    assert contrast["active_columns"] == 6
    assert query["low_variance_columns"] == 1
