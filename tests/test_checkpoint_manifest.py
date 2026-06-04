from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.policies.bandit import DirectMethodBandit
from selective_rag_rl.reports.checkpoint_manifest import export_checkpoint_manifest
from selective_rag_rl.core.policy_io import save_checkpoint


def test_export_checkpoint_manifest_summarizes_saved_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "outputs" / "checkpoints" / "toy_policy.pkl"
    output_csv = tmp_path / "outputs" / "results" / "checkpoint_manifest.csv"
    policy = DirectMethodBandit(["keep", "rewrite"], l2=0.5)
    features = np.asarray([[1.0, 0.0, 0.2], [1.0, 1.0, 0.4]], dtype=float)
    policy.fit(features, {"keep": [1.0, 0.5], "rewrite": [0.2, 0.9]})
    save_checkpoint(
        checkpoint,
        policy,
        {
            "dataset": "Toy retrieval policy",
            "dataset_key": "toy",
            "seed": 7,
            "policy_model": "ridge",
            "selected_policy_model": "ridge_l2=0.5",
            "feature_set": "full",
            "semantic_features": "none",
            "semantic_depth": 5,
            "num_train_examples": 2,
            "num_test_examples": 1,
            "retrieval_call_cost": 0.03,
            "actions": ["keep", "rewrite"],
        },
    )

    exported = export_checkpoint_manifest([checkpoint], output_csv=output_csv, root=tmp_path)
    row = pd.read_csv(exported).iloc[0]

    assert row["checkpoint_id"] == "toy_policy"
    assert row["path"] == "outputs/checkpoints/toy_policy.pkl"
    assert bool(row["exists"]) is True
    assert row["model_class"] == "DirectMethodBandit"
    assert row["dataset"] == "Toy retrieval policy"
    assert row["dataset_key"] == "toy"
    assert row["seed"] == 7
    assert row["policy_model"] == "ridge"
    assert row["selected_policy_model"] == "ridge_l2=0.5"
    assert row["feature_set"] == "full"
    assert row["semantic_features"] == "none"
    assert row["semantic_depth"] == 5
    assert row["action_count"] == 2
    assert row["actions"] == "keep|rewrite"
    assert row["feature_width"] == 3
    assert row["num_train_examples"] == 2
    assert row["num_test_examples"] == 1
    assert row["retrieval_call_cost"] == 0.03


def test_export_checkpoint_manifest_records_missing_checkpoint(tmp_path: Path) -> None:
    output_csv = tmp_path / "manifest.csv"
    missing = tmp_path / "outputs" / "checkpoints" / "missing.pkl"

    exported = export_checkpoint_manifest([missing], output_csv=output_csv, root=tmp_path)
    row = pd.read_csv(exported).iloc[0]

    assert row["checkpoint_id"] == "missing"
    assert bool(row["exists"]) is False
    assert pd.isna(row["model_class"])
