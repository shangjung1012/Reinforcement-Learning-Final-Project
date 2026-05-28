from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from selective_rag_rl.policy_io import load_checkpoint


CHECKPOINT_MANIFEST_COLUMNS = [
    "checkpoint_id",
    "path",
    "exists",
    "model_class",
    "dataset",
    "dataset_key",
    "seed",
    "stage",
    "policy_model",
    "selected_policy_model",
    "feature_set",
    "semantic_features",
    "semantic_depth",
    "action_count",
    "actions",
    "feature_width",
    "num_train_examples",
    "num_test_examples",
    "retrieval_call_cost",
]


def export_checkpoint_manifest(
    checkpoint_paths: list[Path],
    *,
    output_csv: Path,
    root: Path | None = None,
) -> Path:
    root = (root or Path.cwd()).resolve()
    rows = [_checkpoint_row(path, root=root) for path in checkpoint_paths]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CHECKPOINT_MANIFEST_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def final_project_checkpoint_paths(root: Path | None = None) -> list[Path]:
    root = root or Path(".")
    checkpoint_dir = root / "outputs" / "checkpoints"
    return [
        checkpoint_dir / "hotpot_bandit_policy.pkl",
        checkpoint_dir / "hotpot_retrieval_policy.pkl",
        checkpoint_dir / "hotpot_llm_retrieval_policy.pkl",
        checkpoint_dir / "hotpot_fqi_q0.pkl",
        checkpoint_dir / "hotpot_fqi_q1.pkl",
        checkpoint_dir / "nq_bandit_policy.pkl",
        checkpoint_dir / "nq_retrieval_policy.pkl",
        checkpoint_dir / "scifact_retrieval_policy.pkl",
        checkpoint_dir / "nfcorpus_retrieval_policy.pkl",
    ]


def _checkpoint_row(path: Path, *, root: Path) -> dict[str, object]:
    resolved_path = path if path.is_absolute() else (root / path).resolve()
    base_row = {
        "checkpoint_id": resolved_path.stem,
        "path": _relative_path(resolved_path, root),
        "exists": resolved_path.exists(),
        "model_class": "",
        "dataset": "",
        "dataset_key": "",
        "seed": "",
        "stage": "",
        "policy_model": "",
        "selected_policy_model": "",
        "feature_set": "",
        "semantic_features": "",
        "semantic_depth": "",
        "action_count": 0,
        "actions": "",
        "feature_width": 0,
        "num_train_examples": "",
        "num_test_examples": "",
        "retrieval_call_cost": "",
    }
    if not resolved_path.exists():
        return base_row

    checkpoint = load_checkpoint(resolved_path)
    metadata = checkpoint.get("metadata", {})
    model = checkpoint["model"]
    actions = _actions(metadata, model)
    base_row.update(
        {
            "model_class": type(model).__name__,
            "dataset": _metadata_value(metadata, "dataset"),
            "dataset_key": _metadata_value(metadata, "dataset_key"),
            "seed": _metadata_value(metadata, "seed"),
            "stage": _metadata_value(metadata, "stage"),
            "policy_model": _metadata_value(metadata, "policy_model"),
            "selected_policy_model": _metadata_value(metadata, "selected_policy_model"),
            "feature_set": _metadata_value(metadata, "feature_set"),
            "semantic_features": _metadata_value(metadata, "semantic_features"),
            "semantic_depth": _metadata_value(metadata, "semantic_depth"),
            "action_count": len(actions),
            "actions": "|".join(actions),
            "feature_width": _feature_width(metadata, model),
            "num_train_examples": _first_metadata_value(metadata, ["num_train_examples", "train_examples"]),
            "num_test_examples": _first_metadata_value(metadata, ["num_test_examples", "test_examples"]),
            "retrieval_call_cost": _metadata_value(metadata, "retrieval_call_cost"),
        }
    )
    return base_row


def _actions(metadata: dict[str, Any], model: object) -> list[str]:
    metadata_actions = metadata.get("actions")
    if isinstance(metadata_actions, list):
        return [str(action) for action in metadata_actions]
    model_actions = getattr(model, "actions", [])
    if isinstance(model_actions, list):
        return [str(action) for action in model_actions]
    return []


def _feature_width(metadata: dict[str, Any], model: object) -> int:
    metadata_width = metadata.get("feature_width")
    if metadata_width not in (None, ""):
        return int(metadata_width)

    mean = getattr(model, "mean", None)
    if isinstance(mean, np.ndarray) and mean.ndim == 1:
        return int(mean.shape[0])

    weights = getattr(model, "weights", None)
    if isinstance(weights, dict) and weights:
        first = next(iter(weights.values()))
        if isinstance(first, np.ndarray):
            return int(first.shape[0])

    models = getattr(model, "models", None)
    if isinstance(models, dict) and models:
        first_model = next(iter(models.values()))
        width = getattr(first_model, "n_features_in_", None)
        if width is not None:
            return int(width)

    by_action = getattr(model, "by_action", None)
    if isinstance(by_action, dict) and by_action:
        for features, _values in by_action.values():
            if isinstance(features, np.ndarray) and features.ndim == 2 and features.shape[1] > 0:
                return int(features.shape[1])

    return 0


def _metadata_value(metadata: dict[str, Any], key: str) -> object:
    value = metadata.get(key, "")
    if value is None:
        return ""
    return value


def _first_metadata_value(metadata: dict[str, Any], keys: list[str]) -> object:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return ""


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
