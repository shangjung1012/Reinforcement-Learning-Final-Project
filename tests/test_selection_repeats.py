from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.selection_repeats import run_repeated_policy_model_selection


def test_run_repeated_policy_model_selection_writes_manifest_and_stability(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_path.mkdir()

    metadata = run_repeated_policy_model_selection(
        dataset="nfcorpus",
        data_path=data_path,
        output_dir=output_dir,
        seeds=[1, 2],
        policy_models=["ridge", "auto"],
        feature_sets=["full", "no_semantic"],
        num_train_examples=4,
        num_test_examples=3,
        sweep_runner=_fake_sweep_runner,
    )

    manifest = pd.read_csv(metadata["manifest_csv"])
    stability = pd.read_csv(metadata["stability_csv"]).iloc[0]
    diagnostics = pd.read_csv(metadata["diagnostics_csv"])

    assert metadata["seeds"] == [1, 2]
    assert list(manifest["seed"]) == [1, 2]
    assert all(Path(path).exists() for path in manifest["grid_csv"])
    assert list(diagnostics["seed"]) == [1, 2]
    assert list(diagnostics["dataset"]) == ["nfcorpus_repeated_selection_seed_1", "nfcorpus_repeated_selection_seed_2"]
    assert "grid_csv" in diagnostics.columns
    assert diagnostics.loc[0, "validation_selected_config"] == "full/ridge/ridge_l2=1.0"
    assert diagnostics.loc[0, "heldout_best_config"] == "no_semantic/ridge/ridge_l2=1.0"
    assert diagnostics.loc[0, "guardrail_decision"] == "fallback_train_best_fixed"
    assert stability["dataset"] == "nfcorpus_repeated_selection"
    assert stability["n_runs"] == 2
    assert stability["n_unique_validation_selected"] == 2
    assert stability["validation_matches_heldout_rate"] == 0.5


def _fake_sweep_runner(
    *,
    dataset: str,
    data_path: Path,
    output_dir: Path,
    policy_models: list[str] | None,
    feature_sets: list[str] | None,
    seed: int,
    **_: object,
) -> Path:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    selected_feature = "full" if seed == 1 else "no_semantic"
    rows = [
        _row(dataset, "full", "ridge", "ridge_l2=1.0", validation=0.90 if seed == 1 else 0.30, reward=0.20),
        _row(dataset, "full", "auto", "knn_k=1", validation=0.80, reward=0.30),
        _row(
            dataset,
            "no_semantic",
            "ridge",
            "ridge_l2=1.0",
            validation=0.70 if selected_feature == "full" else 0.95,
            reward=0.95,
        ),
    ]
    csv_path = results_dir / f"{dataset}_policy_model_sweep.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _row(
    dataset: str,
    feature_set: str,
    policy_model: str,
    selected: str,
    validation: float,
    reward: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "policy_model": policy_model,
        "selected_policy_model": selected,
        "train_size": 4,
        "test_size": 3,
        "feature_width": 14,
        "validation_reward": validation,
        "best_fixed_validation_reward": 0.40,
        "selective_reward": reward,
        "best_fixed_reward": 0.40,
        "best_fixed_retrieval_calls": 0.50,
        "oracle_reward": 1.00,
        "selective_recall_at_5": 0.10,
        "best_fixed_recall_at_5": 0.08,
        "selective_retrieval_calls": 1.0,
    }
