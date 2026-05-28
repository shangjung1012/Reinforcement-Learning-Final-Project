# Adaptive RAG Depth and Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add query-difficulty diagnostics and cross-dataset policy-transfer experiments for the cost-aware retrieval-action policy.

**Architecture:** First enrich detailed retrieval-policy CSVs with deployable state-feature columns, then build replay-only complexity diagnostics over those CSVs. Add a separate cross-dataset transfer module that reuses existing BEIR loaders, action evaluation, feature transforms, and direct-method policies instead of duplicating the retrieval stack.

**Tech Stack:** Python 3.13, pandas, numpy, pytest, uv, existing `selective_rag_rl` modules.

---

## File Structure

- Create `src/selective_rag_rl/complexity_diagnostics.py`
  - Loads detailed policy CSVs, assigns low/mid/high buckets, aggregates method metrics, and exports bucket/action-family CSVs.
- Create `scripts/run_complexity_diagnostics.py`
  - CLI wrapper for SciFact/NFCorpus complexity diagnostics.
- Create `tests/test_complexity_diagnostics.py`
  - Unit tests using synthetic detailed rows.
- Create `src/selective_rag_rl/cross_dataset_transfer.py`
  - Trains on one BEIR dataset and evaluates the source-trained policy on another.
- Create `scripts/run_cross_dataset_transfer.py`
  - CLI wrapper for the SciFact/NFCorpus transfer matrix.
- Create `tests/test_cross_dataset_transfer.py`
  - Unit tests for source-trained policy application and summary shape.
- Modify `src/selective_rag_rl/retrieval_policy_experiment.py`
  - Add difficulty feature columns to every detailed method row.
- Modify `tests/test_retrieval_policy_experiment.py`
  - Assert detailed outputs include difficulty feature columns.
- Modify `src/selective_rag_rl/artifact_index.py` and `tests/test_artifact_index.py`
  - Register new complexity and transfer artifacts.
- Modify `README.md` and `FINAL_REPORT.md`
  - Add commands and final result interpretation after formal runs.

## Task 1: Detailed CSV Difficulty Features

**Files:**
- Modify: `src/selective_rag_rl/retrieval_policy_experiment.py`
- Modify: `tests/test_retrieval_policy_experiment.py`

- [ ] **Step 1: Write failing test for feature columns**

Add assertions to `test_run_retrieval_policy_experiment_writes_summary_and_checkpoint` after the summary checks:

```python
detailed = pd.read_csv(metadata["outputs"]["detailed_csv"])
assert {
    "state_question_length",
    "state_capitalized_spans",
    "state_bm25_top1",
    "state_bm25_gap",
    "state_bm25_entropy",
    "oracle_reward_margin",
    "oracle_tie_count",
    "action_reward_std",
    "bm25_dense_doc_overlap",
    "dense_new_doc_rate",
    "hybrid_new_doc_rate",
} <= set(detailed.columns)
policy_rows = detailed[detailed["method"] == "Selective retrieval policy"]
assert policy_rows["state_bm25_top1"].notna().all()
assert policy_rows["oracle_tie_count"].ge(1).all()
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest -q tests/test_retrieval_policy_experiment.py::test_run_retrieval_policy_experiment_writes_summary_and_checkpoint
```

Expected: FAIL because `state_question_length` and related columns do not exist.

- [ ] **Step 3: Implement feature extraction helper**

In `src/selective_rag_rl/retrieval_policy_experiment.py`, add this helper near `_method_row`:

```python
def _difficulty_feature_row(action_eval: dict[str, object], actions: list[str], k: int) -> dict[str, object]:
    features = np.asarray(action_eval["features"], dtype=float)
    rewards = np.asarray([action_eval["actions"][action]["reward"] for action in actions], dtype=float)
    sorted_rewards = np.sort(rewards)
    top_reward = float(sorted_rewards[-1]) if sorted_rewards.size else 0.0
    runner_up = float(sorted_rewards[-2]) if sorted_rewards.size >= 2 else top_reward
    return {
        "state_question_length": float(features[1]) if features.size > 1 else 0.0,
        "state_capitalized_spans": float(features[2]) if features.size > 2 else 0.0,
        "state_bm25_top1": float(features[3]) if features.size > 3 else 0.0,
        "state_bm25_gap": float(features[4]) if features.size > 4 else 0.0,
        "state_bm25_entropy": float(features[5]) if features.size > 5 else 0.0,
        "oracle_reward_margin": top_reward - runner_up,
        "oracle_tie_count": int(np.sum(np.isclose(rewards, top_reward))),
        "action_reward_std": float(np.std(rewards)) if rewards.size else 0.0,
        "bm25_dense_doc_overlap": _overlap_rate(
            _top_doc_ids(action_eval["actions"], "bm25_keep"),
            _top_doc_ids(action_eval["actions"], "dense_keep"),
            k,
        ),
        "dense_new_doc_rate": _new_doc_rate(
            _top_doc_ids(action_eval["actions"], "dense_keep"),
            _top_doc_ids(action_eval["actions"], "bm25_keep"),
            k,
        ),
        "hybrid_new_doc_rate": _new_doc_rate(
            _top_doc_ids(action_eval["actions"], "hybrid_keep"),
            _top_doc_ids(action_eval["actions"], "bm25_keep"),
            k,
        ),
    }
```

Update `_method_row` signature:

```python
def _method_row(
    split_name: str,
    method_name: str,
    action: str,
    ex: QAExample,
    action_eval: dict[str, object],
    extra: dict[str, object] | None = None,
    difficulty_features: dict[str, object] | None = None,
) -> dict[str, object]:
```

Update row construction:

```python
    if difficulty_features:
        row.update(difficulty_features)
```

before `if extra:`.

In `run_retrieval_policy_on_examples`, inside the per-example loop after `action_eval = evaluate_retrieval_actions(...)`, add:

```python
difficulty_features = _difficulty_feature_row(action_eval, actions, k)
```

Pass `difficulty_features=difficulty_features` into every `_method_row(...)` call in that loop.

- [ ] **Step 4: Verify targeted test passes**

Run:

```bash
uv run pytest -q tests/test_retrieval_policy_experiment.py::test_run_retrieval_policy_experiment_writes_summary_and_checkpoint
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/selective_rag_rl/retrieval_policy_experiment.py tests/test_retrieval_policy_experiment.py
git commit -m "Add retrieval difficulty features to detailed outputs"
```

## Task 2: Complexity Diagnostics Module

**Files:**
- Create: `src/selective_rag_rl/complexity_diagnostics.py`
- Create: `tests/test_complexity_diagnostics.py`
- Create: `scripts/run_complexity_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_complexity_diagnostics.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.complexity_diagnostics import export_complexity_diagnostics


def test_export_complexity_diagnostics_writes_bucket_and_action_tables(tmp_path: Path) -> None:
    detailed = tmp_path / "detailed.csv"
    bucket_csv = tmp_path / "buckets.csv"
    action_csv = tmp_path / "actions.csv"
    pd.DataFrame(_rows()).to_csv(detailed, index=False)

    export_complexity_diagnostics(
        dataset="toy",
        detailed_csv=detailed,
        output_csv=bucket_csv,
        action_distribution_csv=action_csv,
    )

    buckets = pd.read_csv(bucket_csv)
    actions = pd.read_csv(action_csv)

    assert {
        "dataset",
        "split",
        "bucket_feature",
        "bucket",
        "method",
        "n_queries",
        "reward",
        "reward_delta_vs_train_best",
        "reward_delta_vs_heuristic",
        "retrieval_calls",
    } <= set(buckets.columns)
    assert set(buckets["method"]) >= {
        "Train-best retrieval action",
        "Heuristic retrieval router",
        "Selective retrieval policy",
        "Oracle retrieval action",
    }
    assert {"dense_action_rate", "hybrid_action_rate"} <= set(actions.columns)
    assert actions["n_queries"].sum() > 0


def _rows() -> list[dict[str, object]]:
    rows = []
    methods = [
        ("Train-best retrieval action", "dense_keep", 0.4),
        ("Heuristic retrieval router", "bm25_keep", 0.45),
        ("Selective retrieval policy", "hybrid_keyword", 0.6),
        ("Oracle retrieval action", "hybrid_keyword", 0.7),
    ]
    for idx, top1 in enumerate([0.1, 0.2, 0.8]):
        for method, action, base_reward in methods:
            rows.append(
                {
                    "split": "test",
                    "method": method,
                    "action": action,
                    "qid": f"q{idx}",
                    "recall_at_5": 0.1 + idx,
                    "mrr": 0.2 + idx,
                    "ndcg_at_5": 0.3 + idx,
                    "reward": base_reward + idx * 0.01,
                    "rewrite_cost": 0.0,
                    "retrieval_calls": 2 if action.startswith("hybrid") else 1,
                    "state_question_length": 0.2 + idx * 0.2,
                    "state_bm25_top1": top1,
                    "state_bm25_gap": 0.05 + idx * 0.1,
                    "state_bm25_entropy": 0.9 - idx * 0.2,
                    "predicted_action_margin": 0.01 + idx * 0.01,
                    "oracle_reward_margin": 0.02 + idx * 0.01,
                    "oracle_tie_count": 1,
                    "action_reward_std": 0.1,
                    "bm25_dense_doc_overlap": 0.2,
                    "dense_new_doc_rate": 0.8,
                    "hybrid_new_doc_rate": 0.6,
                }
            )
    return rows
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest -q tests/test_complexity_diagnostics.py
```

Expected: FAIL because `selective_rag_rl.complexity_diagnostics` does not exist.

- [ ] **Step 3: Implement diagnostics module**

Create `src/selective_rag_rl/complexity_diagnostics.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BUCKET_FEATURES = [
    "state_question_length",
    "state_bm25_top1",
    "state_bm25_gap",
    "state_bm25_entropy",
    "predicted_action_margin",
    "oracle_reward_margin",
    "action_reward_std",
    "bm25_dense_doc_overlap",
    "dense_new_doc_rate",
    "hybrid_new_doc_rate",
]

METHODS = [
    "Train-best retrieval action",
    "Heuristic retrieval router",
    "Selective retrieval policy",
    "Confidence-gated retrieval policy",
    "Oracle retrieval action",
]


def export_complexity_diagnostics(
    *,
    dataset: str,
    detailed_csv: Path,
    output_csv: Path,
    action_distribution_csv: Path,
    split: str = "test",
) -> tuple[Path, Path]:
    detailed = pd.read_csv(detailed_csv)
    detailed = detailed[detailed["split"] == split].copy()
    available_features = [feature for feature in BUCKET_FEATURES if feature in detailed.columns]
    bucket_rows = []
    action_rows = []
    for feature in available_features:
        method_rows = detailed[detailed["method"].isin(METHODS)].copy()
        method_rows = method_rows[method_rows[feature].notna()].copy()
        if method_rows.empty:
            continue
        method_rows["bucket"] = _bucket_series(method_rows[feature])
        bucket_rows.extend(_metric_rows(dataset, split, feature, method_rows))
        action_rows.extend(_action_distribution_rows(dataset, split, feature, method_rows))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    action_distribution_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bucket_rows).to_csv(output_csv, index=False)
    pd.DataFrame(action_rows).to_csv(action_distribution_csv, index=False)
    return output_csv, action_distribution_csv


def _bucket_series(values: pd.Series) -> pd.Series:
    unique = values.dropna().unique()
    if len(unique) < 3:
        return pd.Series(np.where(values <= values.median(), "low", "high"), index=values.index)
    try:
        return pd.qcut(values, q=3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(np.where(values <= values.median(), "low", "high"), index=values.index)


def _metric_rows(dataset: str, split: str, feature: str, rows: pd.DataFrame) -> list[dict[str, object]]:
    baselines = rows.pivot_table(index=["qid", "bucket"], columns="method", values="reward", aggfunc="first")
    output = []
    for (bucket, method), group in rows.groupby(["bucket", "method"], sort=True):
        train_best = _mean_baseline(baselines, bucket, "Train-best retrieval action")
        heuristic = _mean_baseline(baselines, bucket, "Heuristic retrieval router")
        output.append(
            {
                "dataset": dataset,
                "split": split,
                "bucket_feature": feature,
                "bucket": bucket,
                "method": method,
                "n_queries": int(group["qid"].nunique()),
                "recall_at_5": float(group["recall_at_5"].mean()),
                "mrr": float(group["mrr"].mean()),
                "ndcg_at_5": float(group["ndcg_at_5"].mean()),
                "reward": float(group["reward"].mean()),
                "reward_delta_vs_train_best": float(group["reward"].mean() - train_best),
                "reward_delta_vs_heuristic": float(group["reward"].mean() - heuristic),
                "rewrite_cost": float(group["rewrite_cost"].mean()),
                "retrieval_calls": float(group["retrieval_calls"].mean()),
                "oracle_margin": float(group.get("oracle_reward_margin", pd.Series(dtype=float)).mean()),
                "oracle_tie_count": float(group.get("oracle_tie_count", pd.Series(dtype=float)).mean()),
            }
        )
    return output


def _mean_baseline(baselines: pd.DataFrame, bucket: str, method: str) -> float:
    if method not in baselines.columns:
        return 0.0
    values = baselines.xs(bucket, level="bucket")[method].dropna()
    return float(values.mean()) if not values.empty else 0.0


def _action_distribution_rows(dataset: str, split: str, feature: str, rows: pd.DataFrame) -> list[dict[str, object]]:
    policy = rows[rows["method"] == "Selective retrieval policy"].copy()
    output = []
    for bucket, group in policy.groupby("bucket", sort=True):
        actions = group["action"].astype(str)
        n = int(group["qid"].nunique())
        output.append(
            {
                "dataset": dataset,
                "split": split,
                "bucket_feature": feature,
                "bucket": bucket,
                "n_queries": n,
                "bm25_action_rate": float(actions.str.startswith("bm25").mean()),
                "dense_action_rate": float(actions.str.startswith("dense").mean()),
                "hybrid_action_rate": float(actions.str.startswith("hybrid").mean()),
                "keyword_action_rate": float(actions.str.contains("keyword").mean()),
                "mean_reward": float(group["reward"].mean()),
                "mean_retrieval_calls": float(group["retrieval_calls"].mean()),
            }
        )
    return output
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/run_complexity_diagnostics.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.complexity_diagnostics import export_complexity_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detailed-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--action-distribution-csv", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    bucket_csv, action_csv = export_complexity_diagnostics(
        dataset=args.dataset,
        detailed_csv=args.detailed_csv,
        output_csv=args.output_csv,
        action_distribution_csv=args.action_distribution_csv,
        split=args.split,
    )
    print(pd.read_csv(bucket_csv).head(20).to_string(index=False))
    print(pd.read_csv(action_csv).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify tests**

Run:

```bash
uv run pytest -q tests/test_complexity_diagnostics.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/selective_rag_rl/complexity_diagnostics.py scripts/run_complexity_diagnostics.py tests/test_complexity_diagnostics.py
git commit -m "Add complexity bucket diagnostics"
```

## Task 3: Cross-Dataset Transfer Module

**Files:**
- Create: `src/selective_rag_rl/cross_dataset_transfer.py`
- Create: `scripts/run_cross_dataset_transfer.py`
- Create: `tests/test_cross_dataset_transfer.py`

- [ ] **Step 1: Write focused transfer-policy test**

Create `tests/test_cross_dataset_transfer.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.cross_dataset_transfer import TransferInputs, run_transfer_from_evals


def test_run_transfer_from_evals_applies_source_policy_to_target(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    inputs = TransferInputs(
        source_dataset="source",
        target_dataset="target",
        source_train=_evals([0.0, 1.0, 2.0]),
        target_test=_evals([0.5, 1.5]),
        actions=["left", "right"],
        method_names={"left": "Left fixed", "right": "Right fixed"},
        output_dir=output_dir,
        output_prefix="transfer",
    )

    metadata = run_transfer_from_evals(inputs)
    summary = pd.read_csv(metadata["summary_csv"])
    detailed = pd.read_csv(metadata["detailed_csv"])

    assert "Transferred source policy" in set(summary["method"])
    assert "Target train-best action" in set(summary["method"])
    assert "Oracle retrieval action" in set(summary["method"])
    assert len(detailed[detailed["method"] == "Transferred source policy"]) == 2


def _evals(xs: list[float]) -> list[dict[str, object]]:
    rows = []
    for idx, x in enumerate(xs):
        rows.append(
            {
                "qid": f"q{idx}",
                "question": f"question {idx}",
                "features": [1.0, x],
                "actions": {
                    "left": {
                        "recall_at_5": 0.0,
                        "mrr": 0.0,
                        "ndcg_at_5": 0.0,
                        "rewrite_cost": 0.0,
                        "retrieval_calls": 1,
                        "rewrite_tokens": 1,
                        "reward": 1.0 - x,
                        "queries": "left",
                        "top_docs": "d1",
                        "gold_docs": "d1",
                    },
                    "right": {
                        "recall_at_5": 0.0,
                        "mrr": 0.0,
                        "ndcg_at_5": 0.0,
                        "rewrite_cost": 0.0,
                        "retrieval_calls": 1,
                        "rewrite_tokens": 1,
                        "reward": x,
                        "queries": "right",
                        "top_docs": "d2",
                        "gold_docs": "d1",
                    },
                },
            }
        )
    return rows
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest -q tests/test_cross_dataset_transfer.py
```

Expected: FAIL because `selective_rag_rl.cross_dataset_transfer` does not exist.

- [ ] **Step 3: Implement transfer core**

Create `src/selective_rag_rl/cross_dataset_transfer.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.bandit import DirectMethodBandit
from selective_rag_rl.heuristic_policy import heuristic_retrieval_action
from selective_rag_rl.retrieval_policy_experiment import fit_policy_feature_transform, summarize_retrieval_policy


@dataclass(frozen=True)
class TransferInputs:
    source_dataset: str
    target_dataset: str
    source_train: list[dict[str, object]]
    target_test: list[dict[str, object]]
    actions: list[str]
    method_names: dict[str, str]
    output_dir: Path
    output_prefix: str = "cross_dataset_transfer"


def run_transfer_from_evals(inputs: TransferInputs) -> dict[str, str]:
    transform = fit_policy_feature_transform(np.vstack([row["features"] for row in inputs.source_train]), "full")
    source_features = transform.transform(np.vstack([row["features"] for row in inputs.source_train]))
    rewards = {
        action: [float(row["actions"][action]["reward"]) for row in inputs.source_train]
        for action in inputs.actions
    }
    policy = DirectMethodBandit(actions=inputs.actions, l2=1.0)
    policy.fit(source_features, rewards)
    best_fixed_action = max(inputs.actions, key=lambda action: (float(np.mean(rewards[action])), -inputs.actions.index(action)))

    rows = []
    for target_row in inputs.target_test:
        transformed = transform.transform(np.asarray(target_row["features"], dtype=float))
        transferred_action = policy.predict(transformed)
        heuristic_action = heuristic_retrieval_action(np.asarray(target_row["features"], dtype=float), inputs.actions)
        oracle_action = max(inputs.actions, key=lambda action: target_row["actions"][action]["reward"])
        for action in inputs.actions:
            rows.append(_row(inputs, target_row, inputs.method_names[action], action))
        rows.append(_row(inputs, target_row, "Target train-best action", best_fixed_action))
        rows.append(_row(inputs, target_row, "Heuristic retrieval router", heuristic_action))
        rows.append(_row(inputs, target_row, "Transferred source policy", transferred_action))
        rows.append(_row(inputs, target_row, "Oracle retrieval action", oracle_action))

    detailed = pd.DataFrame(rows)
    results_dir = inputs.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = results_dir / f"{inputs.output_prefix}_detailed.csv"
    summary_csv = results_dir / f"{inputs.output_prefix}_summary.csv"
    detailed.to_csv(detailed_csv, index=False)
    summarize_retrieval_policy(detailed).to_csv(summary_csv, index=False)
    return {"detailed_csv": str(detailed_csv), "summary_csv": str(summary_csv)}


def _row(inputs: TransferInputs, target_row: dict[str, object], method: str, action: str) -> dict[str, object]:
    metrics = target_row["actions"][action]
    return {
        "source_dataset": inputs.source_dataset,
        "target_dataset": inputs.target_dataset,
        "split": "test",
        "method": method,
        "action": action,
        "qid": target_row["qid"],
        "question": target_row["question"],
        **metrics,
    }
```

- [ ] **Step 4: Add real BEIR CLI wrapper**

Create `scripts/run_cross_dataset_transfer.py`. It should parse `--output-dir`, `--num-train-examples`, `--num-test-examples`, `--seed`, `--full-corpus`, `--policy-model` only if needed later, and call a real-data helper added to the same module in Step 5. Start with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.cross_dataset_transfer import run_beir_transfer_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scifact-path", type=Path, default=Path("data/raw/scifact"))
    parser.add_argument("--nfcorpus-path", type=Path, default=Path("data/raw/nfcorpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-train-examples", type=int, default=600)
    parser.add_argument("--num-test-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    metadata = run_beir_transfer_matrix(
        scifact_path=args.scifact_path,
        nfcorpus_path=args.nfcorpus_path,
        output_dir=args.output_dir,
        num_train_examples=args.num_train_examples,
        num_test_examples=args.num_test_examples,
        seed=args.seed,
        full_corpus=args.full_corpus,
        embedder_name=args.embedder,
    )
    print(pd.read_csv(metadata["summary_csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement `run_beir_transfer_matrix`**

Extend `src/selective_rag_rl/cross_dataset_transfer.py` with a real-data helper that:

```python
def run_beir_transfer_matrix(
    *,
    scifact_path: Path,
    nfcorpus_path: Path,
    output_dir: Path,
    num_train_examples: int = 600,
    num_test_examples: int = 300,
    seed: int = 42,
    full_corpus: bool = True,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, str]:
```

Implementation rules:

- Load train/test examples with `load_beir_dataset`.
- Use `FakeDenseEmbedder()` if `embedder_name == "fake"`, otherwise `load_sentence_transformer(embedder_name)`.
- Use `evaluate_retrieval_actions` for train and test examples.
- Call `run_transfer_from_evals` for four pairs:
  - SciFact -> SciFact
  - SciFact -> NFCorpus
  - NFCorpus -> NFCorpus
  - NFCorpus -> SciFact
- Concatenate pair summary CSVs into `outputs/results/cross_dataset_transfer_summary.csv`.
- Concatenate pair detailed CSVs into `outputs/results/cross_dataset_transfer_detailed.csv`.

- [ ] **Step 6: Verify transfer test**

Run:

```bash
uv run pytest -q tests/test_cross_dataset_transfer.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/selective_rag_rl/cross_dataset_transfer.py scripts/run_cross_dataset_transfer.py tests/test_cross_dataset_transfer.py
git commit -m "Add cross-dataset transfer experiment"
```

## Task 4: Artifact Index, README, and Formal Runs

**Files:**
- Modify: `src/selective_rag_rl/artifact_index.py`
- Modify: `tests/test_artifact_index.py`
- Modify: `README.md`
- Generated: `outputs/results/scifact_complexity_buckets.csv`
- Generated: `outputs/results/nfcorpus_complexity_buckets.csv`
- Generated: `outputs/results/scifact_complexity_action_distribution.csv`
- Generated: `outputs/results/nfcorpus_complexity_action_distribution.csv`
- Generated: `outputs/results/cross_dataset_transfer_summary.csv`
- Generated: `outputs/results/cross_dataset_transfer_detailed.csv`

- [ ] **Step 1: Add failing artifact-index test**

Append to `tests/test_artifact_index.py`:

```python
def test_final_project_artifact_specs_include_complexity_and_transfer_outputs() -> None:
    specs = {spec.artifact_id: spec for spec in final_project_artifact_specs(Path("."))}

    assert specs["scifact_complexity_buckets"].path.as_posix() == "outputs/results/scifact_complexity_buckets.csv"
    assert specs["nfcorpus_complexity_buckets"].path.as_posix() == "outputs/results/nfcorpus_complexity_buckets.csv"
    assert specs["cross_dataset_transfer_summary"].path.as_posix() == "outputs/results/cross_dataset_transfer_summary.csv"
    assert "difficulty" in specs["scifact_complexity_buckets"].role
    assert "transfer" in specs["cross_dataset_transfer_summary"].role
```

- [ ] **Step 2: Run artifact test and verify failure**

Run:

```bash
uv run pytest -q tests/test_artifact_index.py::test_final_project_artifact_specs_include_complexity_and_transfer_outputs
```

Expected: FAIL with missing artifact IDs.

- [ ] **Step 3: Add artifact specs**

In `src/selective_rag_rl/artifact_index.py`, add specs near the other diagnostic artifacts:

```python
artifact(
    "scifact_complexity_buckets",
    "diagnostic",
    "outputs/results/scifact_complexity_buckets.csv",
    "SciFact full-corpus query difficulty bucket diagnostics for policy behavior.",
    "uv run python scripts/run_complexity_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_complexity_buckets.csv --action-distribution-csv outputs/results/scifact_complexity_action_distribution.csv",
),
artifact(
    "nfcorpus_complexity_buckets",
    "diagnostic",
    "outputs/results/nfcorpus_complexity_buckets.csv",
    "NFCorpus full-corpus query difficulty bucket diagnostics for policy behavior.",
    "uv run python scripts/run_complexity_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_complexity_buckets.csv --action-distribution-csv outputs/results/nfcorpus_complexity_action_distribution.csv",
),
artifact(
    "cross_dataset_transfer_summary",
    "transfer_check",
    "outputs/results/cross_dataset_transfer_summary.csv",
    "SciFact/NFCorpus cross-dataset transfer summary for source-trained retrieval policies.",
    "uv run python scripts/run_cross_dataset_transfer.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus",
),
```

- [ ] **Step 4: Rerun formal detailed CSVs with feature columns**

Run:

```bash
uv run python scripts/run_retrieval_policy_scifact.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --policy-model auto --confidence-gate-margin 0.0
uv run python scripts/run_retrieval_policy_nfcorpus.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --policy-model auto --confidence-gate-margin 0.0
```

Expected: both complete and detailed CSVs now contain `state_bm25_top1`, `oracle_reward_margin`, and related columns.

- [ ] **Step 5: Generate complexity diagnostics**

Run:

```bash
uv run python scripts/run_complexity_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_complexity_buckets.csv --action-distribution-csv outputs/results/scifact_complexity_action_distribution.csv
uv run python scripts/run_complexity_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_complexity_buckets.csv --action-distribution-csv outputs/results/nfcorpus_complexity_action_distribution.csv
```

Expected: four CSVs are created under `outputs/results/`.

- [ ] **Step 6: Generate cross-dataset transfer matrix**

Start with a smoke run:

```bash
uv run python scripts/run_cross_dataset_transfer.py --num-train-examples 20 --num-test-examples 20 --seed 42 --full-corpus --embedder fake
```

Then run the formal experiment:

```bash
uv run python scripts/run_cross_dataset_transfer.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus
```

Expected: `outputs/results/cross_dataset_transfer_summary.csv` and `outputs/results/cross_dataset_transfer_detailed.csv` exist.

- [ ] **Step 7: Update README commands**

Add these commands near the main SciFact/NFCorpus run commands:

```bash
uv run python scripts/run_complexity_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_complexity_buckets.csv --action-distribution-csv outputs/results/scifact_complexity_action_distribution.csv
uv run python scripts/run_complexity_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_complexity_buckets.csv --action-distribution-csv outputs/results/nfcorpus_complexity_action_distribution.csv
uv run python scripts/run_cross_dataset_transfer.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus
```

- [ ] **Step 8: Regenerate final indexes**

Run:

```bash
uv run python scripts/run_checkpoint_manifest.py --output-csv outputs/results/final_checkpoint_manifest.csv
uv run python scripts/run_artifact_index.py --output-csv outputs/results/final_artifact_index.csv
uv run python scripts/run_evidence_consistency.py --output-csv outputs/results/final_evidence_consistency.csv
```

- [ ] **Step 9: Commit**

```bash
git add README.md src/selective_rag_rl/artifact_index.py tests/test_artifact_index.py outputs/results/scifact_complexity_buckets.csv outputs/results/nfcorpus_complexity_buckets.csv outputs/results/scifact_complexity_action_distribution.csv outputs/results/nfcorpus_complexity_action_distribution.csv outputs/results/cross_dataset_transfer_summary.csv outputs/results/cross_dataset_transfer_detailed.csv outputs/results/final_artifact_index.csv outputs/results/final_checkpoint_manifest.csv outputs/results/final_evidence_consistency.csv outputs/results/scifact_retrieval_policy_detailed.csv outputs/results/nfcorpus_retrieval_policy_detailed.csv outputs/results/scifact_retrieval_policy_summary.csv outputs/results/nfcorpus_retrieval_policy_summary.csv outputs/results/scifact_retrieval_policy_summary.json outputs/results/nfcorpus_retrieval_policy_summary.json outputs/results/scifact_retrieval_policy_table.tex outputs/results/nfcorpus_retrieval_policy_table.tex outputs/results/scifact_retrieval_policy_metadata.json outputs/results/nfcorpus_retrieval_policy_metadata.json outputs/checkpoints/scifact_retrieval_policy.pkl outputs/checkpoints/nfcorpus_retrieval_policy.pkl
git commit -m "Add adaptive RAG diagnostics and transfer outputs"
```

## Task 5: Final Report Interpretation

**Files:**
- Modify: `FINAL_REPORT.md`

- [ ] **Step 1: Read generated summaries**

Run:

```bash
python - <<'PY'
import pandas as pd
for path in [
    "outputs/results/scifact_complexity_buckets.csv",
    "outputs/results/nfcorpus_complexity_buckets.csv",
    "outputs/results/cross_dataset_transfer_summary.csv",
]:
    print("\\n==", path)
    print(pd.read_csv(path).head(20).to_string(index=False))
PY
```

- [ ] **Step 2: Add `Query Difficulty and Policy Behavior` section**

Insert after the current confidence-gate section. Include:

- the bucket feature where policy gain over heuristic is largest for SciFact;
- the bucket feature where policy gain over heuristic is largest for NFCorpus;
- whether dense/hybrid action rate increases in high-uncertainty buckets;
- a caveat if the behavior is dataset-specific.

- [ ] **Step 3: Add `Cross-Dataset Policy Transfer` section**

Insert after the new difficulty section. Include:

- SciFact -> NFCorpus transferred reward versus NFCorpus target-trained reward;
- NFCorpus -> SciFact transferred reward versus SciFact target-trained reward;
- whether transfer beats heuristic router;
- interpretation as cross-domain robustness or negative transfer.

- [ ] **Step 4: Update limitations and next steps**

Revise bullets to mention:

- difficulty diagnostics strengthen behavior analysis;
- cross-dataset transfer is stricter than same-dataset held-out testing;
- poor transfer, if observed, means target-domain calibration is needed.

- [ ] **Step 5: Verify report consistency**

Run:

```bash
uv run python scripts/run_evidence_consistency.py --output-csv outputs/results/final_evidence_consistency.csv
uv run python scripts/run_artifact_index.py --output-csv outputs/results/final_artifact_index.csv
```

- [ ] **Step 6: Commit**

```bash
git add FINAL_REPORT.md outputs/results/final_evidence_consistency.csv outputs/results/final_artifact_index.csv
git commit -m "Document adaptive RAG depth experiments"
```

## Task 6: Final Verification and Push

**Files:**
- All changed files.

- [ ] **Step 1: Run full verification**

Run:

```bash
uv run pytest -q && uv run python -m compileall -q src scripts && git diff --check
```

Expected: `pytest` passes, compileall exits 0, diff check exits 0.

- [ ] **Step 2: Inspect status**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: no unintended untracked temp outputs. If smoke transfer files are separate from formal outputs, delete them before committing.

- [ ] **Step 3: Push**

Run:

```bash
git push
```

Expected: branch `main` pushes successfully.

