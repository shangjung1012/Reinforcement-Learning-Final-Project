# Policy Depth and Embedding Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the final project stronger by replacing shallow query-form signals with learned embedding state features, adding stronger policy diagnostics, and producing report-ready evidence that the RL policy is doing nontrivial action selection.

**Architecture:** Keep the current offline contextual-bandit retrieval-action setup. Add cached embedding-derived policy features as an optional state featurizer, then run controlled ablations and diagnostics without changing the action/reward definition. This preserves the current project scope while improving technical depth.

**Tech Stack:** Python, uv, pandas, numpy, scikit-learn, pytest, Vertex AI embedding cache, existing BEIR SciFact/NFCorpus loaders.

---

## File Structure

- Modify `src/selective_rag_rl/retrieval_policy_experiment.py`: expose richer semantic/embedding policy features through the existing `semantic_features` path and include feature metadata in checkpoints.
- Modify `src/selective_rag_rl/vertex_embeddings.py`: keep API access and JSONL caching isolated from policy code.
- Create `src/selective_rag_rl/policy_diagnostics.py`: reusable policy margin, regret, and action-confusion diagnostics.
- Create `scripts/run_policy_diagnostics.py`: CLI for writing diagnostic CSVs from detailed experiment outputs.
- Modify `src/selective_rag_rl/feature_ablation.py`: support semantic feature-set ablations after embedding features are enabled.
- Modify `tests/test_retrieval_policy_experiment.py`: regression tests for feature length, metadata, and feature masking.
- Create `tests/test_policy_diagnostics.py`: tests for margin/regret/action diagnostics.
- Modify `README.md` and `FINAL_REPORT.md`: add commands, tables, and final interpretation.

---

### Task 1: Strengthen Embedding State Features

**Files:**
- Modify: `src/selective_rag_rl/retrieval_policy_experiment.py`
- Test: `tests/test_retrieval_policy_experiment.py`

- [ ] **Step 1: Write failing test for richer semantic feature length**

Add this test near `test_run_retrieval_policy_can_use_semantic_features`:

```python
def test_run_retrieval_policy_records_extended_semantic_feature_width(tmp_path: Path) -> None:
    data_path = tmp_path / "hotpot.json"
    output_dir = tmp_path / "outputs"
    data_path.write_text(json.dumps([_example(i) for i in range(10)]), encoding="utf-8")
    examples = load_hotpotqa(data_path, num_examples=10, seed=7)

    metadata = run_retrieval_policy_on_examples(
        examples=examples,
        output_dir=output_dir,
        dataset_name="test semantic retrieval policy",
        output_prefix="semantic_policy",
        checkpoint_name="semantic_policy.pkl",
        embedder_name="fake",
        semantic_features="fake",
        semantic_embedder=_FakeSemanticEmbedder(),
        knn_k_candidates=[1],
        tuning_folds=2,
    )

    checkpoint = load_checkpoint(Path(metadata["outputs"]["checkpoint"]))
    assert checkpoint["metadata"]["feature_width"] >= 23
    assert checkpoint["model"].mean.shape[0] == checkpoint["metadata"]["feature_width"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_retrieval_policy_experiment.py::test_run_retrieval_policy_records_extended_semantic_feature_width -q
```

Expected: FAIL because `feature_width` is missing or still uses the old 19-feature width.

- [ ] **Step 3: Extend `_semantic_state_features`**

In `src/selective_rag_rl/retrieval_policy_experiment.py`, extend the returned list to include rank-shape signals:

```python
    return [
        top1,
        float(np.mean(similarities)),
        float(np.max(similarities)),
        top1 - top2,
        float(np.mean(similarities > 0.0)),
        float(np.std(similarities)),
        float(np.min(similarities)),
        float(np.median(similarities)),
        float(ranked[0] - ranked[-1]),
    ]
```

Also add feature width to checkpoint and metadata after `features` is built:

```python
feature_width = int(features.shape[1])
```

Then include:

```python
"feature_width": feature_width,
```

in both `save_checkpoint(... metadata ...)` and the returned `metadata`.

- [ ] **Step 4: Run target test**

Run:

```bash
uv run pytest tests/test_retrieval_policy_experiment.py::test_run_retrieval_policy_records_extended_semantic_feature_width -q
```

Expected: PASS.

- [ ] **Step 5: Run full retrieval policy tests**

Run:

```bash
uv run pytest tests/test_retrieval_policy_experiment.py -q
```

Expected: all tests pass.

---

### Task 2: Add Policy Diagnostics Beyond Mean Reward

**Files:**
- Create: `src/selective_rag_rl/policy_diagnostics.py`
- Create: `tests/test_policy_diagnostics.py`
- Create: `scripts/run_policy_diagnostics.py`

- [ ] **Step 1: Write diagnostics tests**

Create `tests/test_policy_diagnostics.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.policy_diagnostics import export_policy_diagnostics


def test_export_policy_diagnostics_writes_regret_and_action_match(tmp_path: Path) -> None:
    detailed_csv = tmp_path / "detailed.csv"
    output_csv = tmp_path / "diagnostics.csv"
    pd.DataFrame(
        [
            _row("q1", "Selective retrieval policy", "dense_keep", 1.0),
            _row("q1", "Train-best retrieval action", "bm25_keep", 0.8),
            _row("q1", "Oracle retrieval action", "dense_keep", 1.2),
            _row("q2", "Selective retrieval policy", "bm25_keep", 0.4),
            _row("q2", "Train-best retrieval action", "bm25_keep", 0.4),
            _row("q2", "Oracle retrieval action", "hybrid_keep", 0.9),
        ]
    ).to_csv(detailed_csv, index=False)

    exported = export_policy_diagnostics(detailed_csv, output_csv, dataset="toy")
    diagnostics = pd.read_csv(exported)

    assert list(diagnostics["qid"]) == ["q1", "q2"]
    assert list(diagnostics["policy_regret"]) == [0.2, 0.5]
    assert list(diagnostics["beats_train_best"]) == [True, False]
    assert list(diagnostics["matches_oracle_action"]) == [True, False]


def _row(qid: str, method: str, action: str, reward: float) -> dict[str, object]:
    return {
        "split": "test",
        "qid": qid,
        "question": f"question {qid}",
        "method": method,
        "action": action,
        "reward": reward,
        "recall_at_5": reward,
        "mrr": reward,
        "ndcg_at_5": reward,
        "retrieval_calls": 1,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_policy_diagnostics.py -q
```

Expected: FAIL because `selective_rag_rl.policy_diagnostics` does not exist.

- [ ] **Step 3: Implement diagnostics module**

Create `src/selective_rag_rl/policy_diagnostics.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

DIAGNOSTIC_COLUMNS = [
    "dataset",
    "qid",
    "question",
    "selected_action",
    "train_best_action",
    "oracle_action",
    "policy_reward",
    "train_best_reward",
    "oracle_reward",
    "policy_regret",
    "beats_train_best",
    "matches_train_best_action",
    "matches_oracle_action",
]


def export_policy_diagnostics(detailed_csv: Path, output_csv: Path, dataset: str) -> Path:
    detailed = pd.read_csv(detailed_csv)
    test_rows = detailed[detailed["split"] == "test"]
    rows = []
    for _, group in test_rows.groupby("qid", sort=False):
        policy = _first(group, "Selective retrieval policy")
        train_best = _first(group, "Train-best retrieval action")
        oracle = _first(group, "Oracle retrieval action")
        if policy is None or train_best is None or oracle is None:
            continue
        policy_reward = float(policy["reward"])
        train_best_reward = float(train_best["reward"])
        oracle_reward = float(oracle["reward"])
        selected_action = str(policy["action"])
        train_best_action = str(train_best["action"])
        oracle_action = str(oracle["action"])
        rows.append(
            {
                "dataset": dataset,
                "qid": policy["qid"],
                "question": policy["question"],
                "selected_action": selected_action,
                "train_best_action": train_best_action,
                "oracle_action": oracle_action,
                "policy_reward": policy_reward,
                "train_best_reward": train_best_reward,
                "oracle_reward": oracle_reward,
                "policy_regret": oracle_reward - policy_reward,
                "beats_train_best": policy_reward > train_best_reward,
                "matches_train_best_action": selected_action == train_best_action,
                "matches_oracle_action": selected_action == oracle_action,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def _first(group: pd.DataFrame, method: str) -> pd.Series | None:
    rows = group[group["method"] == method]
    if rows.empty:
        return None
    return rows.iloc[0]
```

- [ ] **Step 4: Add CLI**

Create `scripts/run_policy_diagnostics.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selective_rag_rl.policy_diagnostics import export_policy_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detailed-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    detailed_csv = args.detailed_csv or Path("outputs") / "results" / f"{args.dataset}_retrieval_policy_detailed.csv"
    output_csv = args.output_csv or Path("outputs") / "results" / f"{args.dataset}_policy_diagnostics.csv"
    csv_path = export_policy_diagnostics(detailed_csv, output_csv, dataset=args.dataset)
    print(pd.read_csv(csv_path).describe(include="all").to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run diagnostics tests**

Run:

```bash
uv run pytest tests/test_policy_diagnostics.py -q
```

Expected: PASS.

---

### Task 3: Run Embedding and Diagnostics Experiments

**Files:**
- Create output: `outputs/results/nfcorpus_semantic_retrieval_policy_summary.csv`
- Create output: `outputs/results/nfcorpus_policy_diagnostics.csv`
- Create output: `outputs/results/scifact_policy_diagnostics.csv`

- [ ] **Step 1: Run NFCorpus semantic-feature policy**

Run:

```bash
uv run python scripts/run_retrieval_policy_nfcorpus.py \
  --num-train-examples 600 \
  --num-test-examples 300 \
  --seed 42 \
  --full-corpus \
  --policy-model ridge \
  --semantic-features vertex \
  --semantic-cache-path outputs/cache/nfcorpus_vertex_embeddings.jsonl \
  --output-dir outputs/semantic_nfcorpus
```

Expected: summary CSV under `outputs/semantic_nfcorpus/results/`.

- [ ] **Step 2: Run feature ablation with semantic features**

Run:

```bash
uv run python scripts/run_feature_ablation.py \
  --dataset nfcorpus \
  --feature-sets full,no_query,no_retrieval,no_wh,retrieval_only \
  --num-train-examples 300 \
  --num-test-examples 150 \
  --seed 42 \
  --full-corpus \
  --policy-model ridge \
  --semantic-features vertex \
  --semantic-cache-path outputs/cache/nfcorpus_ablation_vertex_embeddings.jsonl
```

Expected: `outputs/results/nfcorpus_feature_ablation.csv` updated with embedding-backed features.

- [ ] **Step 3: Export policy diagnostics**

Run:

```bash
uv run python scripts/run_policy_diagnostics.py \
  --dataset scifact \
  --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv \
  --output-csv outputs/results/scifact_policy_diagnostics.csv

uv run python scripts/run_policy_diagnostics.py \
  --dataset nfcorpus \
  --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv \
  --output-csv outputs/results/nfcorpus_policy_diagnostics.csv
```

Expected: both diagnostics CSVs are written.

---

### Task 4: Update Report With Stronger Evidence

**Files:**
- Modify: `FINAL_REPORT.md`
- Modify: `README.md`

- [ ] **Step 1: Add diagnostics commands to README**

Add:

```bash
uv run python scripts/run_policy_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_policy_diagnostics.csv
uv run python scripts/run_policy_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_policy_diagnostics.csv
```

- [ ] **Step 2: Add report section**

In `FINAL_REPORT.md`, add a section after qualitative cases:

```markdown
### Policy Regret Diagnostics

The diagnostics file measures per-query regret against the oracle retrieval action,
whether the policy beats the train-best fixed action, and whether it selects the same
action as the oracle. This complements mean reward by showing whether the policy
has learned useful per-query switching behavior.
```

Then add the computed aggregate numbers from:

```bash
uv run python - <<'PY'
import pandas as pd
for path in ["outputs/results/scifact_policy_diagnostics.csv", "outputs/results/nfcorpus_policy_diagnostics.csv"]:
    df = pd.read_csv(path)
    print(path)
    print("mean_regret", df["policy_regret"].mean())
    print("beats_train_best_rate", df["beats_train_best"].mean())
    print("matches_oracle_rate", df["matches_oracle_action"].mean())
PY
```

- [ ] **Step 3: Run verification**

Run:

```bash
uv run pytest -q
uv run python -m compileall -q src scripts
git diff --check
```

Expected: all pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md FINAL_REPORT.md scripts/run_policy_diagnostics.py src/selective_rag_rl/policy_diagnostics.py tests/test_policy_diagnostics.py outputs/results/*policy_diagnostics.csv
git commit -m "Add policy regret diagnostics"
```

---

## Execution Recommendation

Do these in order:

1. Task 2 first if you want a quick high-confidence improvement with no API calls.
2. Task 1 and Task 3 only if Vertex embedding quota/cost is acceptable.
3. Task 4 after the experiment outputs exist.

The highest-value next commit is Task 2 plus Task 4. The highest-technical-depth path is Task 1 plus semantic feature ablation, but it depends on Vertex API latency and quota.
