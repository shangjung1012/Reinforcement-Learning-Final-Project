# Code Hardening And Evidence Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the project code, not just branch hygiene, by removing a noisy semantic-feature warning and adding a tested evidence dashboard that classifies experiment artifacts by claim strength.

**Architecture:** Keep changes narrow. The warning fix stays inside the semantic rank-agreement helper. The dashboard is a new read-only module plus CLI that scans existing CSV/JSON artifacts, infers evidence levels, and writes a CSV/Markdown summary without requiring raw data or API calls.

**Tech Stack:** Python, pytest, pandas, argparse, pathlib, existing `uv run pytest` workflow.

---

## Remaining Work After This Plan

Even after this code hardening pass, the project still needs:

1. A human decision on whether to run live Gemini/Vertex one-call smoke with explicit quota approval.
2. HotpotQA/NQ raw data if real downstream reader evaluation is required.
3. Repeated full-data reruns before changing final benchmark claims.
4. A human review of whether to merge this branch into `main`.

## Human Assistance Needed

No help is needed for this plan. I will not make live API calls, delete data, or change final claims. Human help is only needed later for quota, missing raw QA data, or final-claim edits.

---

### Task 1: Fix Constant Semantic Rank Correlation Warning

**Files:**
- Modify: `src/selective_rag_rl/retrieval_policy_experiment.py`
- Modify: `tests/test_retrieval_policy_experiment.py`

- [x] **Step 1: Write failing warning regression test**

Add this test near the existing semantic rank tests:

```python
def test_semantic_rank_agreement_handles_constant_scores_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        features = rpe._semantic_rank_agreement(np.ones(4), k=4)

    assert features[3] == 0.0
```

Also add `import warnings` at the top of the test file.

- [x] **Step 2: Verify red**

Run:

```powershell
uv run pytest tests/test_retrieval_policy_experiment.py::test_semantic_rank_agreement_handles_constant_scores_without_warning -q
```

Expected: failure caused by pandas/scipy constant-input warning being treated as an error.

- [x] **Step 3: Implement minimal warning-free correlation**

Add a small helper:

```python
def _safe_spearman_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=float).reshape(-1)
    right_values = np.asarray(right, dtype=float).reshape(-1)
    if left_values.size < 2 or right_values.size < 2:
        return 0.0
    if np.allclose(left_values, left_values[0]) or np.allclose(right_values, right_values[0]):
        return 0.0
    corr = pd.Series(left_values).corr(pd.Series(right_values), method="spearman")
    return 0.0 if pd.isna(corr) else float(corr)
```

Use it inside `_semantic_rank_agreement`.

- [x] **Step 4: Verify green**

Run:

```powershell
uv run pytest tests/test_retrieval_policy_experiment.py::test_semantic_rank_agreement_handles_constant_scores_without_warning tests/test_retrieval_policy_experiment.py::test_semantic_state_features_can_use_deeper_rank_profile -q
```

Expected: both tests pass and no constant-input warning is emitted.

---

### Task 2: Add Experiment Dashboard Module And CLI

**Files:**
- Create: `src/selective_rag_rl/experiment_dashboard.py`
- Create: `scripts/run_experiment_dashboard.py`
- Create: `tests/test_experiment_dashboard.py`
- Modify: `docs/CODEX_REVIEW_SUMMARY.md`

- [x] **Step 1: Write failing module tests**

Create tests that expect:

```python
from selective_rag_rl.experiment_dashboard import build_experiment_dashboard
```

Test cases:

- smoke manifest under `outputs/codex_smoke/smoke_manifest.json` becomes `smoke_synthetic`, `uses_raw_data=False`, `claim_allowed=False`;
- final `outputs/results/scifact_retrieval_policy_summary.csv` becomes `full_benchmark`, dataset `scifact`, `claim_allowed=True`;
- dry-run API preflight JSON becomes `api_preflight`, `uses_external_api=False`, `claim_allowed=False`;
- tiny real-data metadata JSON becomes `tiny_realdata`, with train/test counts preserved.

- [x] **Step 2: Verify red**

Run:

```powershell
uv run pytest tests/test_experiment_dashboard.py -q
```

Expected: import failure because the module does not exist.

- [x] **Step 3: Implement dashboard module**

Create:

```python
@dataclass(frozen=True)
class ExperimentArtifact:
    artifact_path: str
    dataset: str
    experiment_type: str
    evidence_level: str
    uses_raw_data: bool
    uses_external_api: bool
    uses_model_download: bool
    num_train_examples: int | None
    num_test_examples: int | None
    seed: int | None
    policy_model: str
    feature_set: str
    claim_allowed: bool
    notes: str
```

Expose:

```python
def build_experiment_dashboard(input_paths: list[Path], output_csv: Path, output_md: Path | None = None) -> dict[str, object]:
    ...
```

Implementation rules:

- recursively discover `.csv` and `.json`;
- never read raw data under `data/raw`;
- infer dataset from metadata first, then filename/path;
- classify evidence levels using path and content:
  - `codex_smoke` -> `smoke_synthetic`;
  - reader toy summary -> `smoke_toy_reader`;
  - `codex_api_preflight` -> `api_preflight`;
  - `codex_gemini` or `codex_vertex` -> `api_pilot`;
  - `codex_realdata_smoke` -> `tiny_realdata`;
  - `outputs/results/*retrieval_policy_summary.csv` for SciFact/NFCorpus -> `full_benchmark`;
  - final claims/artifact index CSVs -> `final_claim`;
- `claim_allowed=True` only for `full_benchmark` and `final_claim`.

- [x] **Step 4: Implement CLI**

Create `scripts/run_experiment_dashboard.py` with:

```python
parser.add_argument("--input", type=Path, action="append", default=[Path("outputs/results")])
parser.add_argument("--output-csv", type=Path, default=Path("outputs/results/experiment_dashboard.csv"))
parser.add_argument("--output-md", type=Path, default=Path("docs/EXPERIMENT_DASHBOARD.md"))
```

Print the returned JSON summary.

- [x] **Step 5: Verify green**

Run:

```powershell
uv run pytest tests/test_experiment_dashboard.py -q
uv run python scripts/run_experiment_dashboard.py --input outputs/results --input outputs/codex_smoke_overnight --input outputs/codex_api_preflight_overnight --input outputs/codex_realdata_smoke_overnight --output-csv outputs/codex_diagnostics_overnight/experiment_dashboard.csv --output-md docs/EXPERIMENT_DASHBOARD.md
```

Expected: tests pass; CLI writes dashboard CSV and Markdown.

---

### Task 3: Final Verification And Push

**Files:**
- Modify: `docs/codex_runs/20260602_063712/final_report_for_human.md`
- Modify: `docs/codex_runs/latest_status.md`

- [x] **Step 1: Run targeted tests**

```powershell
uv run pytest tests/test_experiment_dashboard.py tests/test_retrieval_policy_experiment.py::test_semantic_rank_agreement_handles_constant_scores_without_warning -q
```

- [x] **Step 2: Run full tests and smoke**

```powershell
uv run pytest -q
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_code_hardening --pytest-mode targeted
```

- [x] **Step 3: Check secret/data safety**

```powershell
git status --short --branch --ignored
git diff --stat
git diff --check
git ls-files .env data/raw outputs/cache outputs/codex_diagnostics_overnight outputs/codex_smoke_code_hardening
```

Expected: no `.env`, credentials, raw data, cache, or generated outputs are tracked.

- [ ] **Step 4: Commit and push**

```powershell
git add src/selective_rag_rl/experiment_dashboard.py scripts/run_experiment_dashboard.py tests/test_experiment_dashboard.py src/selective_rag_rl/retrieval_policy_experiment.py tests/test_retrieval_policy_experiment.py docs/EXPERIMENT_DASHBOARD.md docs/CODEX_REVIEW_SUMMARY.md docs/codex_runs/latest_status.md docs/codex_runs/20260602_063712/final_report_for_human.md
git commit -m "eval: add experiment evidence dashboard"
git push
```

Expected: branch `codex/overnight-improvements-20260602-0636` is pushed with code improvements.
