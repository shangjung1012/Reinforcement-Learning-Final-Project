# Vertex Embedding Budget Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent accidental Vertex embedding API spend when semantic features are enabled, while keeping cache-only and fake-embedder tests deterministic.

**Architecture:** Add a strict budget gate inside `VertexTextEmbeddingProvider`, then thread explicit semantic API controls through the main retrieval-policy experiment entry points and user-facing retrieval scripts. The provider remains the single enforcement point: callers can estimate workloads with existing embedding preflight scripts, but live cache misses require both `allow_api=True` and a positive `max_new_texts` budget.

**Tech Stack:** Python 3, uv, pytest, numpy, Google GenAI client used only behind the existing optional Vertex provider.

---

### Task 1: Provider-Level Budget Gate

**Files:**
- Modify: `src/selective_rag_rl/vertex_embeddings.py`
- Create: `tests/test_vertex_embeddings.py`

- [ ] **Step 1: Write failing tests**

Add tests that create a cached embedding row, instantiate `VertexTextEmbeddingProvider` with injected `fetcher` functions, and assert:

```python
def test_vertex_provider_returns_cached_embedding_without_live_client(tmp_path: Path) -> None:
    ...

def test_vertex_provider_blocks_cache_miss_without_allow_api(tmp_path: Path) -> None:
    ...

def test_vertex_provider_blocks_cache_miss_above_budget(tmp_path: Path) -> None:
    ...

def test_vertex_provider_fetches_and_caches_within_budget(tmp_path: Path) -> None:
    ...

def test_vertex_provider_dry_run_reports_misses_without_fetching(tmp_path: Path) -> None:
    ...
```

Run:

```powershell
uv run pytest tests/test_vertex_embeddings.py -q
```

Expected: tests fail because `VertexEmbeddingBudgetError`, injected fetcher support, dry-run behavior, and budget parameters do not exist yet.

- [ ] **Step 2: Implement minimal provider changes**

Add:

```python
EmbeddingFetcher = Callable[[list[str], str], list[np.ndarray]]

class VertexEmbeddingBudgetError(RuntimeError):
    def __init__(self, message: str, *, missing: int, allowed: int, dry_run: bool) -> None:
        ...
```

Extend `VertexTextEmbeddingProvider.__init__` with:

```python
allow_api: bool = False
max_new_texts: int = 0
dry_run: bool = False
fetcher: EmbeddingFetcher | None = None
```

Move Google client construction into a lazy `_client()` helper so cache-only and dry-run paths do not import or instantiate the live client. In `embed_texts`, compute ordered unique missing texts, update cache hit/miss counters, and call `_check_budget` before any live fetch. If `dry_run=True` and misses exist, raise `VertexEmbeddingBudgetError` before calling the fetcher.

- [ ] **Step 3: Verify provider tests**

Run:

```powershell
uv run pytest tests/test_vertex_embeddings.py -q
```

Expected: provider tests pass.

### Task 2: Thread Controls Through Main Retrieval Experiments

**Files:**
- Modify: `src/selective_rag_rl/retrieval_policy_experiment.py`
- Modify only if needed: `src/selective_rag_rl/bandit_baselines.py`, `src/selective_rag_rl/feature_ablation.py`, `src/selective_rag_rl/policy_model_sweep.py`
- Test: `tests/test_retrieval_policy_experiment.py`

- [ ] **Step 1: Write failing integration tests**

Add or extend tests to assert `_load_semantic_embedder("vertex", ...)` passes through:

```python
semantic_allow_api=True
semantic_max_new_texts=12
semantic_dry_run=False
```

and that default loading blocks live cache misses unless explicitly allowed.

Run:

```powershell
uv run pytest tests/test_retrieval_policy_experiment.py -q
```

Expected: the new tests fail because the controls are not part of the function signatures.

- [ ] **Step 2: Add semantic API parameters**

Add these keyword parameters to main experiment functions that can create a Vertex semantic embedder:

```python
semantic_allow_api: bool = False
semantic_max_new_texts: int = 0
semantic_dry_run: bool = False
```

Pass them into `_load_semantic_embedder`, and pass through to `VertexTextEmbeddingProvider`. Keep `semantic_features="none"` behavior unchanged.

- [ ] **Step 3: Verify retrieval-policy tests**

Run:

```powershell
uv run pytest tests/test_retrieval_policy_experiment.py -q
```

Expected: tests pass.

### Task 3: Add CLI Flags For Safe Semantic Runs

**Files:**
- Modify: `scripts/run_retrieval_policy_hotpot.py`
- Modify: `scripts/run_llm_retrieval_policy_hotpot.py`
- Modify: `scripts/run_retrieval_policy_scifact.py`
- Modify: `scripts/run_retrieval_policy_nfcorpus.py`
- Modify: `scripts/run_retrieval_policy_nq.py`
- Modify opportunistically if simple: `scripts/run_policy_model_sweep.py`, `scripts/run_feature_ablation.py`, `scripts/run_bandit_baselines.py`

- [ ] **Step 1: Add flags consistently**

Add:

```python
parser.add_argument("--semantic-allow-api", action="store_true")
parser.add_argument("--semantic-max-new-texts", type=int, default=0)
parser.add_argument("--semantic-dry-run", action="store_true")
```

Pass the values into the matching run function. Do not change defaults: without flags, Vertex semantic cache misses still fail before API use.

- [ ] **Step 2: Verify CLI help**

Run:

```powershell
uv run python scripts/run_retrieval_policy_nfcorpus.py --help
uv run python scripts/run_retrieval_policy_scifact.py --help
```

Expected: help includes the new semantic API flags.

### Task 4: Documentation And Verification

**Files:**
- Modify: `docs/API_EXPERIMENTS.md`
- Modify: `docs/FINAL_REPRODUCTION.md`

- [ ] **Step 1: Document safe Vertex semantic command flow**

Add the recommended flow:

```powershell
uv run python scripts/run_embedding_preflight.py --dataset nfcorpus --num-train-examples 10 --num-test-examples 10 --cache-path outputs/cache/codex_nfcorpus_vertex_embeddings.jsonl

uv run python scripts/run_retrieval_policy_nfcorpus.py --semantic-features vertex --semantic-cache-path outputs/cache/codex_nfcorpus_vertex_embeddings.jsonl --semantic-max-new-texts 0

$env:CODEX_ALLOW_API_CALLS='1'; uv run python scripts/run_retrieval_policy_nfcorpus.py --semantic-features vertex --semantic-cache-path outputs/cache/codex_nfcorpus_vertex_embeddings.jsonl --semantic-allow-api --semantic-max-new-texts 50
```

Clarify that `--semantic-allow-api` is only for explicitly budgeted live Vertex embedding runs, and that smoke/default tests still make no API calls.

- [ ] **Step 2: Run verification**

Run:

```powershell
uv run pytest tests/test_vertex_embeddings.py -q
uv run pytest tests/test_retrieval_policy_experiment.py -q
uv run pytest -q
git status --short
git diff --stat
```

Expected: tests pass, `.env` and caches are not staged.

- [ ] **Step 3: Commit and push**

Run:

```powershell
git add src/selective_rag_rl/vertex_embeddings.py src/selective_rag_rl/retrieval_policy_experiment.py src/selective_rag_rl/bandit_baselines.py src/selective_rag_rl/feature_ablation.py src/selective_rag_rl/policy_model_sweep.py scripts tests docs/API_EXPERIMENTS.md docs/FINAL_REPRODUCTION.md docs/superpowers/plans/2026-05-31-vertex-embedding-budget-gate.md
git commit -m "api: gate vertex embedding calls"
git push -u origin codex/api-validation-improvements-20260529-2211
```

Expected: coherent commit pushed to the existing improvement branch.
