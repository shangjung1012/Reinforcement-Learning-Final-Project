# Gemini Reader Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded stronger-reader evaluation path so downstream QA evidence is no longer limited to deterministic lexical/span smoke checks.

**Architecture:** Keep the existing deterministic readers as baselines, add an optional Gemini answer reader behind cache, dry-run, allow-api, and max-new-call gates, and commit only sanitized aggregate summaries. Raw detailed outputs, prompts, API caches, and downloaded datasets stay in ignored local directories.

**Tech Stack:** Python, pandas, BM25 retrieval, Google GenAI through Vertex AI, pytest, existing SQuAD-style EM/F1 metrics.

---

## Claim Boundary

Allowed after this plan:

- The repo has a stronger bounded API reader pilot on HotpotQA/NQ.
- The deterministic readers are clearly baselines/diagnostics, not final answer-generation evidence.
- The evidence dashboard labels Gemini reader results as `api_pilot`.

Not allowed after this plan unless evidence is much larger and compared against proper baselines:

- final downstream RAG answer-quality improvement;
- production-ready reader;
- Gemini reader superiority as a main claim.

## Task 1: Implement Bounded Gemini Reader Evaluation

**Files:**
- Create: `src/selective_rag_rl/experiments/gemini_reader.py`
- Create: `scripts/run_gemini_reader_eval.py`
- Create: `tests/test_gemini_reader.py`

- [ ] **Step 1: Add tests first**

Write tests that verify:

- dry-run writes metadata and does not call the provider;
- live runs block when cache misses exceed `--max-new-calls`;
- a fake provider can produce a Gemini-reader row without seeing a gold answer parameter;
- summaries include deterministic baseline rows plus `Gemini reader`.

Run:

```powershell
uv run pytest tests/test_gemini_reader.py -q
```

Expected before implementation: fail because the module/script does not exist.

- [ ] **Step 2: Implement cache, prompt, budget gate, and evaluator**

Implement:

- `GeminiReaderBudgetError`
- `GeminiReaderCache`
- `VertexGeminiAnswerReader`
- `evaluate_gemini_reader(...)`

The evaluator must:

- load `toy`, `hotpot`, or `nq` examples;
- retrieve BM25 top-k passages;
- run lexical/span/answer_type baselines locally;
- run Gemini only when cache misses are within explicit budget and allow-api is set;
- write detailed CSV under the caller output dir;
- write sanitized aggregate summary CSV/JSON;
- never pass gold answers into the Gemini provider.

- [ ] **Step 3: Verify unit tests**

Run:

```powershell
uv run pytest tests/test_gemini_reader.py -q
uv run pytest tests/test_reader.py tests/test_reader_eval.py -q
```

Expected: all pass.

## Task 2: Run Real-Data Gemini Reader Pilots

**Files:**
- Generated ignored outputs: `outputs/codex_gemini_reader_*`
- Commit sanitized summaries only:
  - `outputs/results/hotpot_gemini_reader_pilot_summary.csv`
  - `outputs/results/nq_gemini_reader_pilot_summary.csv`

- [ ] **Step 1: Dry-run HotpotQA and NQ**

```powershell
uv run python scripts/run_gemini_reader_eval.py --dataset hotpot --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_hotpot.jsonl --dry-run --output-dir outputs/codex_gemini_reader_hotpot_dry
uv run python scripts/run_gemini_reader_eval.py --dataset nq --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_nq.jsonl --dry-run --output-dir outputs/codex_gemini_reader_nq_dry
```

- [ ] **Step 2: Live runs only if dry-run misses are within cap**

```powershell
$env:CODEX_ALLOW_API_CALLS='1'; uv run python scripts/run_gemini_reader_eval.py --dataset hotpot --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_hotpot.jsonl --allow-api --max-new-calls 40 --output-dir outputs/codex_gemini_reader_hotpot_40; Remove-Item Env:\CODEX_ALLOW_API_CALLS
$env:CODEX_ALLOW_API_CALLS='1'; uv run python scripts/run_gemini_reader_eval.py --dataset nq --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_nq.jsonl --allow-api --max-new-calls 40 --output-dir outputs/codex_gemini_reader_nq_40; Remove-Item Env:\CODEX_ALLOW_API_CALLS
```

- [ ] **Step 3: Copy sanitized summaries**

Copy only `gemini_reader_summary.csv` into `outputs/results/`.

## Task 3: Evidence Index, Dashboard, And Docs

**Files:**
- Modify: `src/selective_rag_rl/reports/artifact_index.py`
- Modify: `tests/test_artifact_index.py`
- Modify: `docs/API_EXPERIMENTS.md`
- Modify: `docs/READER_EXTENSION.md`
- Modify: `docs/PROJECT_GAP_STATUS.md`
- Modify: `docs/FINAL_REPRODUCTION.md`

- [ ] Add artifact specs for the two Gemini reader pilot summaries.
- [ ] Verify dashboard labels them `api_pilot` and `claim_allowed=false`.
- [ ] Update docs with exact pilot results and limitations.
- [ ] Regenerate:

```powershell
uv run python scripts/run_artifact_index.py --output-csv outputs/results/final_artifact_index.csv
uv run python scripts/run_experiment_dashboard.py --output-csv outputs/results/experiment_dashboard.csv --output-md docs/EXPERIMENT_DASHBOARD.md
```

## Task 4: Final Verification And Push

- [ ] Run:

```powershell
uv run pytest -q
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_gemini_reader --pytest-mode targeted
git diff --check
```

- [ ] Check staged safety:

```powershell
$bad = git diff --cached --name-only | Where-Object { $_ -match '(^|/)\.env$|credentials|application_default_credentials|outputs/cache|^data/raw|outputs/codex_' }; if ($bad) { $bad; exit 1 } else { 'staged safety check passed' }
```

- [ ] Commit:

```powershell
git commit -m "eval: add bounded gemini reader pilots"
git push
```
