# NQ Reader And Repeated API Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the next evidence gaps by restoring Natural Questions data, improving deterministic reader evaluation, and turning Gemini/Vertex pilots into bounded repeated-seed diagnostics without changing unsupported final claims.

**Architecture:** Keep raw data and API outputs local/ignored, commit only reusable code, tests, docs, and sanitized aggregate summaries. Use Hugging Face Hub single-file downloads for NQ, deterministic readers for default tests, and explicit cache/budget gates for all Gemini/Vertex work.

**Tech Stack:** Python, pathlib, pandas, pyarrow, huggingface-hub, pytest, uv, existing reader/retrieval/API budget modules.

---

## Execution Status

Executed on branch `codex/api-reader-fqi-upgrade-20260606-0126`.

Completion note: all tasks in this plan were executed in this pass. The final
verification set was `uv run pytest -q`, `uv run python scripts/run_final_smoke.py
--output-dir outputs/codex_smoke_nq_reader_api --pytest-mode targeted`, and
`git diff --check`.

Planned in this pass:

- Restore one Natural Questions validation parquet shard.
- Add answer-type aware reader improvements and tests.
- Run NQ and larger HotpotQA reader comparisons.
- Run bounded repeated Gemini pilot if cache-miss budget allows.
- Run small cache-first Vertex repeated pilot only if cache/API budget is safe.
- Regenerate evidence dashboard and docs.

## Task 1: NQ Raw-Data Download Path

**Files:**
- Modify: `src/selective_rag_rl/preflight/raw_data_download.py`
- Modify: `tests/test_raw_data_download.py`
- Use: `scripts/download_missing_raw_data.py`

- [ ] Add an NQ Hugging Face Hub spec:
  - dataset key: `nq-validation-shard`
  - repo id: `google-research-datasets/natural_questions`
  - filename: `default/validation-00000-of-00007.parquet`
  - target: `data/raw/natural-questions/default/validation-00000-of-00007.parquet`
  - converter: none, copy parquet directly.

- [ ] Add tests proving `--prefer-hf` copies the mocked parquet to the expected target without using the official URL.

- [ ] Run:

```powershell
uv run pytest tests/test_raw_data_download.py -q
uv run python scripts/download_missing_raw_data.py --dataset nq-validation-shard --dry-run --prefer-hf --output-dir outputs/codex_nq_download_dry
```

- [ ] If dry-run is safe, run:

```powershell
uv run python scripts/download_missing_raw_data.py --dataset nq-validation-shard --prefer-hf --output-dir outputs/codex_nq_download
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight_after_nq
```

## Task 2: Stronger Deterministic Reader

**Files:**
- Modify: `src/selective_rag_rl/core/reader.py`
- Modify: `scripts/run_reader_eval.py`
- Modify: `tests/test_reader.py`
- Modify: `tests/test_reader_eval.py`

- [ ] Add `AnswerTypeHeuristicReader` with:
  - yes/no question handling;
  - date/year extraction;
  - numeric extraction for "how many" questions;
  - capitalized entity fallback;
  - sentence fallback.

- [ ] Add CLI reader option `answer_type`.

- [ ] Run:

```powershell
uv run pytest tests/test_reader.py tests/test_reader_eval.py tests/test_reader_comparison.py -q
```

## Task 3: Real-Data Reader Comparisons

**Files:**
- Add sanitized summaries under `outputs/results/` only.
- Update docs after the runs.

- [ ] Run HotpotQA 200 examples:

```powershell
uv run python scripts/run_reader_comparison.py --dataset hotpot --num-examples 200 --readers lexical,span,answer_type --output-dir outputs/codex_reader_hotpot_realdata_200
```

- [ ] If NQ is available, run NQ 50 examples:

```powershell
uv run python scripts/run_reader_comparison.py --dataset nq --num-examples 50 --readers lexical,span,answer_type --output-dir outputs/codex_reader_nq_realdata_50
```

- [ ] Copy only aggregate summary CSVs to:
  - `outputs/results/hotpot_reader_realdata_200_summary.csv`
  - `outputs/results/nq_reader_realdata_summary.csv`

## Task 4: Repeated Gemini Pilot

**Files:**
- Create optional runner if needed: `scripts/run_repeated_gemini_baseline.py`
- Test: `tests/test_repeated_gemini_baseline.py`
- Add sanitized summary: `outputs/results/hotpot_gemini_repeated_pilot_summary.csv`

- [ ] Implement repeated runner that:
  - accepts seeds and num-examples;
  - runs dry-run first;
  - blocks if total cache misses exceed `--max-new-calls`;
  - writes aggregate per-seed summary only.

- [ ] Run a dry-run for seeds `41,42,43`, 10 examples each.

- [ ] If total misses are within budget, run live with explicit `CODEX_ALLOW_API_CALLS=1` and cap.

## Task 5: Vertex Repeated Semantic Pilot

**Files:**
- Prefer existing scripts; do not add code unless budget/preflight handling is missing.

- [ ] Run cache preflight for small NFCorpus/SciFact semantic runs.
- [ ] If safe, run a tiny repeated semantic pilot with explicit `--semantic-allow-api` and bounded `--semantic-max-new-texts`.
- [ ] Summarize as `api_pilot` only.

## Task 6: Docs, Dashboard, Verification, Push

- [ ] Update:
  - `docs/PROJECT_GAP_STATUS.md`
  - `docs/READER_EXTENSION.md`
  - `docs/API_EXPERIMENTS.md`
  - `docs/FINAL_REPRODUCTION.md`

- [ ] Regenerate:

```powershell
uv run python scripts/run_artifact_index.py --output-csv outputs/results/final_artifact_index.csv
uv run python scripts/run_experiment_dashboard.py --output-csv outputs/results/experiment_dashboard.csv --output-md docs/EXPERIMENT_DASHBOARD.md
uv run python scripts/run_markdown_consistency.py --output-csv outputs/results/final_markdown_consistency.csv
```

- [ ] Verify:

```powershell
uv run pytest -q
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_nq_reader_api --pytest-mode targeted
git diff --check
```

- [ ] Stage only safe files. Do not stage `.env`, credentials, `data/raw`, `outputs/cache`, or `outputs/codex_*`.

- [ ] Commit:

```powershell
git commit -m "eval: add nq reader and repeated api pilots"
git push
```

## Claim Boundary

Allowed if successful:

- NQ data path and reader smoke are available.
- HotpotQA reader comparison is larger and has stronger deterministic baselines.
- Gemini/Vertex repeated pilots are bounded and machine-readable.

Still not allowed unless evidence is strong:

- final downstream RAG answer-quality improvement;
- final Gemini/Vertex superiority;
- production-ready semantic-feature routing.
