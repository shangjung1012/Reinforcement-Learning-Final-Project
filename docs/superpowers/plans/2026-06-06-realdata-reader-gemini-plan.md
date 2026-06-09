# Real-Data Reader And Gemini Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the project from toy downstream-reader smoke toward bounded real-data QA/API evidence without changing the conservative final retrieval-stage claim.

**Architecture:** Add a Windows-friendly missing-data downloader for ignored raw datasets, then run real-data HotpotQA reader comparison and bounded Gemini rewrite baseline only when data/API gates pass. Keep all raw data, API caches, and `outputs/codex_*` artifacts out of git; commit only reusable scripts, tests, docs, and deliberately small sanitized result summaries.

**Tech Stack:** Python, pathlib, urllib, pandas, pytest, uv, existing HotpotQA loader, existing Gemini/Vertex cache and API budget gates.

---

## Execution Status

Executed on branch `codex/api-reader-fqi-upgrade-20260606-0126`.

Completed:

- Added `scripts/download_missing_raw_data.py` and
  `src/selective_rag_rl/preflight/raw_data_download.py`.
- Restored HotpotQA dev distractor locally via Hugging Face parquet mirror and
  converted it into the existing JSON loader schema.
- Ran HotpotQA 50-example lexical/span reader comparison.
- Ran a bounded Gemini HotpotQA pilot with 8 new Gemini calls.
- Added sanitized aggregate summaries under `outputs/results/`.
- Regenerated artifact index, experiment dashboard, and markdown consistency.
- Verified with full pytest and final smoke.

Remaining:

- Natural Questions validation shard is still missing locally.
- Reader/Gemini evidence remains tiny/pilot evidence, not final QA or
  generated-action benchmark evidence.

## File Map

- Create `src/selective_rag_rl/preflight/raw_data_download.py`: cross-platform missing raw-data downloader with dry-run, size/status metadata, and no raw-content logging.
- Create `scripts/download_missing_raw_data.py`: CLI wrapper for downloader.
- Create `tests/test_raw_data_download.py`: mocked downloader tests that never touch network.
- Modify `docs/FINAL_REPRODUCTION.md`: document Windows-friendly data download path and raw-data safety boundary.
- Modify `docs/PROJECT_GAP_STATUS.md`: update HotpotQA/NQ availability and real-data reader/Gemini status after runs.
- Possibly create `outputs/results/hotpot_reader_realdata_summary.csv`: small sanitized summary if HotpotQA real-data reader comparison runs.
- Possibly create `outputs/results/hotpot_gemini_pilot_summary.csv`: small sanitized summary if bounded Gemini pilot runs.
- Do not commit raw files under `data/raw/`, API caches under `outputs/cache/`, or local run folders under `outputs/codex_*`.

## Task 1: Baseline And Plan Check

- [ ] **Step 1: Verify branch and clean status**

Run:

```powershell
git status --short --branch
git log --oneline -5
```

Expected: on `codex/api-reader-fqi-upgrade-20260606-0126`, no uncommitted changes except this plan while writing.

- [ ] **Step 2: Run raw-data preflight**

Run:

```powershell
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight_realdata_plan
```

Expected: SciFact/NFCorpus available, HotpotQA/NQ may be missing.

## Task 2: Cross-Platform Missing Data Downloader

- [ ] **Step 1: Write failing downloader tests**

Create tests for:

- dry-run reports missing HotpotQA target without network;
- existing file is skipped unless overwrite is requested;
- mocked fetch writes bytes to a temporary path and reports `downloaded`;
- unknown dataset key raises a clear error.

Run:

```powershell
uv run pytest tests/test_raw_data_download.py -q
```

Expected before implementation: fail because module/script does not exist.

- [ ] **Step 2: Implement downloader**

Implement `download_missing_raw_data(project_root, dataset_keys, output_dir, dry_run, overwrite, fetcher)` with dataset keys:

- `hotpot-dev-distractor` -> `data/raw/HotpotQA/hotpot_dev_distractor_v1.json`
- `hotpot-dev-fullwiki` -> `data/raw/HotpotQA/hotpot_dev_fullwiki_v1.json`
- `hotpot-train` -> `data/raw/HotpotQA/hotpot_train_v1.1.json`
- `nq-validation-shard` -> `data/raw/natural-questions/default/validation-00000-of-00007.parquet`

The NQ URL may be documented or marked unsupported if direct download cannot be safely resolved. The implementation must never print raw file contents.

- [ ] **Step 3: Add CLI**

Add `scripts/download_missing_raw_data.py` with:

```powershell
uv run python scripts/download_missing_raw_data.py --dataset hotpot-dev-distractor --dry-run --output-dir outputs/codex_data_download_preflight
```

and live download:

```powershell
uv run python scripts/download_missing_raw_data.py --dataset hotpot-dev-distractor --output-dir outputs/codex_data_download_hotpot
```

- [ ] **Step 4: Verify tests**

Run:

```powershell
uv run pytest tests/test_raw_data_download.py -q
```

Expected: pass.

## Task 3: Download HotpotQA And Run Real-Data Reader Comparison

- [ ] **Step 1: Dry-run download**

Run:

```powershell
uv run python scripts/download_missing_raw_data.py --dataset hotpot-dev-distractor --dry-run --output-dir outputs/codex_data_download_preflight
```

Expected: reports whether the file is missing or already available.

- [ ] **Step 2: Live download if missing**

Run only if missing:

```powershell
uv run python scripts/download_missing_raw_data.py --dataset hotpot-dev-distractor --output-dir outputs/codex_data_download_hotpot
```

Expected: raw JSON appears under ignored `data/raw/HotpotQA/`.

- [ ] **Step 3: Run data preflight again**

Run:

```powershell
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight_after_hotpot
```

Expected: HotpotQA dev distractor available; NQ may remain missing.

- [ ] **Step 4: Run HotpotQA reader comparison**

Run:

```powershell
uv run python scripts/run_reader_comparison.py --dataset hotpot --num-examples 50 --readers lexical,span --output-dir outputs/codex_reader_hotpot_realdata_50
```

Expected: summary CSV/JSON with lexical and span EM/F1. This remains tiny real-data evidence, not final QA benchmark evidence.

## Task 4: Bounded Gemini Baseline Pilot

- [ ] **Step 1: Gemini dry-run**

Run:

```powershell
uv run python scripts/run_gemini_baseline.py --data-path data/raw/HotpotQA/hotpot_dev_distractor_v1.json --num-examples 10 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_realdata.jsonl --dry-run --max-new-calls 8 --output-dir outputs/codex_gemini_realdata_dry
```

Expected: estimates cache misses and required calls without API.

- [ ] **Step 2: Live bounded pilot if misses <= 8**

Run:

```powershell
$env:CODEX_ALLOW_API_CALLS='1'; uv run python scripts/run_gemini_baseline.py --data-path data/raw/HotpotQA/hotpot_dev_distractor_v1.json --num-examples 10 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_realdata.jsonl --allow-api --max-new-calls 8 --output-dir outputs/codex_gemini_realdata_pilot; Remove-Item Env:\CODEX_ALLOW_API_CALLS
```

Expected: at most 8 new Gemini calls, cached results under ignored `outputs/cache/`, summary under ignored local output. If misses exceed budget, do not run live.

## Task 5: Sanitized Evidence And Docs

- [ ] **Step 1: Create sanitized summaries if real-data runs completed**

Copy only aggregate summary rows into `outputs/results/` with no raw prompts/passages:

- `outputs/results/hotpot_reader_realdata_summary.csv`
- `outputs/results/hotpot_gemini_pilot_summary.csv`

- [ ] **Step 2: Update docs**

Update:

- `docs/PROJECT_GAP_STATUS.md`
- `docs/API_EXPERIMENTS.md`
- `docs/READER_EXTENSION.md`
- `docs/FINAL_REPRODUCTION.md`

Keep claim boundary conservative.

- [ ] **Step 3: Update dashboard/index if sanitized summaries are committed**

Run:

```powershell
uv run python scripts/run_artifact_index.py --output-csv outputs/results/final_artifact_index.csv
uv run python scripts/run_experiment_dashboard.py --output-csv outputs/results/experiment_dashboard.csv --output-md docs/EXPERIMENT_DASHBOARD.md
```

Expected: reader real-data and Gemini pilot appear as `tiny_realdata` / `api_pilot`, not `final_claim`.

## Task 6: Final Verification, Commit, Push

- [ ] **Step 1: Run tests**

Run:

```powershell
uv run pytest -q
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_realdata_reader_gemini --pytest-mode targeted
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Safety check staged files**

Run:

```powershell
git diff --cached --name-only
```

Expected: no `.env`, no credentials, no `data/raw`, no `outputs/cache`, no `outputs/codex_*`.

- [ ] **Step 3: Commit and push**

Run:

```powershell
git add <intentional files only>
git commit -m "data: add bounded hotpot reader and gemini pilot path"
git push
```

Expected: pushed branch remains reviewable.

## Claim Boundary

Defensible after this plan if successful:

- HotpotQA raw-data reader comparison has been run as tiny real-data evidence.
- Gemini baseline path has been live-tested under a strict call budget.
- API and reader evidence remains pilot/smoke unless repeated, larger, baseline-controlled runs support a stronger statement.

Not defensible:

- Full downstream RAG answer-quality improvement.
- Semantic/Gemini superiority as final evidence.
- Production-ready online routing.
