# Overnight RAG RL Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue from the cleaned `main` merge by validating the local data/API situation and adding only high-value, tested improvements that strengthen reproducibility and final-project defensibility.

**Architecture:** Keep external API use optional and cache-first. Prefer machine-readable preflight/evidence artifacts and small deterministic tests over broad final-claim edits. Work on an isolated feature branch, commit coherent milestones, and leave `main` unchanged until the final review.

**Tech Stack:** Python 3.11+, `uv`, pytest, pandas/numpy, pathlib/argparse scripts, local `.env` loaded only through existing safe preflight code.

---

## Scope And Current State

The cleaned main branch already includes:

- raw-data-free full pytest and final smoke runner;
- reader EM/F1 smoke path;
- API and data preflight scripts;
- Gemini and Vertex budget gates;
- validation guardrail and cost frontier utilities;
- conservative documentation for RL framing, cost model, validation, API use, and reproduction.

Remaining work should focus on evidence and integration, not rewriting the core pipeline.

## What Still Remains After This Run

These are expected to remain for human/manual follow-up unless tonight's evidence proves otherwise:

- Decide whether to spend more Gemini/Vertex quota beyond a one-call or cache-only preflight.
- Run full-scale SciFact/NFCorpus repeated-seed benchmark if final report evidence must be regenerated.
- Run a real downstream answer-generation benchmark before claiming RAG answer-quality gains.
- Review and possibly update `FINAL_REPORT.md` only if new full-scale evidence exists.
- Decide whether to open a PR or merge this overnight branch after reviewing the final diff.

## Assistance Needed From Human

No immediate help is needed for safe local work. Human input is only needed if:

- live API calls fail due Google Cloud permissions, quota, or model access;
- more than a tiny bounded API smoke is desired;
- final claims should be changed in `FINAL_REPORT.md`;
- generated local outputs should be preserved as submitted artifacts.

## File Structure

- Create/update `docs/codex_runs/20260602_063712/*`: run audit, taskboard, command log, API/data reports, final handoff.
- Possibly update `docs/API_EXPERIMENTS.md`, `docs/FINAL_REPRODUCTION.md`, `docs/VALIDATION_PROTOCOL.md`, `docs/COST_MODEL.md`: only if command behavior or interpretation needs clarification.
- Possibly add or update `docs/CODEX_REVIEW_SUMMARY.md`: high-level second/third run summary suitable for a reviewer.
- Prefer adding tests under `tests/` only when a real gap is found.
- Do not commit `.env`, credentials, raw data, cache files, or generated `outputs/codex_*` directories.

---

### Task 1: Re-Audit Current Main And Record Run State

**Files:**
- Create: `docs/codex_runs/20260602_063712/00_initial_audit.md`
- Create: `docs/codex_runs/20260602_063712/taskboard.md`
- Create: `docs/codex_runs/20260602_063712/status.md`
- Create: `docs/codex_runs/20260602_063712/commands.md`
- Create: `docs/codex_runs/20260602_063712/api_preflight.md`
- Modify: `docs/codex_runs/latest_status.md`

- [ ] **Step 1: Capture git and baseline state**

Run:

```powershell
git status --short --branch --ignored
git branch --show-current
git log --oneline -8 --decorate
uv run pytest -q
```

Expected: branch is `codex/overnight-improvements-*`, tracked tree is clean, pytest passes.

- [ ] **Step 2: Record audit**

Write `00_initial_audit.md` with:

- branch and base commit;
- ignored local artifacts summary;
- test baseline;
- docs/scripts/tests already present;
- raw data and `.env` handling policy;
- prioritized backlog.

- [ ] **Step 3: Record taskboard and status**

Write `taskboard.md` with tonight's ordered backlog and `status.md` with command results.

---

### Task 2: Run No-Cost API And Data Preflights

**Files:**
- Update: `docs/codex_runs/20260602_063712/api_preflight.md`
- Add optional generated ignored artifacts under `outputs/codex_api_preflight_overnight/`
- Add optional generated ignored artifacts under `outputs/codex_data_preflight_overnight/`

- [ ] **Step 1: Run API dry-run preflight**

Run:

```powershell
uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight_overnight
```

Expected: no external calls, only variable presence/credential basename in summary.

- [ ] **Step 2: Inspect summary without printing secrets**

Run:

```powershell
Get-Content outputs/codex_api_preflight_overnight/api_preflight_summary.json
```

Expected: no secret values; result identifies present/missing variables and whether live calls are blocked.

- [ ] **Step 3: Run data preflight**

Run:

```powershell
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight_overnight
```

Expected: file availability and row/file counts only; no raw content dump.

- [ ] **Step 4: Decide API posture**

If `CODEX_ALLOW_API_CALLS` is already set to `1`, optionally run the one-call API smoke with strict budgets. If it is not set, do not make live calls and record that human approval/quota decision remains.

---

### Task 3: Run Real-Data Smoke Where Local Raw Data Exists

**Files:**
- Create: `docs/codex_runs/20260602_063712/realdata_smoke_report.md`
- Generated ignored artifacts under `outputs/codex_realdata_smoke_overnight/`

- [ ] **Step 1: Run SciFact tiny full-corpus fake-embedder smoke if data exists**

Run:

```powershell
uv run python scripts/run_retrieval_policy_scifact.py --num-train-examples 30 --num-test-examples 30 --seed 42 --full-corpus --embedder fake --policy-model ridge --tuning-folds 2 --knn-k-candidates 1 --output-dir outputs/codex_realdata_smoke_overnight/scifact_fake
```

Expected: command exits 0 and writes summary/detailed CSVs.

- [ ] **Step 2: Run NFCorpus tiny full-corpus fake-embedder smoke if data exists**

Run:

```powershell
uv run python scripts/run_retrieval_policy_nfcorpus.py --num-train-examples 30 --num-test-examples 30 --seed 42 --full-corpus --embedder fake --policy-model ridge --tuning-folds 2 --knn-k-candidates 1 --output-dir outputs/codex_realdata_smoke_overnight/nfcorpus_fake
```

Expected: command exits 0 and writes summary/detailed CSVs.

- [ ] **Step 3: Run reader real-data smoke only if Hotpot/NQ data exists**

Run Hotpot/NQ reader commands only when preflight says files exist:

```powershell
uv run python scripts/run_reader_eval.py --dataset hotpot --num-examples 20 --output-dir outputs/codex_reader_realdata_smoke_overnight/hotpot_lexical
uv run python scripts/run_reader_eval.py --dataset nq --num-examples 20 --output-dir outputs/codex_reader_realdata_smoke_overnight/nq_lexical
```

Expected: if data is absent, skip and record missing-data status.

- [ ] **Step 4: Summarize smoke results**

Write `realdata_smoke_report.md` with dataset, command, evidence level `tiny_realdata`, and explicit claim boundary.

---

### Task 4: Exercise Guardrail And Cost Frontier On Available Artifacts

**Files:**
- Generated ignored artifacts under `outputs/results/*_overnight*.csv` if outputs/results is intentionally used.
- Update: `docs/codex_runs/20260602_063712/status.md`

- [ ] **Step 1: Run validation guardrail on existing final detailed CSVs if present**

Run:

```powershell
uv run python scripts/run_validation_guardrail.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_validation_guardrail_overnight.csv
uv run python scripts/run_validation_guardrail.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_validation_guardrail_overnight.csv
```

Expected: CSVs identify `analysis_only_no_validation` if no validation split exists.

- [ ] **Step 2: Run cost frontier on existing final summary CSVs if present**

Run:

```powershell
uv run python scripts/run_cost_frontier_summary.py --dataset scifact --summary-csv outputs/results/scifact_retrieval_policy_summary.csv --output-csv outputs/results/scifact_cost_frontier_overnight.csv --budgets 1.0,1.25,1.5,2.0
uv run python scripts/run_cost_frontier_summary.py --dataset nfcorpus --summary-csv outputs/results/nfcorpus_retrieval_policy_summary.csv --output-csv outputs/results/nfcorpus_cost_frontier_overnight.csv --budgets 1.0,1.25,1.5,2.0
```

Expected: no final claims are changed; outputs are diagnostics.

- [ ] **Step 3: Inspect generated CSV heads**

Run:

```powershell
Import-Csv outputs/results/scifact_validation_guardrail_overnight.csv | Select-Object -First 3 | Format-Table
Import-Csv outputs/results/scifact_cost_frontier_overnight.csv | Select-Object -First 3 | Format-Table
```

Expected: columns are interpretable and contain no raw private data.

---

### Task 5: Fill One Documentation/Test Gap Found During Audit

**Files:** To be determined by the audit, but likely one of:
- Modify: `docs/API_EXPERIMENTS.md`
- Modify: `docs/FINAL_REPRODUCTION.md`
- Modify: `docs/CODEX_REVIEW_SUMMARY.md`
- Test: existing targeted test under `tests/`

- [ ] **Step 1: Identify one gap**

Choose the highest-value gap from real evidence. Acceptable examples:

- API preflight output needs clearer interpretation;
- data preflight needs docs in reproduction guide;
- a script has a missing `--help` smoke test;
- docs do not clearly say API pilot evidence is not benchmark evidence.

- [ ] **Step 2: Write failing test first if code behavior changes**

For code changes, add or modify a pytest that fails before implementation.

- [ ] **Step 3: Implement minimal fix**

Use the existing module/script style. Do not add dependencies.

- [ ] **Step 4: Validate targeted and full tests**

Run the targeted test and:

```powershell
uv run pytest -q
```

Expected: full test suite passes.

---

### Task 6: Final Verification, Commit, Push, And Human Handoff

**Files:**
- Create: `docs/codex_runs/20260602_063712/final_report_for_human.md`
- Modify: `docs/codex_runs/latest_status.md`
- Possibly commit selected source/docs/test changes only

- [ ] **Step 1: Check tracked vs ignored state**

Run:

```powershell
git status --short --branch --ignored
git diff --stat
git diff --check
git ls-files .env data/raw outputs/cache outputs/codex_api_preflight_overnight outputs/codex_realdata_smoke_overnight
```

Expected: no secrets/raw data/cache tracked; only intentional docs/source/tests are staged.

- [ ] **Step 2: Run final tests and smoke**

Run:

```powershell
uv run pytest -q
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_overnight --pytest-mode targeted
```

Expected: all pass.

- [ ] **Step 3: Write final handoff**

Write `final_report_for_human.md` with:

- branch and commits;
- tests and smoke results;
- API/data preflight status;
- real-data smoke status;
- new/modified files;
- claims changed or not changed;
- exact remaining work.

- [ ] **Step 4: Commit and push if coherent**

Run:

```powershell
git add <intentional files only>
git commit -m "<area>: <imperative summary>"
git push -u origin codex/overnight-improvements-20260602-0636
```

Expected: push succeeds or failure is documented.
