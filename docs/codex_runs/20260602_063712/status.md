# Codex Overnight Run Status

## Current branch

`codex/overnight-improvements-20260602-0636`

## Base

`main` / `origin/main` at `f316e43 merge: add essential reproducibility and api safety updates`

## Milestones

- [x] Created continuation branch from cleaned main.
- [x] Wrote overnight implementation plan.
- [x] Completed no-cost API/data preflights.
- [x] Completed tiny real-data smoke where data exists.
- [x] Completed diagnostics on existing artifacts.
- [x] Extended `.gitignore` for overnight/generated evidence directories.
- [x] Completed toy reader EM/F1 smoke.
- [x] Completed final verification.

## Command Log

| Time | Command | Result | Notes |
| --- | --- | --- | --- |
| 2026-06-02 06:36 | `uv run pytest -q` | Pass | 176 passed, 1 warning. |
| 2026-06-02 06:40 | `uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight_overnight` | Pass | Dry-run only; 0 API calls/texts. |
| 2026-06-02 06:40 | `uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight_overnight` | Pass | SciFact/NFCorpus available; HotpotQA/NQ missing. |
| 2026-06-02 06:42 | `uv run python scripts/run_retrieval_policy_scifact.py ... --output-dir outputs/codex_realdata_smoke_overnight/scifact_fake` | Pass | Tiny real-data full-corpus smoke, fake embedder, 30 train / 30 test. |
| 2026-06-02 06:42 | `uv run python scripts/run_retrieval_policy_nfcorpus.py ... --output-dir outputs/codex_realdata_smoke_overnight/nfcorpus_fake` | Pass | Tiny real-data full-corpus smoke, fake embedder, 30 train / 30 test. |
| 2026-06-02 06:44 | `uv run python scripts/run_validation_guardrail.py ...` | Pass | SciFact/NFCorpus final detailed artifacts audited as analysis-only due no validation split. |
| 2026-06-02 06:44 | `uv run python scripts/run_cost_frontier_summary.py ...` | Pass | SciFact/NFCorpus cost frontier diagnostics generated under ignored outputs. |
| 2026-06-02 06:45 | `git check-ignore -v outputs/codex_*...` | Pass | Added and verified ignore rules for generated overnight evidence directories. |
| 2026-06-02 06:48 | `uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_overnight` | Pass | EM 0.0, F1 0.490079, retrieval metrics 1.0; smoke only. |
| 2026-06-02 06:50 | `uv run pytest -q` | Pass | 176 passed, 1 warning. |
| 2026-06-02 06:50 | `uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_overnight --pytest-mode targeted` | Pass | Nested pytest 18 passed; no raw data/API/model download. |

## Current Blockers

None for safe local work. Live API calls remain blocked because `CODEX_ALLOW_API_CALLS` is not set.

## Next Work

Commit and push the branch.
