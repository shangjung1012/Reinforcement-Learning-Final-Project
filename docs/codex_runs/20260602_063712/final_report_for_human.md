# Final Report For Human

## Executive Summary

I continued from the cleaned `main` merge on branch `codex/overnight-improvements-20260602-0636`. I did not make live API calls. I validated that `.env` and Google credential file presence can be detected safely, confirmed SciFact and NFCorpus raw data are locally available, ran tiny real-data full-corpus smoke experiments for both datasets, ran guardrail/cost diagnostics on existing final artifacts, and added reviewer-facing documentation that clearly lists what remains.

## Branch And Base

- Branch: `codex/overnight-improvements-20260602-0636`
- Base: `f316e43 merge: add essential reproducibility and api safety updates`

## Changes Made

- Added `docs/CODEX_REVIEW_SUMMARY.md` with current evidence, claim boundaries, and remaining work.
- Added `docs/codex_runs/20260602_063712/` run log files.
- Added `docs/superpowers/plans/2026-06-02-overnight-rag-rl-hardening.md`.
- Updated `.gitignore` to ignore wildcard generated evidence directories:
  - `outputs/codex_api_preflight_*/`
  - `outputs/codex_data_preflight_*/`
  - `outputs/codex_realdata_smoke_*/`
  - `outputs/codex_reader_smoke_*/`
  - `outputs/codex_reader_realdata_smoke_*/`
  - `outputs/codex_gemini_*/`
  - `outputs/codex_vertex_*/`
  - `outputs/codex_diagnostics_*/`

## Tests And Smoke Commands

| Command | Result |
| --- | --- |
| `uv run pytest -q` | 176 passed, 1 warning |
| `uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_overnight --pytest-mode targeted` | Nested pytest 18 passed; smoke status pass |
| `uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_overnight` | Pass; EM 0.0, F1 0.490079 |

The one warning is the existing pandas `ConstantInputWarning` in the semantic feature correlation test.

## API Preflight

| Provider | Mode | Actual calls/texts | Result |
| --- | --- | ---: | --- |
| Gemini | dry-run preflight | 0 | `dry_run_no_api_call` |
| Vertex embedding | dry-run preflight | 0 | `dry_run_no_api_call` |

`.env` exists and required variable names are present. The credential file exists. The run did not print secret values and did not make live API calls because `CODEX_ALLOW_API_CALLS` is not set.

## Data Availability

| Dataset | Status |
| --- | --- |
| SciFact | available |
| NFCorpus | available |
| HotpotQA | blocked_missing_data |
| Natural Questions | blocked_missing_data |

## Real-Data Runs

| Dataset | Command Scope | Evidence Level | Result |
| --- | --- | --- | --- |
| SciFact | 30 train / 30 test, full corpus, fake embedder | `tiny_realdata` | Completed |
| NFCorpus | 30 train / 30 test, full corpus, fake embedder | `tiny_realdata` | Completed |

These runs verify local data and script integration only. They do not change final benchmark claims.

## Diagnostics

- Validation guardrail ran on existing SciFact/NFCorpus final detailed CSVs.
- Both guardrail outputs are `analysis_only_no_validation` because the detailed final artifacts do not include a validation split.
- Cost frontier ran on existing SciFact/NFCorpus final summary CSVs and produced ignored local diagnostic outputs.

## Claims Changed

No final claims were changed. The conservative retrieval-stage claim remains the supported one. No answer-generation, semantic-feature superiority, online RL, or Gemini-improvement claim was added.

## Known Limitations

- No live Gemini or Vertex call was made in this run.
- HotpotQA and NQ raw files are not present locally.
- Tiny real-data smoke is too small for final result claims.
- The validation guardrail can audit existing final artifacts but cannot perform true selection without validation-split inputs.

## Manual Review Checklist

1. Review `docs/CODEX_REVIEW_SUMMARY.md`.
2. Review `.gitignore` wildcard additions.
3. Decide whether run logs under `docs/codex_runs/20260602_063712/` should be merged or kept branch-local.
4. Decide whether to set `CODEX_ALLOW_API_CALLS=1` for a one-call API smoke later.
5. If final report updates are desired, only add a short appendix that points to reproducibility/API docs unless new full-scale evidence is generated.

## Suggested Next Prompt

Continue from branch `codex/overnight-improvements-20260602-0636`. Review the overnight diff, keep only reviewer-useful docs and `.gitignore` changes, optionally run a one-call API smoke if `CODEX_ALLOW_API_CALLS=1` is explicitly set, then decide whether to merge into `main`.
