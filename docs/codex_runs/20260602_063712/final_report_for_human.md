# Final Report For Human

## Executive Summary

I continued from the cleaned `main` merge on branch `codex/overnight-improvements-20260602-0636`. I did not make live API calls. I validated that `.env` and Google credential file presence can be detected safely, confirmed SciFact and NFCorpus raw data are locally available, ran tiny real-data full-corpus smoke experiments for both datasets, ran guardrail/cost diagnostics on existing final artifacts, and added reviewer-facing documentation that clearly lists what remains.

## Branch And Base

- Branch: `codex/overnight-improvements-20260602-0636`
- Base: `f316e43 merge: add essential reproducibility and api safety updates`

## Changes Made

- Added `docs/CODEX_REVIEW_SUMMARY.md` with current evidence, claim boundaries, and remaining work.
- Added `src/selective_rag_rl/experiment_dashboard.py` and `scripts/run_experiment_dashboard.py` to produce a machine-readable evidence dashboard.
- Added `tests/test_experiment_dashboard.py`.
- Fixed semantic rank-agreement correlation so constant semantic scores no longer emit pandas/scipy `ConstantInputWarning`.
- Added `docs/EXPERIMENT_DASHBOARD.md`, generated from existing outputs and local smoke artifacts.
- Added `docs/codex_runs/20260602_063712/` run log files.
- Added `docs/superpowers/plans/2026-06-02-overnight-rag-rl-hardening.md`.
- Added `docs/superpowers/plans/2026-06-02-code-hardening-and-evidence-dashboard.md`.
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
| `uv run pytest -q` | 182 passed |
| `uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_overnight --pytest-mode targeted` | Nested pytest 18 passed; smoke status pass |
| `uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_code_hardening --pytest-mode targeted` | Nested pytest 18 passed; smoke status pass |
| `uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_overnight` | Pass; EM 0.0, F1 0.490079 |
| `uv run pytest tests/test_experiment_dashboard.py -q` | 5 passed |
| `uv run pytest tests/test_retrieval_policy_experiment.py::test_semantic_rank_agreement_handles_constant_scores_without_warning tests/test_retrieval_policy_experiment.py::test_semantic_state_features_can_use_deeper_rank_profile -q` | 2 passed |
| `uv run python scripts/run_experiment_dashboard.py ...` | 200 artifacts classified; 6 claim-allowed artifacts |

The previous semantic-feature constant-input warning has a regression test and is fixed in code. Full pytest now runs without that warning.

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
- Experiment dashboard ran across `outputs/results`, overnight smoke/preflight directories, and reader smoke outputs. It classified 200 artifacts and allowed final/full benchmark claims only for final-claim index artifacts and SciFact/NFCorpus retrieval policy summaries.

## Claims Changed

No final claims were changed. The conservative retrieval-stage claim remains the supported one. No answer-generation, semantic-feature superiority, online RL, or Gemini-improvement claim was added.

## Known Limitations

- No live Gemini or Vertex call was made in this run.
- HotpotQA and NQ raw files are not present locally.
- Tiny real-data smoke is too small for final result claims.
- The validation guardrail can audit existing final artifacts but cannot perform true selection without validation-split inputs.
- The dashboard uses conservative filename/path and metadata heuristics; a human should review labels before using it in slides or final text.

## Manual Review Checklist

1. Review `docs/CODEX_REVIEW_SUMMARY.md`.
2. Review `.gitignore` wildcard additions.
3. Decide whether run logs under `docs/codex_runs/20260602_063712/` should be merged or kept branch-local.
4. Decide whether to set `CODEX_ALLOW_API_CALLS=1` for a one-call API smoke later.
5. If final report updates are desired, only add a short appendix that points to reproducibility/API docs unless new full-scale evidence is generated.

## Suggested Next Prompt

Continue from branch `codex/overnight-improvements-20260602-0636`. Review the overnight diff, keep only reviewer-useful docs and `.gitignore` changes, optionally run a one-call API smoke if `CODEX_ALLOW_API_CALLS=1` is explicitly set, then decide whether to merge into `main`.
