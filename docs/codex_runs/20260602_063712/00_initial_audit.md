# Initial Audit

## Branch And Commit

- Branch: `codex/overnight-improvements-20260602-0636`
- Base commit: `f316e43 merge: add essential reproducibility and api safety updates`
- This is a continuation run after selectively merging only the useful prior branch changes into `main`.

## Working Tree

The tracked working tree started clean. Local ignored artifacts include `.env`, `.venv`, pytest cache, raw SciFact/NFCorpus data, and generated `outputs/codex_*` directories.

## Baseline Tests

`uv run pytest -q` passed with 176 tests and 1 existing warning from a constant-input semantic feature correlation test.

## Existing Useful Assets

- Smoke reproduction: `scripts/run_final_smoke.py`, `docs/FINAL_REPRODUCTION.md`
- API safety: `scripts/run_api_preflight.py`, `docs/API_EXPERIMENTS.md`
- Data checks: `scripts/run_data_preflight.py`
- Validation/cost diagnostics: `scripts/run_validation_guardrail.py`, `scripts/run_cost_frontier_summary.py`
- Reader smoke: `scripts/run_reader_eval.py`, `src/selective_rag_rl/reader.py`, `src/selective_rag_rl/answer_metrics.py`

## Risks

- `.env` exists locally and must not be printed or committed.
- Raw data exists locally for at least some datasets and must remain ignored.
- API calls may spend quota; this run defaults to no-cost preflight unless explicit allow and budget controls are present.
- Tiny real-data smoke is not final benchmark evidence.

## Priority Backlog

1. Run no-cost API and data preflight.
2. Run tiny real-data smoke for available datasets.
3. Exercise validation guardrail and cost frontier diagnostics on checked-in final artifacts if present.
4. Fill one high-value docs/test gap discovered by evidence.
5. Verify, commit, push, and hand off with exact remaining work.
