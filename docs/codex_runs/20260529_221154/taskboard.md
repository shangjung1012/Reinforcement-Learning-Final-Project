# Taskboard

## Current Milestone

- [x] Create second-run branch.
- [x] Run baseline pytest.
- [x] Run second-run raw-data-free final smoke.
- [x] Run second-run reader toy smoke.
- [x] Create second-run run-log files.
- [ ] Commit milestone 0 audit and bootstrap updates.

## Backlog

- [ ] P0: Add `scripts/run_api_preflight.py`, `src/selective_rag_rl/api_preflight.py`, and tests.
- [ ] P0: Run no-cost API preflight.
- [ ] P0: Optionally run one-call Gemini and one-text embedding preflight with explicit allow flag.
- [ ] P3: Add `scripts/run_data_preflight.py`, `src/selective_rag_rl/data_preflight.py`, and tests.
- [ ] P3: Run data preflight and document missing raw datasets.
- [ ] P1: Add executable validation guardrail utility and tests.
- [ ] P2: Add cost frontier summary utility and tests.
- [ ] P4/P5: Add stronger API budget enforcement before any non-preflight pilot.
- [ ] P7/P8: Add dashboard/final second-run report if time remains.

## Notes

- `.env` exists locally and is ignored by git.
- Raw datasets are currently absent.
- Baseline tests passed after branch creation: `151 passed, 1 warning`.
- Second-run smoke commands passed and generated ignored local artifacts.
