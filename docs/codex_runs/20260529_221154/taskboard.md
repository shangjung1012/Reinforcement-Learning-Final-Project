# Taskboard

## Current Milestone

- [x] Create second-run branch.
- [x] Run baseline pytest.
- [x] Run second-run raw-data-free final smoke.
- [x] Run second-run reader toy smoke.
- [x] Create second-run run-log files.
- [x] Commit milestone 0 audit and bootstrap updates.
- [x] Implement P0 API preflight utility and docs.
- [x] Validate and commit P0 API preflight utility.
- [x] Implement P3 data preflight utility.
- [x] Validate and commit P3 data preflight utility.
- [x] Implement P1 validation guardrail utility.
- [x] Validate and commit P1 validation guardrail utility.
- [x] Implement P2 cost frontier utility.
- [x] Validate and commit P2 cost frontier utility.
- [x] Implement Gemini baseline budget gate.
- [ ] Validate and commit Gemini budget gate and pilot report.

## Backlog

- [x] P0: Add `scripts/run_api_preflight.py`, `src/selective_rag_rl/api_preflight.py`, and tests.
- [x] P0: Run no-cost API preflight.
- [x] P0: Run one-call Gemini and one-text embedding preflight with explicit allow flag.
- [x] P3: Add `scripts/run_data_preflight.py`, `src/selective_rag_rl/data_preflight.py`, and tests.
- [x] P3: Run data preflight and document missing raw datasets.
- [x] P1: Add executable validation guardrail utility and tests.
- [x] P2: Add cost frontier summary utility and tests.
- [x] P4: Add stronger Gemini API budget enforcement before non-preflight pilot.
- [ ] P5: Add stronger Vertex embedding budget enforcement if time remains.
- [ ] P7/P8: Add dashboard/final second-run report if time remains.

## Notes

- `.env` exists locally and is ignored by git.
- Raw datasets are currently absent.
- Baseline tests passed after branch creation: `151 passed, 1 warning`.
- Second-run smoke commands passed and generated ignored local artifacts.
- API preflight live smoke used exactly 1 Gemini call and 1 Vertex embedding text.
- Data preflight found all required raw-data paths missing, so real-data smoke is blocked.
- Guardrail outputs for SciFact/NFCorpus are analysis-only because the detailed CSVs do not include validation splits.
- Cost frontier outputs were generated for SciFact/NFCorpus from existing summary CSVs.
- Synthetic Gemini pilot used 4 bounded calls and is cache-resumable; it is API pilot evidence only.
