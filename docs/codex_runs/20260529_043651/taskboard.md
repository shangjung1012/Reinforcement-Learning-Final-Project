# Taskboard

## Current Milestone

- [x] Create autonomous run branch.
- [x] Record initial repository audit.
- [x] Run baseline dependency sync and tests.
- [x] Fix baseline tests that depended on platform newlines or absent raw data.
- [x] Implement P0 smoke reproduction runner and docs.
- [x] Validate P0 with full pytest and commit.
- [x] Implement P1 answer metrics and lightweight reader smoke.
- [x] Validate P1 with full pytest and commit.
- [x] Add P3 RL framing docs and focused bandit/OPE sanity tests.
- [x] Add P2/P5 validation and cost interpretation docs.

## Backlog

- [ ] P0: Add `docs/FINAL_REPRODUCTION.md`.
- [x] P0: Add `docs/FINAL_REPRODUCTION.md`.
- [x] P0: Add `scripts/run_final_smoke.py` that avoids raw data and external APIs.
- [x] P0: Update `.gitignore` for local venv/cache/raw-data safety.
- [x] P0: Document smoke vs full-data reproduction in README.
- [x] P1: Add `answer_metrics.py` with normalized EM/F1.
- [x] P1: Add deterministic lightweight reader and reader eval script.
- [x] P1: Add tests for answer metrics and reader behavior.
- [x] P3: Add `docs/RL_FRAMING.md`.
- [x] P3: Strengthen OPE or selected-action bandit sanity tests if gaps remain.
- [x] P2: Add `docs/VALIDATION_PROTOCOL.md` with selection guardrails and reporting rules.
- [x] P5: Add `docs/COST_MODEL.md`.
- [ ] P6: Write final human report and update latest status.

## Notes

- Raw datasets are currently absent under `data/raw/`.
- Do not run Vertex/Gemini calls unless explicitly safe and cached.
- Keep final claims conservative unless new full evidence is generated.
- Full pytest baseline now passes locally: `142 passed, 1 warning`.
- Smoke runner direct validation passes with targeted pytest: `17 passed`.
- Reader toy smoke writes EM/F1 outputs; it is documented as evaluation plumbing, not final QA evidence.
- RL framing tests and full pytest pass after P3: `151 passed, 1 warning`.
