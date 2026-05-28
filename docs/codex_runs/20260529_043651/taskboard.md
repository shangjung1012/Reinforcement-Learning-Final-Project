# Taskboard

## Current Milestone

- [x] Create autonomous run branch.
- [x] Record initial repository audit.
- [x] Run baseline dependency sync and tests.
- [x] Fix baseline tests that depended on platform newlines or absent raw data.
- [x] Implement P0 smoke reproduction runner and docs.
- [ ] Validate P0 with full pytest and commit.

## Backlog

- [ ] P0: Add `docs/FINAL_REPRODUCTION.md`.
- [x] P0: Add `docs/FINAL_REPRODUCTION.md`.
- [x] P0: Add `scripts/run_final_smoke.py` that avoids raw data and external APIs.
- [x] P0: Update `.gitignore` for local venv/cache/raw-data safety.
- [x] P0: Document smoke vs full-data reproduction in README.
- [ ] P1: Add `answer_metrics.py` with normalized EM/F1.
- [ ] P1: Add deterministic lightweight reader and reader eval script.
- [ ] P1: Add tests for answer metrics and reader behavior.
- [ ] P3: Add `docs/RL_FRAMING.md`.
- [ ] P3: Strengthen OPE or selected-action bandit sanity tests if gaps remain.
- [ ] P2: Add `docs/VALIDATION_PROTOCOL.md` and guardrail tests if not already covered.
- [ ] P5: Add `docs/COST_MODEL.md`.
- [ ] P6: Write final human report and update latest status.

## Notes

- Raw datasets are currently absent under `data/raw/`.
- Do not run Vertex/Gemini calls unless explicitly safe and cached.
- Keep final claims conservative unless new full evidence is generated.
- Full pytest baseline now passes locally: `142 passed, 1 warning`.
- Smoke runner direct validation passes with targeted pytest: `17 passed`.
