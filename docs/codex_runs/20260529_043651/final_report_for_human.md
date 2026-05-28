# Final Report For Human

## Executive Summary

This autonomous run focused on reproducibility, conservative claim boundaries,
and lightweight downstream-evaluation plumbing. The default test suite now runs
without raw HotpotQA data or platform-specific newline assumptions. Reviewers
can run a raw-data-free final smoke command that exercises targeted tests,
retrieval-policy evaluation, OPE diagnostics, and manifest generation. The repo
also now includes deterministic SQuAD-style EM/F1 metrics and a lexical-overlap
reader smoke path, documented as plumbing rather than final QA evidence.

No full-data benchmark claims were changed. Raw SciFact/NFCorpus/Hotpot/NQ data
was absent locally, and no Vertex/Gemini or model-download paths were executed.

## Commits Made

- `afa49f4` - `tests: make baseline suite data independent`
- `6ef6908` - `scripts: add raw-data-free final smoke runner`
- `bec4f96` - `eval: add lightweight reader EM F1 smoke`
- `93bf3af` - `docs: clarify offline bandit framing`
- `304b84b` - `docs: add validation and cost model guides`
- Final report/status commit follows this file update.

## Pushes Attempted

All milestone commits above were pushed successfully to:

```text
origin/codex/autonomous-improvements-20260529-0436
```

No push failed during this run.

## Tests And Smoke Commands Run

- `uv sync` passed.
- Initial `uv run pytest -q` failed on two environment-dependent tests: platform
  newline byte-size expectation and missing raw HotpotQA data.
- Targeted fixes passed:
  - `uv run pytest tests/test_artifact_index.py::test_export_artifact_index_records_existing_and_missing_files -q`
  - `uv run pytest tests/test_core.py::test_hotpot_loader_reads_examples -q`
- Full baseline after fixes: `uv run pytest -q` -> `142 passed, 1 warning`.
- P0 smoke:
  - `uv run pytest tests/test_final_smoke.py -q` passed.
  - `uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke --pytest-mode skip` passed.
  - `uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke` passed with nested targeted pytest `17 passed`.
  - Full suite after P0: `143 passed, 1 warning`.
- P1 reader:
  - `uv run pytest tests/test_answer_metrics.py tests/test_reader.py tests/test_reader_eval.py -q` passed.
  - `uv run python scripts/run_reader_eval.py --help` passed.
  - `uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke` passed.
  - Full suite after P1: `149 passed, 1 warning`.
- P3 RL framing:
  - Focused bandit/OPE tests passed.
  - `uv run pytest tests/test_off_policy_evaluation.py tests/test_bandit_baselines.py -q` -> `11 passed`.
  - Full suite after P3: `151 passed, 1 warning`.
- P2/P5 docs:
  - Final full suite: `uv run pytest -q` -> `151 passed, 1 warning`.
- Final report/status handoff:
  - `uv run pytest -q` -> `151 passed, 1 warning`.

The single warning is an existing pandas `ConstantInputWarning` in
`test_semantic_state_features_can_use_deeper_rank_profile`.

## New Files Added

- `docs/COST_MODEL.md`
- `docs/FINAL_REPRODUCTION.md`
- `docs/READER_EXTENSION.md`
- `docs/RL_FRAMING.md`
- `docs/VALIDATION_PROTOCOL.md`
- `docs/codex_runs/20260529_043651/00_initial_audit.md`
- `docs/codex_runs/20260529_043651/commands.md`
- `docs/codex_runs/20260529_043651/p0_smoke_reproduction_plan.md`
- `docs/codex_runs/20260529_043651/p1_reader_extension_plan.md`
- `docs/codex_runs/20260529_043651/p2_p5_validation_cost_plan.md`
- `docs/codex_runs/20260529_043651/status.md`
- `docs/codex_runs/20260529_043651/taskboard.md`
- `docs/codex_runs/latest_status.md`
- `scripts/run_final_smoke.py`
- `scripts/run_reader_eval.py`
- `src/selective_rag_rl/answer_metrics.py`
- `src/selective_rag_rl/reader.py`
- `tests/test_answer_metrics.py`
- `tests/test_final_smoke.py`
- `tests/test_reader.py`
- `tests/test_reader_eval.py`

## Existing Files Modified

- `.gitignore`
- `README.md`
- `tests/test_artifact_index.py`
- `tests/test_bandit_baselines.py`
- `tests/test_core.py`
- `tests/test_off_policy_evaluation.py`

## Evidence Artifacts Generated

- `outputs/codex_smoke/smoke_manifest.json`
- `outputs/codex_smoke/ope_diagnostics.csv`
- `outputs/codex_smoke/synthetic_hotpot.json`
- `outputs/codex_smoke/retrieval_policy/` smoke outputs
- `outputs/codex_reader_smoke/reader_eval_toy_detailed.csv`
- `outputs/codex_reader_smoke/reader_eval_toy_summary.csv`
- `outputs/codex_reader_smoke/reader_eval_toy_summary.json`

These generated outputs are ignored by git and are intended as local evidence,
not committed benchmark artifacts.

## Known Limitations

- Raw datasets were absent under `data/raw/`, so full SciFact, NFCorpus,
  HotpotQA, and NQ reproduction commands were not run.
- No Vertex/Gemini API calls were made.
- No large model downloads were made.
- The lexical reader is deterministic and lightweight; it is not a neural
  reader and does not support final answer-generation claims.
- The smoke runner uses synthetic data and fake embeddings, so it validates code
  paths rather than benchmark performance.
- Validation guardrail and cost-model changes are documentation-only in this
  run; deeper guardrail utilities/tests remain future work.

## Recommended Manual Review Checklist

1. Review `docs/FINAL_REPRODUCTION.md` and run the final smoke command on a
   clean clone.
2. Inspect `scripts/run_final_smoke.py` to confirm it avoids raw data, external
   APIs, and model downloads.
3. Inspect `scripts/run_reader_eval.py`, `answer_metrics.py`, and `reader.py` to
   confirm the reader path is appropriately labeled as smoke-only.
4. Review `docs/RL_FRAMING.md` for defense wording and ensure it matches the
   intended course-project framing.
5. Review `docs/VALIDATION_PROTOCOL.md` and `docs/COST_MODEL.md` for claim
   conservatism.
6. Run full-data reproduction only after populating `data/raw/` according to the
   README and `docs/FINAL_REPRODUCTION.md`.
7. Confirm generated `outputs/codex_smoke/` and `outputs/codex_reader_smoke/`
   are not staged.

## Suggested Next Autonomous Run Prompt

Continue from branch `codex/autonomous-improvements-20260529-0436`. First run
`git status --short`, `git log --oneline -8`, and `uv run pytest -q`. Then
prioritize adding a tested validation-guardrail utility that consumes repeated
selection/grid CSVs and emits the documented columns from
`docs/VALIDATION_PROTOCOL.md`. If time remains, add a small cost-frontier
summary script over existing result CSVs. Keep full benchmark claims unchanged
unless new full-data evidence is generated.
