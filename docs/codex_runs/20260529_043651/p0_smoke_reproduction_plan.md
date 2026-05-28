# P0 Smoke Reproduction Plan

## Goal

Add a one-command smoke reproduction path that exercises core retrieval-policy
and OPE code without raw datasets, external APIs, or model downloads.

## Design

- Add `scripts/run_final_smoke.py`.
- The script writes a small synthetic HotpotQA-style JSON fixture under the
  requested output directory.
- It loads the fixture through the real `load_hotpotqa` path and runs
  `run_retrieval_policy_experiment` with:
  - `embedder_name="fake"`
  - small example count
  - `policy_model="ridge"`
  - `tuning_folds=2`
- It runs `export_ope_diagnostics` against the generated detailed CSV.
- It writes `smoke_manifest.json` with command settings, generated artifact
  paths, and pass/fail status.
- It supports `--pytest-mode targeted|full|skip`, defaulting to `targeted`.

## Tests First

Add `tests/test_final_smoke.py` before implementation:

- Import `run_final_smoke`.
- Run with `pytest_mode="skip"` and a temporary output directory.
- Assert the synthetic fixture, retrieval summary, OPE CSV, and manifest exist.
- Assert the manifest records no external API use and a successful status.

## Documentation

- Add `docs/FINAL_REPRODUCTION.md`.
- Add a short README pointer for smoke versus full-data reproduction.
- Update `.gitignore` for `.venv/`, `.pytest_cache/`, `__pycache__/`,
  `outputs/cache/`, and raw-data safety.

## Validation

- `uv run pytest tests/test_final_smoke.py -q`
- `uv run pytest -q`
- `uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke --pytest-mode targeted`

## Commit

Commit message target:

`scripts: add raw-data-free final smoke runner`
