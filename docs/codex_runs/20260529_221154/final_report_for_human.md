# Final Report For Human

## Executive Summary

Second autonomous run completed on branch
`codex/api-validation-improvements-20260529-2211`. The run added safe API and
data preflight utilities, executable validation guardrails, cost frontier
summaries, Gemini API budget gates, a tiny synthetic Gemini pilot, and an
experiment evidence dashboard. All meaningful changes were committed and pushed.

Final benchmark claims were not changed. Raw datasets are still absent locally,
so no full-data experiments were run. The only API usage was explicitly bounded:
one Gemini preflight call, one Vertex embedding preflight text, and four Gemini
synthetic-pilot calls.

## Branch And Commits

- Base branch: `codex/autonomous-improvements-20260529-0436`
- Base commit: `809e372 docs: mark autonomous run complete`
- Second-run branch: `codex/api-validation-improvements-20260529-2211`

Commits made:

- `d0d8f16` - `docs: start api validation run audit`
- `c8593c8` - `scripts: add safe api preflight`
- `c6216fe` - `scripts: add raw data preflight`
- `1e8ac9b` - `eval: add validation guardrail utility`
- `3b8ea2b` - `eval: add cost frontier summary utility`
- `5f2b2ca` - `api: add gemini budget gate`
- `c869b3b` - `docs: add experiment evidence dashboard`
- Final report/status commit follows this update.

## Push Status

Every milestone commit above was pushed successfully to:

```text
origin/codex/api-validation-improvements-20260529-2211
```

## Tests Run

- Baseline after branch creation: `uv run pytest -q` -> `151 passed, 1 warning`.
- Second-run final smoke:
  - `uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_second --pytest-mode targeted`
  - passed with nested targeted pytest `18 passed`.
- Reader toy smoke:
  - `uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_second`
  - passed; EM `0.0`, F1 `0.490079`.
- Targeted tests added and passing:
  - `tests/test_api_preflight.py` -> `3 passed`
  - `tests/test_data_preflight.py` -> `2 passed`
  - `tests/test_validation_guardrail.py` -> `6 passed`
  - `tests/test_cost_frontier.py` -> `5 passed`
  - `tests/test_gemini_baseline.py` -> `5 passed`
  - `tests/test_experiment_dashboard.py` -> `2 passed`
- Final full test run:
  - `uv run pytest -q` -> `171 passed, 1 warning`.

The warning is the existing pandas `ConstantInputWarning` from
`test_semantic_state_features_can_use_deeper_rank_profile`.

## API Calls Made

Second-run API usage:

- Gemini new calls: 5 total.
- Vertex embedding new texts: 1 total.
- API preflight:
  - Gemini: estimated 1, actual 1, succeeded.
  - Vertex embedding: estimated 1, actual 1, succeeded.
- Synthetic Gemini pilot:
  - dry-run before live: 0 cache hits, 4 cache misses, 0 calls.
  - bounded live run: 4 cache misses, 4 new Gemini calls.
  - dry-run after live: 4 cache hits, 0 cache misses, 0 calls.

No credentials or `.env` values were printed or committed. API caches and pilot
outputs remain ignored by git.

## Data Runs Made

- Data preflight ran and found required raw data missing:
  - HotpotQA required dev distractor file missing.
  - Natural Questions validation parquet missing.
  - SciFact corpus/queries/train qrels/test qrels missing.
  - NFCorpus corpus/queries/train/dev/test qrels missing.
- No full-data SciFact/NFCorpus/Hotpot/NQ experiment was run.
- No real-data reader evaluation was run.

## New Artifacts

Committed source/scripts/tests:

- `src/selective_rag_rl/api_preflight.py`
- `scripts/run_api_preflight.py`
- `tests/test_api_preflight.py`
- `src/selective_rag_rl/data_preflight.py`
- `scripts/run_data_preflight.py`
- `tests/test_data_preflight.py`
- `src/selective_rag_rl/validation_guardrail.py`
- `scripts/run_validation_guardrail.py`
- `tests/test_validation_guardrail.py`
- `src/selective_rag_rl/cost_frontier.py`
- `scripts/run_cost_frontier_summary.py`
- `tests/test_cost_frontier.py`
- `src/selective_rag_rl/experiment_dashboard.py`
- `scripts/run_experiment_dashboard.py`
- `tests/test_experiment_dashboard.py`

Committed docs/results:

- `docs/API_EXPERIMENTS.md`
- `docs/EXPERIMENT_DASHBOARD.md`
- `outputs/results/scifact_validation_guardrail.csv/json`
- `outputs/results/nfcorpus_validation_guardrail.csv/json`
- `outputs/results/scifact_cost_frontier_summary.csv/json`
- `outputs/results/scifact_cost_frontier_summary_frontier.csv`
- `outputs/results/nfcorpus_cost_frontier_summary.csv/json`
- `outputs/results/nfcorpus_cost_frontier_summary_frontier.csv`
- `outputs/results/experiment_dashboard.csv`
- second-run reports under `docs/codex_runs/20260529_221154/`

Ignored local evidence:

- `outputs/codex_api_preflight/`
- `outputs/codex_data_preflight/`
- `outputs/codex_smoke_second/`
- `outputs/codex_reader_smoke_second/`
- `outputs/codex_gemini_pilot/`
- `outputs/cache/codex_gemini_rewrites_synthetic.jsonl`

## Claims Changed

No final benchmark claim changed. New claims are limited to:

- API credentials and SDK paths were preflighted successfully for tiny calls.
- Raw data is currently missing locally.
- Validation guardrail and cost frontier utilities now exist and have generated
  machine-readable summaries from existing artifacts.
- A tiny synthetic Gemini pilot ran under explicit budget and is cache-resumable.
- The experiment dashboard classifies artifacts by evidence level.

The synthetic Gemini pilot and smoke outputs are not benchmark evidence.

## Known Limitations

- Raw datasets are absent, so full-data reruns remain blocked.
- The Gemini pilot used synthetic data and only two held-out examples.
- Vertex embedding was only tested with one preflight text; no semantic-feature
  policy run was executed.
- Validation guardrail outputs for SciFact/NFCorpus are `analysis_only_no_validation`
  because the detailed CSVs have train/test rows but no validation split.
- Dashboard evidence-level classification is heuristic and should be reviewed
  for older artifact filenames.

## Manual Review Checklist

1. Review `.gitignore` and confirm `.env`, cache, raw data, and API pilot
   outputs remain ignored.
2. Review `docs/API_EXPERIMENTS.md` and `scripts/run_api_preflight.py`.
3. Review `src/selective_rag_rl/gemini_baseline.py` to confirm live API calls
   are blocked unless explicitly allowed and within budget.
4. Inspect `outputs/results/scifact_validation_guardrail.csv` and
   `outputs/results/nfcorpus_validation_guardrail.csv`.
5. Inspect `outputs/results/scifact_cost_frontier_summary.csv` and
   `outputs/results/nfcorpus_cost_frontier_summary.csv`.
6. Review `docs/EXPERIMENT_DASHBOARD.md` and
   `outputs/results/experiment_dashboard.csv` for evidence-level labels.
7. Download raw datasets before attempting any real-data or semantic-feature
   experiment.

## Recommended Next Prompt

Continue from branch `codex/api-validation-improvements-20260529-2211`. First
run `git status --short`, `git pull --ff-only`, and `uv run pytest -q`. Then
prioritize Vertex embedding budget enforcement in `VertexTextEmbeddingProvider`
and semantic-feature scripts. Add explicit `--allow-api` and
`--max-new-embedding-texts` controls before running any semantic-feature pilot.
If raw NFCorpus data is available, run a tiny fake-embedder real-data smoke
first, then a one-digit Vertex semantic pilot only if preflight misses are under
budget.
