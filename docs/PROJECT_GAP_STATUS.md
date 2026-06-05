# Project Gap Status

This file tracks the main remaining gaps after the code-quality and evidence
dashboard fixes.

## Fixed In This Pass

- Pytest warning removed: semantic rank-agreement features now avoid undefined
  Spearman correlation calls when semantic scores are constant.
- Evidence dashboard added: `scripts/run_experiment_dashboard.py` writes
  `outputs/results/experiment_dashboard.csv` and `docs/EXPERIMENT_DASHBOARD.md`.
- Poster claim audit added: `docs/POSTER_CLAIM_AUDIT.md` confirms the poster
  claim boundary is aligned with the retrieval-stage evidence.
- Bandit replay-regret diagnostics added:
  `scripts/run_bandit_replay_diagnostics.py` writes full-corpus
  selected-action replay summaries and regret figures for SciFact and NFCorpus,
  making the distinction between full-information direct-method learning and
  chosen-action feedback explicit.
- API live preflight now succeeds under the updated local `.env`: one bounded
  Gemini generation call and one bounded Vertex embedding text request both
  completed.
- A span heuristic reader was added alongside the lexical reader, plus
  `scripts/run_reader_comparison.py` for explicit toy/real-data reader
  comparisons.
- FQI post-hoc diagnostics were added:
  `scripts/run_fqi_diagnostics.py` writes the reward/call gap against
  train-best fixed trace and the selected trace distribution.

## Data Availability

Checked with:

```bash
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight
```

Current local status:

- SciFact is available: 5,183 corpus rows, 1,109 queries, 919 train qrels, 339
  test qrels.
- NFCorpus is available: 3,633 corpus rows, 3,237 queries, 110,575 train qrels,
  11,385 dev qrels, 12,334 test qrels.
- HotpotQA is blocked: `data/raw/HotpotQA/hotpot_dev_distractor_v1.json` is
  missing.
- Natural Questions is blocked:
  `data/raw/natural-questions/default/validation-00000-of-00007.parquet` is
  missing.

Small full-corpus smoke runs were completed for SciFact and NFCorpus with fake
embeddings and 30 train / 30 test examples. These runs validate real-data code
paths only; they are not final benchmark replacements.

## API Availability

Checked with:

```bash
uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight
```

The dry-run preflight sees `.env`, the required variable names, optional Vertex
settings, and the credential file basename. It made zero Gemini calls and zero
Vertex embedding requests.

A bounded live API preflight was also attempted with both
`CODEX_ALLOW_API_CALLS=1` and `--allow-api`, capped at one Gemini call and one
Vertex embedding text. Under the updated local Google Cloud project, both
providers succeeded. This establishes API reachability only; it does not change
the final benchmark claim.

The Gemini HotpotQA baseline pilot is still blocked because
`data/raw/HotpotQA/hotpot_dev_distractor_v1.json` is missing. Vertex semantic
feature pilots were run on tiny SciFact/NFCorpus splits:

- NFCorpus 10/10: selective reward tied train-best fixed at 0.212549.
- SciFact 10/10: selective reward tied train-best fixed at 0.900000.
- NFCorpus 30/30: selective reward 0.410110 versus train-best fixed 0.401221.
- SciFact 30/30: selective reward 1.058889 versus train-best fixed 1.061667.

These are API pilots, not final evidence. They show the path runs and that the
signal is mixed at tiny scale.

## Reader Smoke

Checked with:

```bash
uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_phase
```

The deterministic lexical reader smoke completed with exact match 0.0 and token
F1 0.490079. The span heuristic reader improves the synthetic toy fixture:

```bash
uv run python scripts/run_reader_comparison.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_comparison_api_run
```

Toy comparison result:

- lexical reader: exact match 0.0, token F1 0.490079;
- span reader: exact match 1.0, token F1 1.0.

This validates stronger downstream metric plumbing on a synthetic fixture only;
it is not evidence for answer-generation quality on HotpotQA or NQ.

## FQI Extension Status

Checked with:

```bash
uv run python scripts/run_fqi_diagnostics.py --dataset hotpot --detailed-csv outputs/results/multistep_detailed.csv --summary-csv outputs/results/hotpot_fqi_diagnostics_summary.csv --trace-csv outputs/results/hotpot_fqi_trace_distribution.csv --split test
```

The existing HotpotQA FQI artifact has 240 test examples. Multi-step FQI reward
is 1.222521, below the train-best fixed trace at 1.264444, while the two-step
oracle reaches 1.376264. FQI also averages 2.4375 retrieval calls versus 2.0 for
the train-best fixed trace. This is a useful RL extension and failure analysis,
but it should not be presented as a main win.

## Remaining Work

- Add HotpotQA and NQ raw data if downstream real-data reader EM/F1 evidence is
  required.
- Keep Gemini/Vertex pilots labeled as API pilots until larger repeated-seed
  guardrail checks support them.
- Keep the deterministic reader paths labeled as smoke unless a stronger reader
  experiment is run on real data with baselines.
- Treat FQI as an extension/limitation unless a stronger state representation or
  value model beats train-best fixed trace under validation.
- Use `docs/EXPERIMENT_DASHBOARD.md` before updating final reports or the
  poster; rows with `claim_allowed=false` should remain analysis-only.
