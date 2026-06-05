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
Vertex embedding text. Both providers returned `403 PERMISSION_DENIED` for
`aiplatform.endpoints.predict`, and successful live calls/texts remained zero.
The next required resource is Google Cloud IAM/model access for Vertex AI
prediction, not a new local `.env` variable.

## Reader Smoke

Checked with:

```bash
uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_phase
```

The deterministic lexical reader smoke completed with exact match 0.0 and token
F1 0.490079. This validates the downstream metric plumbing only; it is not
evidence for answer-generation quality.

## Remaining Work

- Add HotpotQA and NQ raw data if downstream real-data reader EM/F1 evidence is
  required.
- Grant Vertex AI prediction/model access before claiming Gemini/Vertex live API
  availability.
- Keep the deterministic reader path labeled as smoke unless a stronger reader
  experiment is run on real data with baselines.
- Use `docs/EXPERIMENT_DASHBOARD.md` before updating final reports or the
  poster; rows with `claim_allowed=false` should remain analysis-only.
