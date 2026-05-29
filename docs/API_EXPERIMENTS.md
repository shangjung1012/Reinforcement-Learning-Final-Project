# API Preflight And Bounded Experiment Guide

This repository can use Google GenAI through Vertex AI for optional Gemini
rewrite and Vertex embedding experiments. API use is deliberately separate from
the default tests and smoke reproduction.

## Required Local Configuration

Create a local `.env` file with the required variable names. Do not commit this
file.

```text
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=...
GOOGLE_CLOUD_LOCATION=...
GOOGLE_APPLICATION_CREDENTIALS=...
GEMINI_MODEL=...
```

The preflight code reports only variable names, presence/missing status, and
credential-file basename. It does not print `.env` values.

## No-Cost Preflight

Run this first:

```bash
uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight
```

Expected behavior:

- no external API calls;
- `.env` existence and variable-name presence are checked;
- credential file existence is checked by basename only;
- `outputs/codex_api_preflight/api_preflight_summary.json` and `.csv` are
  written as ignored local artifacts.

## Explicit One-Call API Smoke

Only after the dry-run passes, run an explicitly bounded API smoke:

```bash
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_api_preflight.py \
  --provider all \
  --output-dir outputs/codex_api_preflight \
  --allow-api \
  --max-new-gemini-calls 1 \
  --max-new-embedding-texts 1
```

This makes at most one Gemini generation call and one Vertex embedding text
request. If a budget is zero or credentials are missing, the script records a
blocked result instead of calling the API.

## Claim Boundary

Passing API preflight means credentials, model access, and the SDK path work for
tiny calls. It does not support benchmark claims. Full generated-action or
semantic-feature claims require cached/budgeted runs on real data, repeated-seed
validation, and guardrail checks.
