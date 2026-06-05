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

## Latest Local Preflight Result

On the current local setup, dry-run preflight succeeds without printing secrets:
`.env` exists, required variable names are present, and the credential file
basename is visible to the script. After switching to the current local Google
Cloud project in `.env`, a deliberately bounded live preflight was also run with
one Gemini call and one Vertex embedding text allowed:

```bash
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_api_preflight.py \
  --provider all \
  --output-dir outputs/codex_api_preflight_new_project_live \
  --allow-api \
  --max-new-gemini-calls 1 \
  --max-new-embedding-texts 1
```

Both providers succeeded under that cap: one Gemini generation call and one
Vertex embedding text request. This validates the local credential and SDK path,
but it is still only `api_preflight` evidence, not benchmark evidence.

The Gemini HotpotQA baseline pilot is currently blocked by missing local
HotpotQA raw data, not by API access. The Vertex embedding path has been checked
with tiny SciFact/NFCorpus semantic-feature pilots; those pilots remain
`api_pilot` evidence and should not be promoted to final claims without larger
repeated-seed validation and guardrail checks.

## Gemini Baseline Budget Gate

`scripts/run_gemini_baseline.py` now supports a dry-run and explicit call budget:

```bash
uv run python scripts/run_gemini_baseline.py --data-path path/to/hotpot.json --num-examples 10 --cache-path outputs/cache/codex_gemini_rewrites.jsonl --dry-run
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_gemini_baseline.py --data-path path/to/hotpot.json --num-examples 10 --cache-path outputs/cache/codex_gemini_rewrites.jsonl --allow-api --max-new-calls 8
```

When the cache misses exceed `--max-new-calls`, or when live calls are not
explicitly allowed, the script stops before constructing the live Vertex client.

## Vertex Semantic Embedding Budget Gate

Semantic-feature runs use `VertexTextEmbeddingProvider` only when
`--semantic-features vertex` is selected. The provider now enforces the same
cache-first rule as Gemini: cache misses do not trigger live API calls unless
the run passes explicit semantic API flags.

Estimate the embedding workload first:

```bash
uv run python scripts/run_embedding_preflight.py \
  --dataset nfcorpus \
  --num-train-examples 10 \
  --num-test-examples 10 \
  --cache-path outputs/cache/codex_nfcorpus_vertex_embeddings.jsonl
```

Then run cache-only or budget-blocking retrieval commands with the default
zero budget:

```bash
uv run python scripts/run_retrieval_policy_nfcorpus.py \
  --num-train-examples 10 \
  --num-test-examples 10 \
  --embedder fake \
  --policy-model ridge \
  --semantic-features vertex \
  --semantic-cache-path outputs/cache/codex_nfcorpus_vertex_embeddings.jsonl \
  --semantic-max-new-texts 0
```

Only after the preflight misses are acceptable should a live semantic-feature
pilot include both an allow flag and a strict text budget:

```bash
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_retrieval_policy_nfcorpus.py \
  --num-train-examples 10 \
  --num-test-examples 10 \
  --embedder fake \
  --policy-model ridge \
  --semantic-features vertex \
  --semantic-cache-path outputs/cache/codex_nfcorpus_vertex_embeddings.jsonl \
  --semantic-allow-api \
  --semantic-max-new-texts 50
```

The semantic budget controls are available on the main retrieval scripts,
policy sweeps, feature ablations, learning curves, repeated selection, semantic
depth sweeps, and selected-action bandit baseline scripts. They are not used by
default tests or smoke reproduction.
