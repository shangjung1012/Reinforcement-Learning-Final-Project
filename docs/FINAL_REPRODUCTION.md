# Final Reproduction Quickstart

This project has two reproduction paths:

- Smoke reproduction: verifies core code paths without raw datasets, external
  APIs, or model downloads.
- Full-data reproduction: reruns the reported experiments when the raw datasets
  are available under `data/raw/`.

## Environment

```bash
uv sync
```

## Smoke Reproduction

Use this command for a fast reviewer check:

```bash
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke
```

The smoke runner:

- runs a targeted pytest subset by default;
- writes a synthetic HotpotQA-style fixture;
- runs a fake-embedder retrieval-action policy smoke;
- runs OPE diagnostics from the generated detailed CSV;
- writes `outputs/codex_smoke/smoke_manifest.json`.

It does not use raw datasets, Vertex/Gemini, paid APIs, or downloaded embedding
models. To run the entire pytest suite as part of the smoke:

```bash
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke --pytest-mode full
```

For script-only debugging without nested pytest:

```bash
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke --pytest-mode skip
```

`outputs/codex_smoke/` is ignored by git because it is generated local evidence.

## Tests

Run the default test suite:

```bash
uv run pytest -q
```

The default tests are intended to be raw-data-free. Full-data experiments are
covered by separate scripts and require the datasets below.

## Evidence Dashboard

After generating or updating result artifacts, run the evidence dashboard to
separate final benchmark evidence from smoke checks and API pilots:

```bash
uv run python scripts/run_experiment_dashboard.py --output-csv outputs/results/experiment_dashboard.csv --output-md docs/EXPERIMENT_DASHBOARD.md
```

Use `claim_allowed=false` rows as analysis-only or engineering checks. In
particular, smoke runs, toy reader checks, and Vertex/Gemini pilot artifacts
should not be promoted to final claims without stronger real-data evidence.

## Full-Data Reproduction

Raw datasets should be placed under this layout:

```text
data/raw/HotpotQA/hotpot_dev_distractor_v1.json
data/raw/HotpotQA/hotpot_dev_fullwiki_v1.json
data/raw/HotpotQA/hotpot_train_v1.1.json
data/raw/natural-questions/default/validation-00000-of-00007.parquet
data/raw/scifact/corpus.jsonl
data/raw/scifact/queries.jsonl
data/raw/scifact/qrels/train.tsv
data/raw/scifact/qrels/test.tsv
data/raw/nfcorpus/corpus.jsonl
data/raw/nfcorpus/queries.jsonl
data/raw/nfcorpus/qrels/train.tsv
data/raw/nfcorpus/qrels/dev.tsv
data/raw/nfcorpus/qrels/test.tsv
```

The download helper is:

```bash
cd data/raw
bash download.sh
```

Before launching real-data experiments, check local data availability:

```bash
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight
```

After data is available, use the commands in `README.md` to rerun the full
SciFact, NFCorpus, HotpotQA, Natural Questions, diagnostics, and final artifact
generation flows. The main full-data result artifacts are already checked in
under `outputs/results/`, `outputs/figures/`, and `outputs/checkpoints/`.

## External APIs

Vertex/Gemini rewrite and embedding experiments are optional. Do not run them
for smoke reproduction. They require credentials and may use quota. Cached
artifacts under `outputs/cache/` are local generated data and are ignored by git.

Before any API-backed experiment, run the no-cost preflight:

```bash
uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight
```

To intentionally make one tiny Gemini call and one Vertex embedding request,
use both an allow flag and explicit budgets:

```bash
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight --allow-api --max-new-gemini-calls 1 --max-new-embedding-texts 1
```

See `docs/API_EXPERIMENTS.md` for the full API safety workflow.

For Vertex semantic-feature experiments, first estimate cache misses without
calling the API:

```bash
uv run python scripts/run_embedding_preflight.py --dataset nfcorpus --num-train-examples 10 --num-test-examples 10 --cache-path outputs/cache/codex_nfcorpus_vertex_embeddings.jsonl
```

Then use `--semantic-features vertex` only with cache coverage or explicit
semantic API controls. The default `--semantic-max-new-texts 0` blocks cache
misses before the live client is created. A deliberate pilot must include both
`--semantic-allow-api` and a bounded `--semantic-max-new-texts` value.

## Claim Boundary

The main supported claim remains retrieval-stage and cost-aware:

> A lightweight offline contextual-bandit policy can improve cost-aware
> retrieval-stage RAG performance over strong fixed-action baselines on SciFact
> and NFCorpus, with bootstrap, OPE, selected-action baseline, constrained
> utility, and robustness diagnostics.

Do not interpret the smoke runner as full benchmark evidence. It is a fast
integration check for the code paths that produce retrieval-policy and OPE
artifacts.
