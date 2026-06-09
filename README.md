# Selective and Cost-Aware Retrieval Routing for RAG

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Package](https://img.shields.io/badge/package-selective--rag--rl-informational)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)](tests/)

Final project for the NYCU Reinforcement Learning course.

This repository formulates retrieval-stage RAG control as an offline contextual-bandit problem. Instead of applying the same retrieval strategy to every query, a lightweight policy routes each query to a cost-aware retrieval action such as BM25, dense retrieval, hybrid retrieval, or keyword rewrite. The goal is to improve evidence retrieval quality while explicitly accounting for retrieval-call and rewrite costs.

## Key Claim

A lightweight offline contextual-bandit policy can improve cost-aware evidence retrieval over strong fixed-action baselines on BEIR SciFact and NFCorpus, with paired bootstrap confidence intervals, selected-action replay diagnostics, constrained-utility analysis, and offline policy evaluation checks.

This is a retrieval-stage project. It does not claim LLM fine-tuning, a full production RAG system, or final answer-generation improvement.

## Highlights

- Offline contextual-bandit formulation for per-query retrieval routing.
- Deployable state features only: query features and BM25 confidence signals.
- Cost-aware reward combining Recall@5, MRR, rewrite cost, and retrieval-call cost.
- Retrieval actions including BM25, dense retrieval, hybrid retrieval, and keyword rewriting.
- Full-information direct-method policy training with selected-action bandit baselines.
- Doubly robust offline policy evaluation diagnostics under simulated logging policies.
- Robustness checks: bootstrap intervals, repeated seeds, budget sweeps, difficulty buckets, and cross-dataset transfer.
- Raw-data-free smoke reproduction path for reviewers.

## Main Results

Reward is defined as:

```text
Recall@5 + 0.5 * MRR - 0.10 * 1[rewrite] - 0.03 * (calls - 1)
```

| Dataset | Method | Reward | Delta vs. train-best | 95% CI | Calls/query |
| --- | --- | ---: | ---: | ---: | ---: |
| SciFact | Train-best fixed | 1.057 | — | — | 2.00 |
| SciFact | Selected-action bandit | 1.062 | +0.005 | — | 1.03 |
| SciFact | Selective policy | 1.091 | +0.034 | [0.009, 0.058] | 1.41 |
| NFCorpus | Train-best fixed | 0.377 | — | — | 1.00 |
| NFCorpus | Selected-action bandit | 0.404 | +0.027 | — | 1.14 |
| NFCorpus | Selective policy | 0.407 | +0.030 | [0.005, 0.055] | 1.00 |

Additional diagnostics show that the selective policy closes 22.6% of the SciFact oracle gap and 34.4% of the NFCorpus oracle gap. On NFCorpus, the main gain is achieved without increasing average retrieval calls.

## Method Overview

Each query is treated as a one-step decision:

```text
state s  ->  policy pi(s)  ->  retrieval action a  ->  cost-aware reward R
```

### State Features

The default deployable state contains 14 lightweight features:

- Query length features.
- Capitalized-span features.
- BM25 top score, score gap, and entropy.
- WH-word indicators.

No LLM inference is required at routing time for the main policy.

### Actions

Main retrieval actions:

- `bm25`: sparse lexical retrieval.
- `dense`: sentence-transformer retrieval.
- `hybrid`: BM25 + dense retrieval.
- `rewrite`: keyword-compression rewrite followed by retrieval.

Optional API-backed extensions include Gemini rewrite/decomposition actions and Vertex semantic features. These are treated as analysis or pilot extensions, not as the primary claim.

### Evaluation

The main evaluation focuses on retrieval-stage evidence quality and cost:

- Recall@5
- MRR
- nDCG@5
- Retrieval calls per query
- Rewrite and retrieval-call penalties
- Paired bootstrap confidence intervals
- Offline policy evaluation: DM, IPS, SNIPS, and doubly robust estimators

## Repository Layout

```text
.
├── src/selective_rag_rl/        # Package source code
├── scripts/                     # Experiment, evaluation, and artifact scripts
├── tests/                       # Raw-data-free tests
├── docs/                        # Reproduction notes, validation protocol, and claim boundaries
├── outputs/results/             # Checked-in CSV/JSON result artifacts
├── outputs/figures/             # Checked-in final figures
├── outputs/checkpoints/         # Saved policy checkpoints
├── data/raw/                    # Local raw datasets, ignored by git
├── FINAL_REPORT.md              # Full final report
├── FINAL_RESULTS_SUMMARY.md     # Compact final result summary
├── FINAL_SLIDES.md              # Presentation draft
└── pyproject.toml               # Python package and dependency metadata
```

## Installation

This project uses `uv`.

```bash
uv sync
```

The package requires Python 3.10 or newer.

## Quickstart: Raw-Data-Free Smoke Check

Run this path first when reviewing the repository. It does not require raw datasets, model downloads, Vertex/Gemini credentials, or paid API calls.

```bash
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke
```

To run the default test suite:

```bash
uv run pytest -q
```

For a lightweight reader-metric plumbing check:

```bash
uv run python scripts/run_reader_comparison.py \
  --dataset toy \
  --num-examples 4 \
  --output-dir outputs/codex_reader_comparison
```

The toy reader comparison is a smoke test only and should not be cited as final answer-quality evidence.

## Data Setup for Full Experiments

Raw datasets should be placed under `data/raw/`. They are intentionally ignored by git.

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

Check local data availability before running full-data experiments:

```bash
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight
```

## Reproducing the Main Retrieval Results

Run the two main full-corpus retrieval-policy experiments:

```bash
uv run python scripts/run_retrieval_policy_scifact.py \
  --num-train-examples 600 \
  --num-test-examples 300 \
  --seed 42 \
  --full-corpus \
  --policy-model auto

uv run python scripts/run_retrieval_policy_nfcorpus.py \
  --num-train-examples 600 \
  --num-test-examples 300 \
  --seed 42 \
  --full-corpus \
  --policy-model auto
```

Run selected-action bandit baselines:

```bash
uv run python scripts/run_bandit_baselines.py \
  --dataset scifact \
  --num-train-examples 600 \
  --num-test-examples 300 \
  --seed 42 \
  --full-corpus \
  --alpha 1.0

uv run python scripts/run_bandit_baselines.py \
  --dataset nfcorpus \
  --num-train-examples 600 \
  --num-test-examples 300 \
  --seed 42 \
  --full-corpus \
  --alpha 1.0
```

Run cross-dataset transfer:

```bash
uv run python scripts/run_cross_dataset_transfer.py \
  --num-train-examples 600 \
  --num-test-examples 300 \
  --seed 42 \
  --full-corpus
```

Regenerate final claim tables and paper-ready assets:

```bash
uv run python scripts/run_checkpoint_manifest.py --output-csv outputs/results/final_checkpoint_manifest.csv
uv run python scripts/run_evidence_consistency.py --output-csv outputs/results/final_evidence_consistency.csv
uv run python scripts/run_final_claims_matrix.py --output-csv outputs/results/final_claims_matrix.csv
uv run python scripts/run_final_paper_assets.py --results-dir outputs/results --figures-dir outputs/figures
uv run python scripts/run_artifact_index.py --output-csv outputs/results/final_artifact_index.csv
```

## Optional API-Backed Experiments

Vertex/Gemini experiments are optional and should not be run as part of the default smoke path. They may use quota and require credentials.

Run the no-cost API preflight first:

```bash
uv run python scripts/run_api_preflight.py \
  --provider all \
  --output-dir outputs/codex_api_preflight
```

Live API calls should be explicitly enabled and bounded:

```bash
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_api_preflight.py \
  --provider all \
  --output-dir outputs/codex_api_preflight \
  --allow-api \
  --max-new-gemini-calls 1 \
  --max-new-embedding-texts 1
```

## Final Defense Assets

Use these Markdown files for the report, presentation, and defense preparation:

- [`FINAL_RESULTS_SUMMARY.md`](FINAL_RESULTS_SUMMARY.md): main numbers, confidence intervals, evidence files, and limitations.
- [`FINAL_REPORT.md`](FINAL_REPORT.md): full written report.
- [`FINAL_SLIDES.md`](FINAL_SLIDES.md): slide draft with key messages and figure references.
- [`FINAL_DEFENSE_QA.md`](FINAL_DEFENSE_QA.md): prepared answers for likely defense questions.
- [`FINAL_PRESENTATION_OUTLINE.md`](FINAL_PRESENTATION_OUTLINE.md): shorter presentation outline.

Important supporting docs:

- [`docs/FINAL_REPRODUCTION.md`](docs/FINAL_REPRODUCTION.md): smoke and full-data reproduction paths.
- [`docs/RL_FRAMING.md`](docs/RL_FRAMING.md): precise RL interpretation.
- [`docs/VALIDATION_PROTOCOL.md`](docs/VALIDATION_PROTOCOL.md): model-selection and validation guardrails.
- [`docs/COST_MODEL.md`](docs/COST_MODEL.md): reward and cost interpretation.
- [`docs/API_EXPERIMENTS.md`](docs/API_EXPERIMENTS.md): optional API safety workflow.

## Claim Boundary

Supported claim:

> A lightweight offline contextual-bandit policy improves cost-aware retrieval-stage RAG performance over strong fixed-action baselines on SciFact and NFCorpus, with uncertainty and offline-evaluation diagnostics.

Do not claim:

- A new full RL-RAG architecture.
- LLM fine-tuning.
- Production deployment readiness.
- Final generated-answer quality improvement.
- Semantic-feature superiority from the API-backed pilot runs.

## Authors

- Tsai Shang-Jung
- Li Yi-Hsin
- Li Cheng-Wei

National Yang Ming Chiao Tung University

## License

No license file is currently included. Add a license before public reuse or redistribution.
