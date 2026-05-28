# Lightweight Reader Evaluation Extension

This repository now includes a small downstream QA evaluation path:

```bash
uv run python scripts/run_reader_eval.py --dataset toy --output-dir outputs/codex_reader_smoke
```

The default mode is intentionally lightweight:

- no raw datasets;
- no model downloads;
- no Vertex/Gemini calls;
- BM25 retrieval;
- deterministic lexical-overlap reader;
- SQuAD-style exact match and token F1 metrics.

## What It Measures

The script retrieves passages, selects a sentence from the retrieved passages
with `LexicalOverlapReader`, and writes:

- `reader_eval_detailed.csv`
- `reader_eval_summary.csv`
- `reader_eval_summary.json`

The detailed CSV includes:

- `qid`
- `question`
- `gold_answer`
- `predicted_answer`
- `exact_match`
- `f1`
- retrieval metrics: `recall_at_5`, `mrr`, `ndcg_at_5`

## Claim Boundary

This is a downstream reader smoke check, not a new final benchmark. The reader
is deterministic and useful for testing evaluation plumbing, but it is not a
competitive QA model and does not justify changing the main project claim.

The main supported claim remains retrieval-stage:

> A lightweight offline contextual-bandit policy can improve cost-aware
> retrieval-stage RAG performance over strong fixed-action baselines on SciFact
> and NFCorpus.

Do not claim generated-answer improvement unless a stronger reader is run on
real datasets and those artifacts are added to the final evidence matrix.

## Full-Data Modes

If local raw data exists, the script can also run:

```bash
uv run python scripts/run_reader_eval.py --dataset hotpot --num-examples 20 --output-dir outputs/codex_reader_smoke
uv run python scripts/run_reader_eval.py --dataset nq --num-examples 20 --output-dir outputs/codex_reader_smoke
```

If the requested raw file is missing, the script raises a clear error and
suggests using `--dataset toy` for a raw-data-free smoke run.
