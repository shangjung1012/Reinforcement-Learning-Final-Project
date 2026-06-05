# Lightweight Reader Evaluation Extension

This repository now includes a small downstream QA evaluation path:

```bash
uv run python scripts/run_reader_eval.py --dataset toy --reader lexical --output-dir outputs/codex_reader_smoke
uv run python scripts/run_reader_eval.py --dataset toy --reader span --output-dir outputs/codex_reader_span_smoke
```

The default mode is intentionally lightweight:

- no raw datasets;
- no model downloads;
- no Vertex/Gemini calls;
- BM25 retrieval;
- deterministic lexical-overlap or span-heuristic reader;
- SQuAD-style exact match and token F1 metrics.

## What It Measures

The script retrieves passages, selects text from the retrieved passages, and
writes:

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

This is a downstream reader smoke check, not a new final benchmark. Both reader
modes are deterministic and useful for testing evaluation plumbing, but neither
is a competitive QA model and neither justifies changing the main project claim.

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

## Latest Local Smoke Result

The current raw-data-free smoke command completed successfully:

```bash
uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_phase
```

It produced exact match 0.0 and token F1 0.490079 with retrieval Recall@5 1.0.
That is enough to verify EM/F1 plumbing and output schemas, but not enough to
claim downstream RAG answer-quality improvement.

The span heuristic reader can also be compared against the lexical reader:

```bash
uv run python scripts/run_reader_comparison.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_comparison_api_run
```

On the synthetic toy fixture, lexical reader exact match is 0.0 with token F1
0.490079, while the span reader reaches exact match 1.0 and token F1 1.0. This
means the toy fixture now exercises answer-span extraction, not only sentence
selection. It is still `smoke_toy_reader` evidence because HotpotQA and NQ raw
data are missing locally.
