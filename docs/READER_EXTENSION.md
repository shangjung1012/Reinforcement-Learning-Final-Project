# Lightweight Reader Evaluation Extension

This repository now includes a small downstream QA evaluation path:

```bash
uv run python scripts/run_reader_eval.py --dataset toy --reader lexical --output-dir outputs/codex_reader_smoke
uv run python scripts/run_reader_eval.py --dataset toy --reader span --output-dir outputs/codex_reader_span_smoke
uv run python scripts/run_reader_eval.py --dataset toy --reader answer_type --output-dir outputs/codex_reader_answer_type_smoke
```

The default mode is intentionally lightweight:

- no raw datasets;
- no model downloads;
- no Vertex/Gemini calls;
- BM25 retrieval;
- deterministic lexical-overlap, span-heuristic, or answer-type heuristic
  reader;
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
selection. It is still `smoke_toy_reader` evidence.

After restoring HotpotQA dev distractor locally, a tiny real-data comparison was
also run:

```bash
uv run python scripts/run_reader_comparison.py --dataset hotpot --num-examples 50 --readers lexical,span --output-dir outputs/codex_reader_hotpot_realdata_50
```

Summary:

- lexical reader: exact match 0.0, token F1 0.046297, Recall@5 0.83;
- span reader: exact match 0.02, token F1 0.076500, Recall@5 0.83.

This result is useful because it confirms the deterministic reader plumbing on
real HotpotQA examples. It is still not final QA benchmark evidence: the run is
small, the readers are heuristic, and Natural Questions remains unavailable
locally.

After restoring Natural Questions and adding the answer-type reader, two larger
real-data diagnostic comparisons were run:

```bash
uv run python scripts/run_reader_comparison.py --dataset hotpot --num-examples 200 --readers lexical,span,answer_type --output-dir outputs/codex_reader_hotpot_realdata_200
uv run python scripts/run_reader_comparison.py --dataset nq --num-examples 50 --readers lexical,span,answer_type --output-dir outputs/codex_reader_nq_realdata_50
```

HotpotQA 200 summary:

- lexical reader: exact match 0.0, token F1 0.050195, Recall@5 0.82;
- span reader: exact match 0.035, token F1 0.084691, Recall@5 0.82;
- answer-type reader: exact match 0.035, token F1 0.084691, Recall@5 0.82.

Natural Questions 50 summary:

- lexical reader: exact match 0.0, token F1 0.038915, Recall@5 0.98;
- span reader: exact match 0.0, token F1 0.013333, Recall@5 0.98;
- answer-type reader: exact match 0.0, token F1 0.013333, Recall@5 0.98.

The main lesson is negative but useful: deterministic extraction is not strong
enough to support downstream RAG answer-quality claims, even when retrieval
coverage is high. These rows remain `tiny_realdata` diagnostics.

## Policy-Routed Deterministic Reader Diagnostic

The repository now also connects retrieval-policy outputs to deterministic
reader EM/F1 without calling external APIs:

```bash
uv run python scripts/run_policy_reader_comparison.py --dataset hotpot --detailed-csv outputs/results/retrieval_policy_detailed.csv --num-examples 50 --readers lexical,span,answer_type --output-dir outputs/codex_policy_reader_hotpot_50 --publish-results
uv run python scripts/run_policy_reader_comparison.py --dataset nq --detailed-csv outputs/results/nq_retrieval_policy_detailed.csv --num-examples 50 --source-num-examples 500 --readers lexical,span,answer_type --output-dir outputs/codex_policy_reader_nq_50 --publish-results
```

HotpotQA 50 summary:

- Vanilla BM25 + best deterministic reader: exact match 0.06, token F1
  0.120706, retrieval reward 1.249167.
- Train-best fixed retrieval + best deterministic reader: exact match 0.08,
  token F1 0.130706, retrieval reward 1.316167.
- Learned retrieval policy + best deterministic reader: exact match 0.06,
  token F1 0.120706, retrieval reward 1.346367.

Natural Questions 50 summary:

- Vanilla BM25 + answer-type reader: exact match 0.04, token F1 0.072857,
  retrieval reward 1.456667.
- Train-best fixed retrieval + answer-type reader: exact match 0.04, token F1
  0.072857, retrieval reward 1.495000.
- Learned retrieval policy + answer-type reader: exact match 0.04, token F1
  0.072857, retrieval reward 1.495000.

These diagnostics are useful because they test the missing policy-routed reader
plumbing. They also make the limitation clearer: better retrieval reward does
not automatically become better answer EM/F1 when the reader is a deterministic
heuristic. The rows are labeled
`tiny_realdata_policy_reader_diagnostic`, not final QA benchmark evidence.

## Bounded Gemini Reader Pilot

To test a stronger reader without pretending the deterministic heuristics are
competitive QA models, the repository also includes a bounded Gemini answer
reader:

```bash
uv run python scripts/run_gemini_reader_eval.py --dataset hotpot --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_hotpot.jsonl --dry-run --output-dir outputs/codex_gemini_reader_hotpot_dry
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_gemini_reader_eval.py --dataset hotpot --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_hotpot.jsonl --allow-api --max-new-calls 40 --output-dir outputs/codex_gemini_reader_hotpot_40

uv run python scripts/run_gemini_reader_eval.py --dataset nq --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_nq.jsonl --dry-run --output-dir outputs/codex_gemini_reader_nq_dry
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_gemini_reader_eval.py --dataset nq --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_nq.jsonl --allow-api --max-new-calls 40 --output-dir outputs/codex_gemini_reader_nq_40
```

The prompt gives Gemini only the question and retrieved BM25 passages, not the
gold answer. On the current 40-example pilots:

- HotpotQA: Gemini reader exact match 0.575 and token F1 0.628081, versus span
  reader exact match 0.0 and token F1 0.062292.
- Natural Questions: Gemini reader exact match 0.225 and token F1 0.265417,
  versus lexical reader exact match 0.0 and token F1 0.029894.

This is the first meaningful downstream-reader signal in the repo, but it is
still `api_pilot` evidence: single seed, 40 examples per dataset, BM25-only
retrieval, and no policy-routed Gemini-reader comparison.
