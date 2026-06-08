# Downstream QA Gap Table

The main project claim remains retrieval-stage: the learned policy improves
cost-aware evidence retrieval on SciFact and NFCorpus. Downstream QA evidence is
tracked separately so that small reader checks are not confused with final
answer-quality benchmarks.

| Experiment | Done? | Evidence Level | Can Claim? | Reason |
| --- | --- | --- | --- | --- |
| BM25 + deterministic reader | yes | tiny_realdata | no | Heuristic reader only; validates EM/F1 plumbing but not final QA quality. |
| BM25 + Gemini reader | yes | api_pilot | no | 40 examples per dataset and BM25-only retrieval. |
| train-best fixed + deterministic reader | yes | tiny_realdata | no | Policy-routed reader diagnostic only with heuristic readers. |
| learned policy + deterministic reader | yes | tiny_realdata | no | Policy retrieval is connected to EM/F1 plumbing, but deterministic readers remain weak. |
| train-best fixed + Gemini reader | no | missing | no | Not run. |
| learned policy + Gemini reader | no | missing | no | Not run. |
| retrieval-stage policy benchmark | yes | full_benchmark | yes | Main final claim is retrieval-stage cost-aware reward improvement. |

Machine-readable table: `outputs/results/downstream_qa_gap_table.csv`.

## Latest Policy-Routed Reader Diagnostic

Two deterministic reader diagnostics now connect retrieval-policy outputs to
EM/F1 evaluation without calling external APIs:

```bash
uv run python scripts/run_policy_reader_comparison.py --dataset hotpot --detailed-csv outputs/results/retrieval_policy_detailed.csv --num-examples 50 --readers lexical,span,answer_type --output-dir outputs/codex_policy_reader_hotpot_50 --publish-results
uv run python scripts/run_policy_reader_comparison.py --dataset nq --detailed-csv outputs/results/nq_retrieval_policy_detailed.csv --num-examples 50 --source-num-examples 500 --readers lexical,span,answer_type --output-dir outputs/codex_policy_reader_nq_50 --publish-results
```

HotpotQA 50 shows the retrieval policy has higher retrieval reward than BM25
and train-best fixed on this slice, but deterministic reader EM/F1 does not
improve over train-best fixed. Natural Questions 50 shows the same pattern:
retrieval improves, but heuristic answer extraction remains nearly unchanged.

This is useful diagnostic evidence, not an answer-quality claim. A final
downstream QA claim would require a stronger reader, larger repeated splits, and
the still-missing `train-best fixed + Gemini reader` versus
`learned policy + Gemini reader` comparison.
