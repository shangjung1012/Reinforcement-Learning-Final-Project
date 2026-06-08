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
| train-best fixed + Gemini reader | yes | api_pilot | no | 30 examples per dataset; policy-routed Gemini pilot only. |
| learned policy + Gemini reader | yes | api_pilot | no | 30 examples per dataset; policy-routed Gemini pilot only. |
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

This is useful diagnostic evidence, not an answer-quality claim. The stronger
policy-routed Gemini reader comparison has also been run as a bounded API pilot:

```bash
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_policy_gemini_reader_comparison.py --dataset hotpot --detailed-csv outputs/results/retrieval_policy_detailed.csv --num-examples 30 --cache-path outputs/cache/codex_policy_gemini_reader_hotpot.jsonl --output-dir outputs/codex_policy_gemini_reader_hotpot_30 --allow-api --max-new-calls 75 --publish-results
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_policy_gemini_reader_comparison.py --dataset nq --detailed-csv outputs/results/nq_retrieval_policy_detailed.csv --num-examples 30 --source-num-examples 500 --cache-path outputs/cache/codex_policy_gemini_reader_nq.jsonl --output-dir outputs/codex_policy_gemini_reader_nq_30 --allow-api --max-new-calls 68 --publish-results
```

HotpotQA 30 shows a promising policy-routed Gemini-reader signal: BM25 exact
match/F1 is 0.733333/0.823420, train-best fixed is 0.766667/0.851905, and the
learned policy is 0.800000/0.890087. Natural Questions 30 does not show the
same answer-quality separation: all three methods have exact match 0.133333,
with F1 0.171005 for BM25, 0.166106 for train-best fixed, and 0.169916 for the
learned policy.

This remains `api_pilot` evidence. A final downstream QA claim would still
require larger repeated splits, cache-first reruns, and guardrail checks showing
the learned policy consistently improves answer EM/F1 rather than only retrieval
reward.
