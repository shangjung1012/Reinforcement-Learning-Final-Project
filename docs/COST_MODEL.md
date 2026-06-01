# Cost Model

The project uses a proxy cost model to make retrieval-action routing
cost-aware. The costs are simple by design: they make policy decisions
comparable without requiring live latency measurements or paid API calls in
default tests.

## Main Reward

The retrieval-policy experiments use the following style of reward:

```text
Recall@5 + 0.5 * MRR - rewrite_cost - extra_retrieval_call_cost
```

The exact fields are recorded in experiment outputs and should be checked for
the specific script and dataset. This reward is a retrieval-stage utility, not
an end-to-end answer-generation metric.

## Rewrite Cost

Rewrite cost penalizes actions that alter or generate the query. Deterministic
keyword rewrites usually have small proxy costs. LLM-generated rewrite,
decomposition, HyDE, and multi-query actions should include explicit generation
costs or be treated as expensive experts.

When reporting generated actions, record:

- whether the action was actually enabled,
- whether cache hits avoided new API calls,
- generation model or provider when applicable,
- token or proxy cost if available,
- output cache path if used.

## Retrieval-Call Cost

Retrieval-call cost penalizes actions that make more than one retrieval call.
For example:

- BM25 keep or dense keep usually counts as one retrieval call.
- Hybrid retrieval can cost more if it combines multiple retrieval passes.
- Multi-query or decomposition actions should account for each generated query
  that triggers retrieval.

The cost should be interpreted as a latency and systems-load proxy, not as a
measured wall-clock benchmark unless timing was explicitly measured.

## Constrained Utility

Constrained or Lagrangian sweeps vary a call-cost penalty to show how much
reward the policy can keep under tighter retrieval-call budgets. These sweeps
are useful for deployment-style tradeoff analysis:

- low penalty emphasizes retrieval quality,
- high penalty emphasizes cheaper actions,
- the selected point should be justified by the intended budget.

When presenting the frontier, include expected calls and reward or utility on
the same table or figure.

Executable frontier summaries are available through:

```bash
uv run python scripts/run_cost_frontier_summary.py --dataset scifact --summary-csv outputs/results/scifact_retrieval_policy_summary.csv --output-csv outputs/results/scifact_cost_frontier_summary.csv --budgets 1.0,1.25,1.5,2.0
uv run python scripts/run_cost_frontier_summary.py --dataset nfcorpus --summary-csv outputs/results/nfcorpus_retrieval_policy_summary.csv --output-csv outputs/results/nfcorpus_cost_frontier_summary.csv --budgets 1.0,1.25,1.5,2.0
```

The budget CSV records the best non-oracle feasible method at each expected-call
budget. The paired frontier CSV marks methods that are dominated by another
method with no lower reward and no higher call cost.

## Smoke Versus Full Evidence

Raw-data-free smoke runs use synthetic fixtures and fake embeddings. They check
that reward and cost fields flow through the pipeline, but they do not estimate
real latency, real provider cost, or final benchmark performance.

Full-data SciFact and NFCorpus runs can support cost-aware retrieval claims
when they record:

- dataset and split,
- number of examples and corpus mode,
- action space,
- reward formula,
- retrieval-call penalty,
- rewrite/generation cost assumptions,
- policy model and feature set,
- seed,
- whether external semantic or LLM features were used.

## Limitations

- Proxy retrieval-call cost is not a substitute for measured latency.
- Token-cost estimates are only reliable when token accounting is recorded by
  the provider or script output.
- Retrieval-stage utility does not prove generated-answer EM/F1 improvement.
- Changing cost weights changes the decision problem, so compare policies only
  under a shared reward formula unless the comparison is explicitly a frontier.
