# Adaptive RAG Depth and Transfer Experiment Design

## Objective

Extend the final project from reporting aggregate policy gains to explaining
when and why the learned retrieval-action policy helps. The new experiments
should support two claims:

1. The policy adapts retrieval strategy based on query difficulty and retrieval
   uncertainty, not only on simple keyword heuristics.
2. The learned retrieval-confidence signal can be tested for cross-domain
   robustness across BEIR SciFact and NFCorpus.

## Scope

This design covers two experiment tracks.

### Track A: Query Difficulty and Policy Behavior

Add diagnostics that replay existing full-corpus detailed CSVs and bucket
queries by deployable difficulty signals:

- BM25 top-1 score.
- BM25 top-score gap.
- BM25 retrieval entropy.
- Query length.
- Predicted policy confidence margin.
- Oracle reward margin.
- Retriever/action disagreement, measured from per-action rewards and selected
  actions already present in the detailed CSV.

For each dataset and bucket, export:

- query count;
- train-best, heuristic-router, selective-policy, confidence-gated, and oracle
  reward;
- reward gaps versus train-best and heuristic-router;
- Recall@5, MRR, nDCG@5, rewrite cost, and retrieval calls;
- dense/hybrid/action-family usage rate for the selective policy;
- oracle headroom and oracle tie rate where available.

The first implementation should run on:

- `outputs/results/scifact_retrieval_policy_detailed.csv`;
- `outputs/results/nfcorpus_retrieval_policy_detailed.csv`.

Expected artifacts:

- `outputs/results/scifact_complexity_buckets.csv`;
- `outputs/results/nfcorpus_complexity_buckets.csv`;
- `outputs/results/scifact_complexity_action_distribution.csv`;
- `outputs/results/nfcorpus_complexity_action_distribution.csv`.

### Track C: Cross-Dataset Policy Robustness

Add a transfer evaluation path that trains a retrieval-action policy on one
BEIR dataset and evaluates it on another dataset's full-corpus test split using
the same non-semantic 14-dimensional feature space and six base retrieval
actions.

The initial transfer matrix should include:

- train SciFact, evaluate SciFact;
- train SciFact, evaluate NFCorpus;
- train NFCorpus, evaluate NFCorpus;
- train NFCorpus, evaluate SciFact.

The evaluation should compare:

- train-best fixed action from the target evaluation setting;
- heuristic retrieval router;
- source-trained transferred policy;
- target-trained policy;
- oracle retrieval action.

If the transferred policy underperforms target-trained policy, that is still a
valid result. The report should interpret it as evidence that retrieval routing
needs target-domain calibration, not as a failed experiment.

Expected artifacts:

- `outputs/results/cross_dataset_transfer_summary.csv`;
- `outputs/results/cross_dataset_transfer_detailed.csv`;
- optional checkpoints under `outputs/checkpoints/` if the transfer pipeline
  saves source-trained policies separately.

Few-shot adaptation is intentionally second priority. It should be added only
after source-only transfer works. The planned few-shot settings are 50, 100, and
200 target train examples, comparing target-only small-data policy against
source-plus-target adaptation.

## Architecture

### Complexity Diagnostics

Create a small analysis module that works from detailed experiment rows instead
of rerunning retrieval. The module should:

1. Load a detailed CSV.
2. Infer query-level records by grouping rows on split, query id, and method.
3. Extract feature columns and method metrics.
4. Build bucket definitions from quantiles or deterministic thresholds.
5. Aggregate method metrics per bucket.
6. Export a compact bucket summary and action-family distribution table.

This keeps Track A fast and reproducible. It should not call Vertex, download
datasets, or recompute dense embeddings.

### Transfer Evaluation

Prefer reusing the current retrieval-policy experiment code rather than adding a
separate training stack. The transfer path should:

1. Build source train examples and target test examples using the existing
   SciFact/NFCorpus loaders.
2. Evaluate all six base actions on both source and target examples.
3. Train the direct-method bandit on source action rewards and source features.
4. Apply the trained bandit to target features.
5. Report target metrics using the same `_method_row` aggregation conventions
   as the main experiment.

The implementation should keep action names identical across datasets:

- `bm25_keep`;
- `bm25_keyword`;
- `dense_keep`;
- `dense_keyword`;
- `hybrid_keep`;
- `hybrid_keyword`.

No semantic Vertex features are required for the first transfer experiment.

## Testing

Add focused tests with synthetic detailed rows and tiny fake examples:

- bucket assignment produces stable low/mid/high groups;
- bucket summaries include selective, heuristic, train-best, and oracle rows;
- action-family distribution counts dense and hybrid selections correctly;
- transfer evaluation can apply a source-trained policy to a target dataset
  without retraining on target rewards;
- artifact index includes the new Track A and Track C outputs.

Full verification remains:

```bash
uv run pytest -q
uv run python -m compileall -q src scripts
git diff --check
```

## Reporting

Add two report sections after the current policy diagnostics:

1. `Query Difficulty and Policy Behavior`
   - Summarize where learned policy beats heuristic-router and train-best.
   - Emphasize whether gains concentrate in high-uncertainty buckets.
   - Report action-family shifts across easy versus hard buckets.

2. `Cross-Dataset Policy Transfer`
   - Report source-to-target transfer reward and calls.
   - Compare transferred policy against target-trained policy and heuristic
     router.
   - Interpret negative transfer as domain-calibration evidence if it occurs.

The report should keep the claim scoped: this is cost-aware retrieval-action
selection, not end-to-end RAG answer generation or LLM fine-tuning.

## Success Criteria

The next implementation is successful if:

- complexity bucket CSVs are generated for SciFact and NFCorpus;
- the bucket analysis identifies at least one interpretable difference between
  learned policy and heuristic-router behavior;
- cross-dataset transfer runs in both directions and exports summary/detailed
  CSVs;
- all new artifacts are indexed;
- tests and compile checks pass;
- final report text explains both positive and negative results without
  overstating claims.

