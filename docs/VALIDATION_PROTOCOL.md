# Validation Protocol And Selection Guardrails

This project evaluates a selective retrieval-action router as an offline
contextual bandit. The validation protocol is designed to keep model-selection
claims separate from held-out-test evidence and to prevent unstable feature
experiments from being presented as final improvements.

## Data Separation

- Use train data to fit action-value models, fixed-action baselines, and any
  cost penalty or model-family candidates.
- Use validation data to choose among policy model families, feature sets,
  semantic-depth settings, confidence-gate margins, and fallback rules.
- Use held-out test data only once a selection rule is fixed.
- If a script runs repeated seeds, report both per-seed rows and aggregate
  summaries. Do not choose the final method from held-out-test seed averages
  unless that selection is explicitly labeled as analysis.

## Required Baselines

At minimum, compare a learned policy against:

- Each fixed action in the action space, such as `bm25_keep`, `dense_keep`,
  `hybrid_keyword`, and any generated-query action that is enabled.
- The train-best fixed action selected without looking at held-out test
  rewards.
- A simple deployable heuristic or confidence-gated policy when available.
- Selected-action bandit baselines when the experiment is about bandit feedback
  rather than full-information direct-method learning.

The main final comparison should keep the train-best fixed action visible. It
is the clearest baseline for judging whether contextual routing added value.

## Selection Guardrail

Before promoting a validation-selected policy, check whether it is dominated by
the train-best fixed action on validation or repeated pilot runs.

A conservative dominated rule is:

- the candidate has no higher validation reward than the train-best fixed
  action, and
- the candidate has no lower expected retrieval-call cost.

When this happens, recommend the train-best fixed action or the existing
deployment fallback unless a pre-registered reason justifies the learned policy.
For repeated pilots, require the recommendation to be stable across seeds or
label the result as exploratory.

Suggested machine-readable columns for selection diagnostics:

```text
dataset, seed, selected_config, heldout_best_config,
validation_reward, heldout_reward, reward_gap, call_gap,
dominated_by_train_best, recommendation, reason
```

Executable guardrail checks are available through:

```bash
uv run python scripts/run_validation_guardrail.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_validation_guardrail.csv
uv run python scripts/run_validation_guardrail.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_validation_guardrail.csv
```

When a detailed CSV has only train/test rows and no validation split, the tool
emits `analysis_only_no_validation`. That output is useful for auditing reward
and call gaps, but it should not be used as a model-selection rule.

## Semantic Feature Policy

Semantic state features can be useful diagnostics, but they add cache,
embedding, and validation complexity. Treat them as analysis-only unless all of
the following are true:

- the embedding cache or preflight confirms expected coverage,
- train/validation transforms do not use test statistics,
- repeated-seed validation selects the semantic configuration consistently,
- held-out-test evidence improves reward or cost at the same claim boundary,
  and
- the report records the embedding model, semantic depth, cache path, and
  whether any external API calls were made.

If these conditions are not met, keep the non-semantic or fake-embedder path as
the reproducible smoke/default path.

## OPE Diagnostics

OPE outputs are diagnostics for whether logged bandit evaluation is plausible
under a simulated or actual behavior policy. Report:

- behavior policy name,
- target policy name,
- match rate,
- effective sample size,
- IPS, SNIPS, direct-method, and doubly robust values when available,
- confidence intervals or seed variability when available.

If match rate or effective sample size is low, use the OPE result as a warning
signal rather than as the primary final claim.

## Reporting Rules

- Full-data SciFact/NFCorpus results can support retrieval-stage claims when
  the command, data split, action space, reward formula, and seed are recorded.
- Toy, smoke, and fake-embedder runs verify code paths only.
- Reader EM/F1 smoke results verify downstream metric plumbing only unless a
  full reader experiment is executed and documented.
- Do not claim online deployment, LLM fine-tuning, or final RAG answer-quality
  gains from retrieval-stage evidence alone.
