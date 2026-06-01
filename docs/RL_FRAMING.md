# RL Framing

This project is best described as retrieval-stage offline contextual-bandit
control for RAG. It is not an online deep-RL deployment and it does not
fine-tune an LLM.

## Contextual Bandit Formulation

For each query, the environment exposes a context:

- lexical query features;
- question type features;
- shallow BM25 retrieval confidence features;
- optional semantic or retrieval-contrast features in pilot experiments.

The policy chooses one action from a finite set. In the main retrieval-action
experiments, actions combine retriever choice and query form:

- `bm25_keep`
- `bm25_keyword`
- `dense_keep`
- `dense_keyword`
- `hybrid_keep`
- `hybrid_keyword`

Optional generated-query actions add Gemini rewrite, decomposition, HyDE, and
multi-query experts, but these are treated as expensive analysis extensions.

The main reward is:

```text
Recall@5 + 0.5 * MRR - RewriteCost - ExtraRetrievalCallCost
```

This is a one-step decision problem: the policy observes the query state,
chooses one action, and receives retrieval reward minus cost.

## Why Direct Method Is Still Bandit Policy Learning

The main policy uses full-information offline replay. For every training query,
the code can evaluate each retrieval action against the same fixed corpus and
gold qrels. The direct-method policy then trains one reward predictor per action
and selects the action with the highest predicted reward.

This has supervised regression inside the estimator, but the optimized object is
policy value, not a human class label. The learned decision is:

```text
argmax_a E[reward | state, action=a]
```

That is why the final evaluation reports realized policy reward, retrieval
calls, cost, bootstrap intervals, and policy regret rather than classification
accuracy.

## Selected-Action Bandit Feedback

The repository also includes selected-action replay baselines:

- LinUCB;
- epsilon-greedy linear regression;
- linear Thompson sampling.

These baselines train sequentially over the replay table. At each training step
they choose one action, observe only that chosen action's reward, update that
arm/model, and record regret against the per-query oracle. This separates the
harder bandit-feedback setting from the full-information direct-method policy.

The selected-action baselines are important for RL framing, but they are not the
main strongest method in the current evidence.

## Two-Step FQI Extension

The HotpotQA multi-step script is closer to a finite-horizon MDP. It includes:

- a state after an initial retrieval;
- stop/refine actions;
- feedback-aware actions such as `feedback_expand` and `title_bridge`;
- fitted-Q estimators for two decision steps.

This extension demonstrates sequential refinement and cost accumulation, but
the final BEIR claims rely on the one-step retrieval-action contextual-bandit
experiments.

## Off-Policy Evaluation

The OPE diagnostics simulate logged bandit feedback from the full-information
action table. They compare:

- direct method;
- inverse propensity scoring;
- self-normalized IPS;
- doubly robust estimation.

The diagnostic goal is not to inflate the main result. It is to show how
logging coverage affects offline policy-value estimation. When the logged
behavior rarely chooses the target policy's actions, IPS/SNIPS can have high
error or no effective support.

## Cost-Aware And Constrained Variants

The constrained-policy sweep retrains a direct-method utility policy under
different retrieval-call penalties:

```text
Recall@5 + 0.5 * MRR - RewriteCost - lambda * max(calls - 1, 0)
```

This is a Lagrangian offline replay over the existing action table. It shows a
reward/call frontier, but it is not a production constrained MDP with hard
latency guarantees.

## What To Claim

Defensible:

> A lightweight offline contextual-bandit policy can improve cost-aware
> retrieval-stage RAG performance over strong fixed-action baselines on SciFact
> and NFCorpus, with bootstrap, selected-action replay, constrained utility, and
> OPE diagnostics.

Not defensible from the current evidence:

- full RAG answer-generation improvement;
- online deployment or real logged user feedback;
- LLM fine-tuning;
- production-ready semantic-feature routing;
- a new general RL-RAG architecture.
