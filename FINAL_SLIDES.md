# Final Slides

## Slide 1: Title

Title:

Selective and Cost-Aware Offline Contextual Bandits for RAG Retrieval Routing

Message:

This project studies retrieval-stage control in RAG: when a query arrives, a
lightweight policy chooses the retrieval or rewrite action that maximizes
evidence retrieval reward under explicit cost.

Use:

- One-sentence thesis from `FINAL_RESULTS_SUMMARY.md`.

## Slide 2: Problem

Message:

Fixed RAG pipelines use the same retrieval strategy for every query. That is
wasteful because some queries are easy enough for one cheap retrieval call,
while others may benefit from dense retrieval, hybrid retrieval, or query
rewriting.

Key point:

The decision problem is not just "which retriever is best on average"; it is
"which action should this query use under a cost-aware reward."

## Slide 3: Related Work And Positioning

Message:

The project is related to RL query rewriting, adaptive retrieval, and heavier
RL-RAG systems.

Positioning:

- CONQRR / Rewrite-Retrieve-Read / MaFeRw: retrieval-oriented query rewriting.
- Adaptive-RAG / MBA-RAG: route retrieval strategy by query complexity.
- Self-RAG / R3-RAG / RAG-RL: heavier reader or generator-side training.
- This project: lightweight, auditable retrieval-stage offline contextual
  bandits.

Defense point:

Do not claim full novelty over these papers. Claim low-compute, reproducible,
cost-aware retrieval-stage analysis.

## Slide 4: Contextual-Bandit Formulation

Message:

Each query is one contextual-bandit decision.

State:

- Query features.
- BM25 confidence features.
- Retrieval score shape and overlap diagnostics.
- Optional semantic features in pilot experiments.

Actions:

- BM25 keep / keyword.
- Dense keep / keyword.
- Hybrid keep / keyword.
- Optional Gemini rewrite / decomposition experts.

Reward:

```text
Recall@5 + 0.5 * MRR - RewriteCost - ExtraRetrievalCallCost
```

## Slide 5: Main BEIR Results

Figure:

- `outputs/figures/final_reward_delta_ci.png`

Table:

- `outputs/results/final_main_results_table.csv`

Message:

The learned selective policy improves reward over the train-best fixed action
on both BEIR datasets.

Numbers:

- SciFact: +0.033711 reward, CI [0.008852, 0.058341].
- NFCorpus: +0.029942 reward, CI [0.005428, 0.054799].

Defense point:

The baseline is not vanilla BM25. The comparison is against the train-best fixed
retrieval action selected from BM25, dense, and hybrid actions.

## Slide 6: Stronger Baselines

Figure:

- `outputs/figures/final_linucb_comparison.png`

Message:

The main table uses one selected-action bandit row: the best policy among
LinUCB, epsilon-greedy, and linear Thompson replay.

Numbers:

- SciFact best selected-action bandit: 1.062139 reward versus 1.056778
  train-best, with 1.033333 calls.
- NFCorpus best selected-action bandit: 0.404025 reward versus 0.377153
  train-best, with 1.136667 calls.

Defense point:

The selected-action bandit baseline is for RL framing. The direct-method policy
is still the main method because it can use replayable full-information action
rewards.

## Slide 7: Cost-Aware Constrained Policy

Figure:

- `outputs/figures/final_cost_reward_frontier.png`

Message:

The constrained policy retrains under Lagrangian call penalties and produces a
reward/call frontier.

Numbers:

- SciFact lambda=0.03 utility delta: +0.047350, CI [0.014315, 0.082247].
- NFCorpus lambda=0.03 utility delta: +0.032656, CI [0.010587, 0.058431].

Defense point:

This makes the cost objective explicit instead of treating cost as an informal
side metric.

## Slide 8: Off-Policy Evaluation

Figure:

- `outputs/figures/final_ope_estimator_error.png`

Message:

Offline contextual-bandit work should discuss how policy value would be
estimated from logged feedback.

Numbers:

- SciFact IPS error under sparse train-best-epsilon logging: 0.267959.
- SciFact IPS error under uniform logging: 0.127173.
- NFCorpus doubly robust uniform-log error: 0.038419 versus 0.052150 for direct
  method.

Defense point:

OPE is not used to inflate the main result. It is a diagnostic that exposes
coverage and estimator tradeoffs.

## Slide 9: Repeated-Seed Robustness

Message:

The seed-42 table is the main result, but repeated full-corpus runs check that
the conclusion is not purely a single random split artifact.

Numbers:

- SciFact, 3 seeds: selective policy mean delta +0.021304, constrained mean
  delta +0.025856.
- NFCorpus, 3 seeds: selective policy mean delta +0.024816, constrained mean
  delta +0.031224.
- Best selected-action bandit has positive mean delta on both datasets.

Defense point:

The repeated-seed table is supporting evidence, not a new main benchmark. It
shows positive mean gains with some seed-level failures, which is the honest
claim for a lightweight offline contextual-bandit policy.

## Slide 10: Semantic And LLM Extensions

Message:

The project supports Vertex Gemini rewrite/decomposition actions and optional
Vertex embedding features, but the final claim is conservative.

What happened:

- Gemini actions are expensive experts with explicit costs.
- Semantic features were evaluated with cache-aware pilots and repeated
  selection diagnostics.
- The deployment decision keeps semantic features analysis-only until stronger
  validation stability is available.

Defense point:

This is a strength, not a weakness: the project avoids overclaiming unstable
semantic-feature gains.

## Slide 11: Limitations

Message:

The project is retrieval-stage RL, not full RAG answer generation.

Limitations:

- No reader model and no answer EM/F1.
- Offline replay rather than real online logs.
- Direct-method contextual bandit can look supervised, so the bandit framing
  must be explained clearly.
- Semantic features are not final deployment evidence.

## Slide 12: Takeaway

Message:

The contribution is a reproducible, low-compute framework for studying
cost-aware retrieval routing as offline contextual bandits.

Final line:

The value of the project is not training a large model; it is formalizing RAG
retrieval control as an auditable decision problem with costs, baselines,
uncertainty, OPE, and constrained policy analysis.

## Slide 13: Backup Evidence

Files:

- `outputs/results/final_main_results_table.csv`
- `outputs/results/final_claims_matrix.csv`
- `outputs/results/final_artifact_index.csv`
- `outputs/results/final_checkpoint_manifest.csv`
- `outputs/results/scifact_ope_stability.csv`
- `outputs/results/nfcorpus_ope_stability.csv`

Use this slide only if asked for reproducibility or artifact traceability.
