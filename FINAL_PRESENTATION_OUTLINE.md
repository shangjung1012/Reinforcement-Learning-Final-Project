# Final Presentation Outline

## Thesis

This project studies lightweight offline contextual-bandit control for
cost-aware RAG retrieval routing. The contribution is not LLM fine-tuning; it is
an auditable retrieval-stage RL formulation with explicit action costs,
selected-action baselines, OPE diagnostics, constrained utility sweeps, and
evidence-linked artifacts.

## Slide Plan

1. Title and one-sentence framing
   - Lightweight offline contextual bandits for cost-aware retrieval routing and
     query rewriting in RAG.

2. Problem
   - Fixed RAG retrieval wastes calls and cannot adapt to query difficulty.
   - Query rewriting and hybrid retrieval can help, but should be used only when
     the expected retrieval reward justifies the cost.

3. Related work positioning
   - Query rewriting RL: CONQRR, Rewrite-Retrieve-Read, MaFeRw.
   - Adaptive retrieval: Adaptive-RAG, MBA-RAG.
   - Heavier RL-RAG: Self-RAG, R3-RAG, RAG-RL.
   - This project focuses on reproducible retrieval-stage control rather than
     training a reader or fine-tuning an LLM.

4. Formulation
   - State: query features plus shallow retrieval-confidence statistics.
   - Action: BM25, dense, hybrid, keyword rewrite, and optional LLM rewrite /
     decomposition experts.
   - Reward: Recall@5 + 0.5 * MRR - rewrite cost - extra retrieval-call cost.
   - Policy: offline direct-method contextual bandit with cross-validated reward
     predictors.

5. Main BEIR results
   - SciFact: selective policy reward 1.090489 versus train-best 1.056778;
     delta +0.033711, CI [0.008852, 0.058341].
   - NFCorpus: selective policy reward 0.407095 versus train-best 0.377153;
     delta +0.029942, CI [0.005428, 0.054799].
   - Evidence: `outputs/results/final_main_results_table.csv` and
     `outputs/figures/final_reward_delta_ci.png`.

6. Why this is RL / bandit, not only supervised learning
   - The decision objective is expected reward under a policy.
   - The best selected-action bandit baseline uses replay feedback and observes
     only the chosen action reward.
   - Direct method is the low-variance full-information offline estimator used
     because retrieval actions can be replayed without online deployment.
   - Figure: `outputs/figures/final_linucb_comparison.png`.

7. Cost-aware constrained result
   - Lagrangian sweep retrains policy under call penalties.
   - SciFact lambda=0.03 utility delta: +0.047350, CI [0.014315, 0.082247].
   - NFCorpus lambda=0.03 utility delta: +0.032656, CI [0.010587, 0.058431].
   - Figure: `outputs/figures/final_cost_reward_frontier.png`.

8. OPE diagnostics
   - DM, IPS, SNIPS, and doubly robust estimates are compared against known
     full-information values.
   - SciFact IPS error rises from 0.127173 under uniform logging to 0.267959
     under sparse train-best-epsilon logging.
   - NFCorpus doubly robust improves uniform-log absolute error to 0.038419
     relative to 0.052150 for direct method.
   - Figure: `outputs/figures/final_ope_estimator_error.png`.

9. Repeated-seed robustness
   - Three-seed full-corpus runs keep the same retrieval-action setup but use
     300 train and 150 test examples per seed.
   - SciFact: selective mean delta +0.021304; constrained mean delta +0.025856.
   - NFCorpus: selective mean delta +0.024816; constrained mean delta
     +0.031224.
   - Evidence:
     `outputs/repeated_main_runs/results/repeated_main_robustness_aggregate.csv`.

10. What fails or remains uncertain
   - Semantic embedding features are analysis-only in the current evidence
     because validation stability is weak on small cached Vertex runs.
   - The selected-action baseline validates the bandit framing but does not beat
     the full-information direct-method policy.
   - No reader/answer EM is included, so claims are limited to evidence
     retrieval.

11. Takeaway
   - The project is a reproducible, low-compute study of cost-aware retrieval
     routing as offline contextual bandits.
   - The main evidence is not a single score; it is the combination of main
     reward deltas, best selected-action bandit baseline, constrained utility
     frontiers, OPE diagnostics, repeated-seed robustness, and bootstrap
     confidence intervals.

## Defense Questions

### Is this really RL?

Yes, framed as offline contextual bandits. Each query is a state, each retrieval
or rewrite choice is an action, and the objective is expected retrieval reward
minus cost. The direct-method policy is a stable offline estimator, and the best
selected-action replay baseline validates the bandit framing.

### Why not fine-tune an LLM?

The project intentionally isolates retrieval-stage control. Fine-tuning a reader
or generator would change a different part of the RAG stack and require much
larger compute. The point here is to test whether a lightweight policy can make
auditable retrieval decisions.

### How is this different from MBA-RAG?

MBA-RAG also uses bandit-style adaptive retrieval. This project is narrower and
more reproducible: it focuses on retrieval-stage rewrite/routing actions,
explicit cost penalties, OPE diagnostics, constrained sweeps, and artifact-level
evidence tracking across small BEIR benchmarks.

### Is the baseline too weak?

The project includes fixed BM25, dense, hybrid, train-best fixed action,
heuristic adaptive routing, selected-action replay baselines, oracle upper
bounds, constrained fixed-action comparisons, and budget curves. The main table
reports the best selected-action baseline to keep the comparison compact.

### What should not be overclaimed?

Do not claim a novel RL-RAG architecture or answer-generation improvement. The
defensible claim is that a low-compute offline contextual-bandit formulation can
improve retrieval-stage reward and expose cost/coverage tradeoffs in an
auditable way.
