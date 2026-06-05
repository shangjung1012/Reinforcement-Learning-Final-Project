# Final Defense Q&A

## Is This Really Reinforcement Learning?

Short answer:

Yes, but specifically offline contextual bandits, not online deep RL.

Answer:

Each query is a context, each retrieval or rewrite option is an action, and the
objective is expected retrieval reward minus cost. The direct-method policy
learns action-value predictors from replayable offline rewards, then chooses the
action with the highest predicted reward. The main table reports the best
selected-action replay baseline, chosen from LinUCB, epsilon-greedy, and linear
Thompson sampling, where the learner observes only the reward of the action it
chose during replay.

Phrase to use:

> We formulate the problem as an offline contextual bandit because deployment
> interaction is unavailable and the retrieval environment can be replayed
> offline.

## Is This Just Supervised Learning?

Short answer:

The estimator has supervised components, but the objective and evaluation are
bandit policy optimization.

Answer:

The direct method uses supervised reward prediction internally, but the target
is not a class label. The target is the expected reward of each action under a
policy. The final decision is an argmax over predicted action values, and the
policy is evaluated by realized retrieval reward, retrieval calls, cost,
bootstrap uncertainty, and OPE diagnostics.

Phrase to use:

> Direct-method contextual bandits often use supervised reward models, but they
> are optimizing policy value, not predicting a human label.

## Are The Baselines Too Weak?

Short answer:

No. The main comparison is not only vanilla BM25.

Answer:

The project compares against BM25, dense retrieval, hybrid retrieval,
train-best fixed retrieval action, heuristic adaptive routing, the best
selected-action replay baseline, budget curves, constrained fixed-action
comparisons, and oracle upper bounds. The main reported gain is over the
train-best fixed action selected from BM25, dense, and hybrid actions.

Evidence:

- `outputs/results/final_main_results_table.csv`
- `outputs/figures/final_linucb_comparison.png`

## What Does Selected-Action Replay Show?

Answer:

It separates two learning regimes that can otherwise sound identical. The
direct-method policy observes every action reward for each replayed query, while
LinUCB, epsilon-greedy, and Thompson replay observe only the action they choose
at each step. The replay-regret curves therefore measure the extra difficulty of
bandit feedback under the same action table.

Numbers:

- SciFact: LinUCB selected-action replay cumulative regret is 82.080 versus
  67.427 for the full-information direct method.
- NFCorpus: LinUCB selected-action replay cumulative regret is 43.202 versus
  29.617 for the full-information direct method.

Evidence:

- `outputs/results/scifact_bandit_replay_summary.csv`
- `outputs/results/nfcorpus_bandit_replay_summary.csv`
- `outputs/figures/scifact_bandit_replay_regret.png`
- `outputs/figures/nfcorpus_bandit_replay_regret.png`

## What Is The Main Result?

Answer:

The selective policy improves reward over the train-best fixed retrieval action
on both full-corpus BEIR checks.

Numbers:

- SciFact: +0.033711 reward, CI [0.008852, 0.058341].
- NFCorpus: +0.029942 reward, CI [0.005428, 0.054799].

Evidence:

- `outputs/results/final_claims_matrix.csv`
- `outputs/figures/final_reward_delta_ci.png`

## Is The Result Stable Across Seeds?

Answer:

The main table uses the primary seed-42 full-corpus run, and I added a
three-seed robustness check with 300 train and 150 test examples per seed. The
policy does not win every seed, but the average deltas remain positive on both
datasets.

Numbers:

- SciFact: selective mean delta +0.021304; constrained mean delta +0.025856.
- NFCorpus: selective mean delta +0.024816; constrained mean delta +0.031224.

Defense point:

This should be presented as supporting evidence, not as a stronger benchmark
than the main 600/300 seed-42 table.

Evidence:

- `outputs/repeated_main_runs/results/repeated_main_robustness_aggregate.csv`
- `outputs/repeated_main_runs/results/repeated_main_robustness_per_seed.csv`

## Why Use A Cost-Aware Reward?

Answer:

Retrieval quality alone would always favor more expensive actions such as
hybrid retrieval or LLM rewrite experts. RAG systems care about both evidence
quality and cost, so the reward includes recall, MRR, rewrite cost, and extra
retrieval-call cost.

Formula:

```text
Recall@5 + 0.5 * MRR - RewriteCost - ExtraRetrievalCallCost
```

Defense point:

The constrained sweep makes this even more explicit by retraining under
different retrieval-call penalties.

## What Does The Constrained Policy Add?

Answer:

The constrained policy studies a Lagrangian version of the same problem:
maximize retrieval utility while penalizing extra calls. This turns cost from a
side metric into a policy-training objective.

Numbers:

- SciFact lambda=0.03 utility delta: +0.047350, CI [0.014315, 0.082247].
- NFCorpus lambda=0.03 utility delta: +0.032656, CI [0.010587, 0.058431].

Evidence:

- `outputs/figures/final_cost_reward_frontier.png`
- `outputs/results/scifact_constrained_policy_bootstrap.csv`
- `outputs/results/nfcorpus_constrained_policy_bootstrap.csv`

## Why Include OPE?

Answer:

Offline RL and contextual bandits need a way to reason about policy value from
logged feedback. The OPE diagnostics compare direct method, IPS, SNIPS, and
doubly robust estimates against known full-information values. This shows how
behavior-policy coverage affects estimator reliability.

Numbers:

- SciFact IPS error rises from 0.127173 under uniform logging to 0.267959 under
  sparse train-best-epsilon logging.
- NFCorpus doubly robust error is 0.038419 versus 0.052150 for direct method
  under uniform logging.

Evidence:

- `outputs/figures/final_ope_estimator_error.png`

## How Is This Different From MBA-RAG Or Adaptive-RAG?

Answer:

Adaptive-RAG routes strategies by question complexity, and MBA-RAG frames
retrieval strategy selection as a bandit. This project is close to that family
but narrower and more reproducible: it focuses on retrieval-stage rewrite and
retrieval actions, explicit reward/cost accounting, selected-action replay
baselines, OPE diagnostics, constrained utility, bootstrap intervals, and
artifact traceability.

Phrase to use:

> The novelty is not claiming a new RAG paradigm; it is an auditable,
> course-feasible retrieval-stage contextual-bandit study with cost and
> uncertainty controls.

## Why Not Fine-Tune An LLM?

Answer:

Fine-tuning a reader or generator would change the scope and require much more
compute. The project intentionally isolates the retrieval stage to ask whether a
lightweight policy can choose retrieval or rewrite actions efficiently. Vertex
Gemini is used as an optional expensive action, not as a model to train.

## Did The Policy Actually Train?

Answer:

Yes. The policy trains action-value models from offline reward tables. The
project also saves checkpoints and reports model classes, action spaces, feature
widths, and selected policy families in the checkpoint manifest.

Evidence:

- `outputs/results/final_checkpoint_manifest.csv`
- `outputs/checkpoints/scifact_retrieval_policy.pkl`
- `outputs/checkpoints/nfcorpus_retrieval_policy.pkl`

## Why Did Semantic Embeddings Not Become The Main Result?

Answer:

The semantic embedding extension was implemented and tested with Vertex cache
preflights, feature ablations, and repeated selection diagnostics. The evidence
was not stable enough to claim deployment improvement, so the final report marks
semantic features as analysis-only. This avoids overclaiming.

Phrase to use:

> Semantic features are implemented, but the validation evidence was not strong
> enough to replace the stronger non-semantic retrieval-policy checkpoint.

## What Is The Biggest Limitation?

Answer:

The project evaluates evidence retrieval, not end-to-end generated answers. It
does not include a neural reader or answer EM/F1. Also, OPE is simulated from
full-information retrieval tables rather than real production logs.

## What Is The Strongest Final Claim?

Answer:

> A low-compute offline contextual-bandit formulation can improve cost-aware
> retrieval-stage RAG performance over strong fixed-action baselines on SciFact
> and NFCorpus, and the improvement is supported by bootstrap intervals,
> selected-action bandit replay, constrained utility analysis, and OPE
> diagnostics.

## What Should Not Be Claimed?

Do not claim:

- This is a new full RL-RAG architecture.
- It fine-tunes an LLM.
- It improves generated-answer EM/F1.
- Semantic embeddings are proven to improve the final deployed policy.
- OPE results are from real online logs.
