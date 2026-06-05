# Final Project Report

## Title

Selective and Cost-Aware RL Query Rewriting for Retrieval-Augmented Generation

## Goal

The original proposal targeted a RAG system that learns when and how to rewrite
queries. The final implementation focuses on the retrieval stage: the reader is
held out of scope, and policies are evaluated by whether they retrieve the gold
evidence efficiently.

## Method

Each question is treated as a contextual-bandit decision. The policy observes
query features and shallow BM25 retrieval statistics, then chooses one rewrite
action:

- `keep`
- `keyword_compress`
- `entity_expand`
- `decompose`

The reward is:

```text
Recall@5 + 0.5 * MRR - RewriteCost
```

The retrieval-action extension keeps the same offline contextual-bandit setup
but expands the action space to choose both a rewrite and a retriever:

- `bm25_keep`
- `bm25_keyword`
- `dense_keep`
- `dense_keyword`
- `hybrid_keep`
- `hybrid_keyword`
- `bm25_llm_rewrite`
- `bm25_llm_decompose`

For this setting, the cost term also includes a small penalty for each extra
retrieval call beyond the first:

```text
Recall@5 + 0.5 * MRR - RewriteCost - ExtraRetrievalCallCost
```

The multi-step extension adds feedback-aware actions and trains a two-step fitted
Q policy:

- `feedback_expand`
- `title_bridge`
- `stop`

Training is lightweight offline RL rather than LLM fine-tuning. The one-step
policy is a direct-method contextual bandit: for each action, it learns a reward
predictor from the observed state features, then selects the action with the
highest predicted reward. This is stronger than a hand-coded keyword heuristic
because the policy can consume retrieval-confidence and semantic features
directly. The implementation supports KNN, ridge regression, ExtraTrees,
RandomForest, and an MLP neural reward predictor, with an `auto` mode that
chooses among them by cross-validation. The two-step policy trains fitted-Q
estimators for the first and second decision steps. The trained policies are
saved as checkpoints under `outputs/checkpoints/`.

For the full-corpus BEIR retrieval-action experiments, the report also includes
a stronger non-learned adaptive baseline called `Heuristic retrieval router`.
It uses only deployment-time state features, specifically query length plus BM25
top-score, score-gap, and entropy statistics, to choose between BM25, dense, and
hybrid actions. This separates the value of adaptive routing itself from the
value of learning the routing policy from offline rewards.

Relative to related RAG work, this project is closest to adaptive retrieval and
query-rewriting systems: Self-RAG and FLARE motivate deciding when retrieval is
needed, CONQRR and later RL query-rewriting work optimize rewritten queries with
retrieval rewards, and MBA-RAG frames retrieval-strategy selection as a bandit
problem. The implementation here is narrower but auditable: it does not train a
reader or fine-tune an LLM, and instead studies whether a learned, cost-aware
router can beat fixed and heuristic retrieval actions on BEIR evidence
retrieval.

As an optional extension, the retrieval-action policy can augment the state with
Vertex AI `gemini-embedding-001` semantic confidence features. These features
compare the query embedding with the initial BM25 top passages and record
similarity rank-shape statistics such as top similarity, mean similarity,
dispersion, and top-to-bottom spread, plus a top-k rank-aware similarity profile
that preserves the BM25-order similarity shape. The current implementation also
adds four semantic score-shape features comparing the head and tail of the
similarity list, then appends six BM25/semantic rank-agreement signals, compact
random projections of the query embedding, top-passage centroid, and
query-centroid delta, plus six lexical-semantic interaction features that
combine BM25 top-score, score gap, and entropy with semantic similarity margins
and spread. They are cached locally under `outputs/cache/`. They are not used
for the main HotpotQA table because the first full-split probe improved over
the fixed hybrid baseline only slightly and underperformed the current
non-semantic policy.

The LLM-action extension uses Vertex Gemini as an expensive rewrite expert. The
policy can choose cached Gemini rewrite/decomposition actions, but those actions
receive an explicit cost of `1.0 + 0.01 * rewritten_tokens` so the learned policy
must decide when the API call is worth it.

For deployment-style evaluation, the retrieval-action policy also exposes the
predicted reward score for each action. A confidence-gated variant compares the
top predicted action score to the runner-up score and falls back to the
train-best fixed action when that margin is below a configured threshold. This
gate is based only on model predictions available before evaluation, not on
held-out realized rewards. A separate threshold-sweep diagnostic replays the
detailed policy output over multiple margin values to measure fallback rate,
reward delta, and retrieval-call delta without rerunning retrieval.

## Datasets

- HotpotQA distractor dev set for multi-hop evidence retrieval.
- Natural Questions validation shard for single-hop title retrieval.
- BEIR SciFact for scientific-claim evidence retrieval.
- BEIR NFCorpus for biomedical and nutrition evidence retrieval.

For Natural Questions, each example uses the original document as the gold
passage and samples other documents from the same shard as negatives.

For SciFact, the main experiment uses the official BEIR train/test qrels and
evaluates each query against the full 5,183-document SciFact corpus.

For NFCorpus, the main experiment uses the official BEIR train/test qrels and
evaluates each query against the full 3,633-document collection.

## Main Results

### HotpotQA One-Step Retrieval

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla BM25 | 0.800 | 0.872 | 0.753 | 1.236 | 0.000 | 1.000 |
| Rewrite-all keyword | 0.833 | 0.880 | 0.780 | 1.272 | 0.000 | 1.000 |
| Rewrite-all entity expansion | 0.810 | 0.882 | 0.767 | 1.202 | 0.049 | 1.000 |
| Selective bandit | 0.812 | 0.880 | 0.766 | 1.244 | 0.008 | 1.035 |
| Oracle best action | 0.884 | 0.923 | 0.821 | 1.338 | 0.007 | 1.045 |

### HotpotQA Ablation

| Variant | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selective bandit | 0.812 | 0.880 | 0.766 | 1.244 | 0.008 | 1.035 |
| No keep action | 0.829 | 0.880 | 0.778 | 1.258 | 0.010 | 1.040 |
| No cost penalty | 0.805 | 0.880 | 0.762 | 1.206 | 0.039 | 1.188 |
| Retrieval-only reward | 0.804 | 0.875 | 0.758 | 1.213 | 0.029 | 1.153 |

### HotpotQA Two-Step Retrieval

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla BM25 | 0.785 | 0.878 | 0.744 | 1.225 | 0.000 | 1.000 |
| Rewrite-all keyword | 0.823 | 0.923 | 0.790 | 1.264 | 0.020 | 2.000 |
| Multi-step FQI | 0.815 | 0.888 | 0.769 | 1.223 | 0.036 | 2.438 |
| Oracle two-step | 0.912 | 0.959 | 0.854 | 1.376 | 0.016 | 1.571 |

The two-step FQI extension should be interpreted as a sequential-RL diagnostic,
not as the final strongest result. It exposes stop/refine actions and fitted-Q
training, but the current lightweight KNN value model underperforms the
train-best fixed trace in both Recall@5 and reward. The large two-step oracle
gap is the useful signal: the action space has headroom, but the current state
features and value estimator are not yet strong enough to exploit it reliably.

### Vertex AI Gemini Rewrite Baseline

This baseline uses Gemini through Vertex AI as a strong rewrite-all system. The
results below are from a small API-controlled sample: 20 HotpotQA examples, 8
held-out test questions, with rewrites cached in `outputs/cache/`.

| Method | Recall@5 | MRR | nDCG@5 | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini rewrite-all | 0.750 | 0.906 | 0.743 | 1.060 | 1.000 |
| Gemini decompose | 0.875 | 0.906 | 0.840 | 1.089 | 2.000 |

### Selective Gemini Retrieval-Action Policy

This experiment uses the same 20-example HotpotQA API-controlled sample as the
Gemini baseline, but makes Gemini rewrite/decomposition available as optional
policy actions alongside BM25, dense, and hybrid retrieval. The `auto` policy
selected a ridge direct-method reward predictor (`ridge_l2=1.0`) by 3-fold
validation.

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla BM25 | 0.750 | 0.906 | 0.753 | 1.203 | 0.000 | 1.000 |
| BM25 keyword | 0.750 | 1.000 | 0.797 | 1.250 | 0.000 | 1.000 |
| Dense original | 0.812 | 0.938 | 0.781 | 1.281 | 0.000 | 1.000 |
| Dense keyword | 0.875 | 0.938 | 0.824 | 1.344 | 0.000 | 1.000 |
| Hybrid original | 0.938 | 1.000 | 0.875 | 1.408 | 0.030 | 2.000 |
| Hybrid keyword | 0.875 | 1.000 | 0.849 | 1.345 | 0.030 | 2.000 |
| Gemini rewrite action | 0.750 | 0.906 | 0.743 | 0.143 | 1.060 | 1.000 |
| Gemini decompose action | 0.875 | 0.906 | 0.840 | 0.209 | 1.119 | 2.000 |
| Train-best retrieval action | 0.875 | 0.938 | 0.824 | 1.344 | 0.000 | 1.000 |
| Selective retrieval policy | 0.938 | 0.938 | 0.877 | 1.399 | 0.008 | 1.250 |
| Oracle retrieval action | 0.938 | 1.000 | 0.906 | 1.434 | 0.004 | 1.125 |

### Dense and Hybrid Retrieval Baseline

This baseline uses `sentence-transformers/all-MiniLM-L6-v2` for dense retrieval
and a reciprocal-rank-style merge for BM25+dense hybrid retrieval. The run uses
300 HotpotQA examples with 120 held-out test questions, so it is a stronger
retriever comparison but not a directly identical split to the one-step BM25/RL
table above.

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense original | 0.833 | 0.839 | 0.758 | 1.253 | 0.000 | 1.000 |
| Dense keyword | 0.812 | 0.849 | 0.747 | 1.237 | 0.000 | 1.000 |
| Hybrid original | 0.863 | 0.870 | 0.790 | 1.297 | 0.000 | 2.000 |
| Hybrid keyword | 0.887 | 0.887 | 0.813 | 1.331 | 0.000 | 2.000 |

### Cost-Aware Retrieval-Action Policy

This experiment uses the same 300-example HotpotQA sample and 120 held-out test
questions as the dense/hybrid baseline, but trains a contextual bandit to select
among BM25, dense, and hybrid retrieval actions. Hybrid retrieval receives a
0.03 cost penalty because it performs both BM25 and dense retrieval. The KNN
policy uses 5-fold cross-validation on the training split to choose `k=5`.

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla BM25 | 0.804 | 0.851 | 0.742 | 1.229 | 0.000 | 1.000 |
| BM25 keyword | 0.829 | 0.896 | 0.780 | 1.277 | 0.000 | 1.000 |
| Dense original | 0.833 | 0.839 | 0.758 | 1.253 | 0.000 | 1.000 |
| Dense keyword | 0.812 | 0.849 | 0.747 | 1.237 | 0.000 | 1.000 |
| Hybrid original | 0.863 | 0.870 | 0.790 | 1.268 | 0.030 | 2.000 |
| Hybrid keyword | 0.887 | 0.887 | 0.813 | 1.301 | 0.030 | 2.000 |
| Train-best retrieval action | 0.887 | 0.887 | 0.813 | 1.301 | 0.030 | 2.000 |
| Selective retrieval policy | 0.879 | 0.896 | 0.813 | 1.320 | 0.007 | 1.242 |
| Oracle retrieval action | 0.950 | 0.942 | 0.868 | 1.420 | 0.001 | 1.033 |

### BEIR SciFact Retrieval-Action Policy

This cross-dataset check trains on 600 SciFact train-qrels examples and evaluates
on 300 test-qrels examples against the full 5,183-document corpus. The `auto`
policy uses 5-fold validation and selects an ExtraTrees direct-method reward
predictor.

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla BM25 | 0.727 | 0.613 | 0.635 | 1.033 | 0.000 | 1.000 |
| BM25 keyword | 0.728 | 0.628 | 0.646 | 1.042 | 0.000 | 1.000 |
| Dense original | 0.741 | 0.601 | 0.632 | 1.042 | 0.000 | 1.000 |
| Dense keyword | 0.724 | 0.602 | 0.629 | 1.025 | 0.000 | 1.000 |
| Hybrid original | 0.762 | 0.647 | 0.671 | 1.056 | 0.030 | 2.000 |
| Hybrid keyword | 0.763 | 0.648 | 0.673 | 1.057 | 0.030 | 2.000 |
| Train-best retrieval action | 0.763 | 0.648 | 0.673 | 1.057 | 0.030 | 2.000 |
| Heuristic retrieval router | 0.733 | 0.632 | 0.651 | 1.048 | 0.002 | 1.053 |
| Selective retrieval policy | 0.774 | 0.658 | 0.681 | 1.090 | 0.012 | 1.413 |
| Confidence-gated retrieval policy | 0.774 | 0.658 | 0.681 | 1.090 | 0.012 | 1.413 |
| Oracle retrieval action | 0.839 | 0.736 | 0.758 | 1.206 | 0.000 | 1.007 |

### BEIR NFCorpus Retrieval-Action Policy

This cross-domain check trains on 600 NFCorpus train-qrels examples and
evaluates on 300 test-qrels examples against the full 3,633-document corpus. The
`auto` policy uses 5-fold validation and selects a ridge direct-method reward
predictor (`ridge_l2=1.0`).

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla BM25 | 0.118 | 0.509 | 0.344 | 0.373 | 0.000 | 1.000 |
| BM25 keyword | 0.124 | 0.523 | 0.354 | 0.385 | 0.000 | 1.000 |
| Dense original | 0.122 | 0.510 | 0.358 | 0.377 | 0.000 | 1.000 |
| Dense keyword | 0.119 | 0.497 | 0.350 | 0.367 | 0.000 | 1.000 |
| Hybrid original | 0.131 | 0.530 | 0.364 | 0.366 | 0.030 | 2.000 |
| Hybrid keyword | 0.133 | 0.536 | 0.368 | 0.371 | 0.030 | 2.000 |
| Train-best retrieval action | 0.122 | 0.510 | 0.358 | 0.377 | 0.000 | 1.000 |
| Heuristic retrieval router | 0.129 | 0.521 | 0.352 | 0.390 | 0.000 | 1.000 |
| Selective retrieval policy | 0.136 | 0.543 | 0.372 | 0.407 | 0.000 | 1.000 |
| Confidence-gated retrieval policy | 0.136 | 0.543 | 0.372 | 0.407 | 0.000 | 1.000 |
| Oracle retrieval action | 0.157 | 0.616 | 0.439 | 0.464 | 0.001 | 1.020 |

### Confidence-Gate Sweep

The full-corpus confidence-gate sweeps show that the margin threshold should be
very small if it is used at all. On NFCorpus, margin `0.0` keeps the best reward
at 0.407. A threshold of `0.001` still beats train-best, 0.400 versus 0.377, but
loses 0.007 reward compared with the ungated policy; thresholds at or above
`0.01` mostly collapse back to train-best behavior. On SciFact, threshold
`0.001` slightly improves reward from 1.090 to 1.091 with a 7.0% fallback rate,
but larger thresholds reduce reward. Thus confidence gating is useful as a
diagnostic and possible very-low-threshold guardrail, not as a high-threshold
fallback policy.

### Repeated-Seed Robustness

The seed-42 full-corpus BEIR experiments are still the main results, but I also
reran the retrieval-action setup over three seeds with 300 training and 150 test
examples per seed. This checks whether the learned policy gains are stable
enough to defend as a contextual-bandit result rather than a single-split
accident. The run keeps the same action space and full-corpus evaluation, and
compares the direct-method selective policy, the lambda=0.03 constrained policy,
and the best selected-action bandit baseline from LinUCB, epsilon-greedy, and
linear Thompson replay.

| Dataset | Seeds | Selective Win Rate | Selective Mean Delta | Selective Delta Std | Constrained Win Rate | Constrained Mean Delta | Best Bandit Win Rate | Best Bandit Mean Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 3 | 0.667 | +0.021 | 0.037 | 0.667 | +0.026 | 1.000 | +0.016 |
| NFCorpus | 3 | 0.667 | +0.025 | 0.025 | 1.000 | +0.031 | 0.667 | +0.004 |

The repeated-seed result is deliberately reported as supporting evidence rather
than a new main benchmark. It shows that the average gains remain positive
across both datasets, but also that the lightweight policy can lose on an
individual seed. That is the right final claim: the project has a real offline
RL decision problem with positive repeated-run evidence, but it should not be
presented as a large-model RL-RAG breakthrough.

### Policy Behavior Diagnostics

The final full-corpus BEIR runs also write action-distribution diagnostics for
the selective policy. These diagnostics check whether the policy is actually
choosing different retrieval/rewrite actions rather than collapsing to one fixed
baseline.

| Dataset | Action | Test Count | Test Share |
| --- | --- | ---: | ---: |
| SciFact | `bm25_keep` | 53 | 17.7% |
| SciFact | `bm25_keyword` | 54 | 18.0% |
| SciFact | `dense_keep` | 60 | 20.0% |
| SciFact | `dense_keyword` | 9 | 3.0% |
| SciFact | `hybrid_keep` | 40 | 13.3% |
| SciFact | `hybrid_keyword` | 84 | 28.0% |
| NFCorpus | `bm25_keep` | 98 | 32.7% |
| NFCorpus | `bm25_keyword` | 36 | 12.0% |
| NFCorpus | `dense_keep` | 108 | 36.0% |
| NFCorpus | `dense_keyword` | 58 | 19.3% |

### Query Difficulty Bucket Diagnostics

The detailed BEIR outputs now include deployable difficulty features: query
length, BM25 top score, BM25 score gap, BM25 entropy, predicted action margin,
BM25/dense overlap, dense/hybrid new-document rates, plus oracle-side reward
margin diagnostics for analysis. The bucket exporter groups held-out queries
into low/mid/high bins and recomputes each method's reward inside each bin.

| Dataset | Feature Bucket | Queries | Policy Reward | Delta vs Train-Best | Delta vs Heuristic | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | `state_bm25_gap=low` | 100 | 0.838 | +0.054 | +0.091 | 1.390 |
| SciFact | `action_reward_std=high` | 144 | 1.111 | +0.049 | +0.098 | 1.493 |
| SciFact | `state_question_length=low` | 100 | 1.098 | +0.029 | +0.069 | 1.560 |
| NFCorpus | `action_reward_std=high` | 100 | 0.514 | +0.096 | +0.049 | 1.000 |
| NFCorpus | `state_bm25_top1=mid` | 100 | 0.488 | +0.042 | +0.030 | 1.000 |
| NFCorpus | `state_question_length=mid` | 49 | 0.316 | -0.015 | -0.013 | 1.000 |

This makes the adaptive-RAG result less dependent on a single aggregate mean.
On SciFact, the largest gains occur when BM25 confidence is ambiguous or when
candidate actions have high reward dispersion, but the policy pays for extra
hybrid calls in those bins. On NFCorpus, the policy's strongest buckets improve
reward without extra retrieval calls, while the short mid-length subset remains
a failure case. The diagnostics therefore support the claim that the learned
router is conditional on query/retrieval difficulty, not just a global
preference for one retrieval method.

### Cross-Dataset Policy Transfer

The transfer experiment trains the same direct-method policy on one BEIR source
dataset and evaluates it on either the same target domain or the other domain.
Each target is compared against the target-domain train-best fixed action and
the deployable heuristic router.

| Source | Target | Policy Reward | Target Train-Best | Heuristic | Recall@5 | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | SciFact | 1.093 | 1.057 | 1.048 | 0.776 | 1.477 |
| SciFact | NFCorpus | 0.386 | 0.377 | 0.390 | 0.137 | 1.597 |
| NFCorpus | NFCorpus | 0.407 | 0.377 | 0.390 | 0.136 | 1.000 |
| NFCorpus | SciFact | 1.064 | 1.057 | 1.048 | 0.748 | 1.000 |

Same-dataset transfer recovers the main learned-policy result. Cross-domain
transfer is more nuanced: the SciFact-trained policy still beats the NFCorpus
target train-best fixed action but pays many hybrid calls and trails the
NFCorpus heuristic by 0.003 reward; the NFCorpus-trained policy transfers to
SciFact with a small reward gain over both target train-best and heuristic while
using only one retrieval call. This is useful research evidence because it shows
that the learned routing rule is not purely memorizing one dataset, but it also
shows domain mismatch affects cost behavior.

### NFCorpus Learning Curve

This learning-curve run evaluates whether the policy improves as more training
qrels are available. It uses NFCorpus full-corpus retrieval, 150 held-out test
queries, and `auto` model selection including the MLP candidate.

| Train Examples | Selected Model | Policy Reward | Train-Best Reward | Oracle Reward | Policy Recall@5 | Calls |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 50 | `knn_k=9` | 0.362 | 0.360 | 0.439 | 0.119 | 1.020 |
| 100 | `knn_k=1` | 0.371 | 0.360 | 0.439 | 0.117 | 1.000 |
| 200 | `ridge_l2=1.0` | 0.377 | 0.340 | 0.439 | 0.126 | 1.047 |

### Contextual-Bandit Baselines and Budget Curves

The main selective policy is a direct-method offline contextual bandit: it uses
the full action table during training to learn a reward predictor per action.
To make the RL/bandit comparison explicit without cluttering the main result, I
report only the best selected-action baseline from LinUCB, epsilon-greedy, and
linear Thompson replay. Each baseline chooses one action per training query,
observes only that selected action's reward, and updates that action's linear
model. The individual baseline rows remain in the evidence CSVs.

| Dataset | Best Selected-Feedback Baseline | Reward | Train-Best Reward | Oracle Reward | Recall@5 | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | Linear Thompson | 1.062 | 1.057 | 1.206 | 0.739 | 1.033 |
| NFCorpus | Linear Thompson | 0.404 | 0.377 | 0.464 | 0.136 | 1.137 |

Linear Thompson sampling is the strongest selected-feedback baseline in these
runs: it improves over train-best by 0.005361 reward on SciFact and 0.026873 on
NFCorpus. The full-information direct-method selective policy remains stronger
on the same held-out tables, especially on SciFact where it reaches 1.090489
reward. This is the expected tradeoff: selected-feedback baselines are closer to
a deployable online bandit protocol, while the direct method is stronger when
offline full-information action rewards are available for training.

The project also adds retrieval-call budget curves. These curves exclude the
oracle row and ask which reported method is best under a fixed expected
retrieval-call budget.

| Dataset | Call Budget | Selected Method | Reward | Recall@5 | Calls |
| --- | ---: | --- | ---: | ---: | ---: |
| SciFact | 1.00 | Dense original | 1.042 | 0.741 | 1.000 |
| SciFact | 1.25 | Heuristic retrieval router | 1.048 | 0.733 | 1.053 |
| SciFact | 1.50 | Confidence-gated retrieval policy | 1.090 | 0.774 | 1.413 |
| SciFact | 2.00 | Confidence-gated retrieval policy | 1.090 | 0.774 | 1.413 |
| NFCorpus | 1.00 | Confidence-gated retrieval policy | 0.407 | 0.136 | 1.000 |
| NFCorpus | 1.25 | Confidence-gated retrieval policy | 0.407 | 0.136 | 1.000 |
| NFCorpus | 1.50 | Confidence-gated retrieval policy | 0.407 | 0.136 | 1.000 |
| NFCorpus | 2.00 | Confidence-gated retrieval policy | 0.407 | 0.136 | 1.000 |

This strengthens the cost-aware claim. On SciFact, increasing the call budget
changes the best feasible strategy from a single-call dense retriever, to the
cheap heuristic router, to the learned policy. On NFCorpus, the learned policy
is already the best feasible method at budget 1.0, so additional call budget
does not improve the reported non-oracle frontier.

The budget curve above is post-hoc: it selects among already reported methods.
I therefore added a Lagrangian constrained-policy sweep that retrains the
direct-method policy with utility
`Recall@5 + 0.5 * MRR - RewriteCost - lambda * max(Calls - 1, 0)`. This makes
the retrieval-call penalty part of policy learning instead of only part of
analysis.

| Dataset | Lambda | Policy Reward | Utility | Recall@5 | Calls | Primary Action |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SciFact | 0.00 | 1.088 | 1.088 | 0.774 | 1.503 | `hybrid_keyword` |
| SciFact | 0.01 | 1.095 | 1.091 | 0.774 | 1.427 | `hybrid_keyword` |
| SciFact | 0.03 | 1.098 | 1.089 | 0.775 | 1.273 | `dense_keep` |
| SciFact | 0.06 | 1.084 | 1.081 | 0.761 | 1.047 | `dense_keep` |
| SciFact | 0.10 | 1.075 | 1.075 | 0.755 | 1.000 | `dense_keep` |
| NFCorpus | 0.00 | 0.410 | 0.410 | 0.137 | 1.000 | `dense_keep` |
| NFCorpus | 0.10 | 0.410 | 0.410 | 0.137 | 1.000 | `dense_keep` |

The SciFact frontier shows a real cost/reward transition: low penalties favor
hybrid-heavy behavior, moderate penalties keep reward high while reducing calls,
and high penalties collapse to single-call dense retrieval. NFCorpus is
different: the constrained policy already prefers one-call dense/BM25-style
actions, so increasing lambda does not change the call profile. This is useful
because it shows the cost-aware mechanism adapts differently by dataset rather
than imposing a fixed preference for hybrid retrieval.

Paired bootstrap intervals over held-out queries show that most constrained
utility gains are not just noise. On SciFact, the constrained policy improves
utility over the train-selected fixed action by +0.031 at lambda 0.00
with 95% CI [0.011, 0.054], by +0.047 at lambda 0.03 with CI [0.014, 0.082],
and by +0.033 at lambda 0.10 with CI [0.002, 0.066]. At lambda 0.20, the lower
bound touches zero, CI [-0.001, 0.068], so that high-penalty setting is less
secure. On NFCorpus, the utility gain is +0.033 for all tested lambdas with
positive CIs, and the retrieval-call delta is exactly 0.0 because both the
learned constrained policy and the train-selected fixed action use one call.

### Off-Policy Evaluation Diagnostics

To further connect the project to offline contextual bandits, I added simulated
logged-feedback OPE diagnostics. The exporter starts from the full-information
detailed retrieval table, samples one logged action per held-out query under a
behavior policy, records the logged reward and action propensity, and estimates
target-policy value using direct method, IPS, self-normalized IPS, and doubly
robust estimators. Because the full action table is available, the experiment
can compare every OPE estimate against the true target-policy reward.

For the selective retrieval policy, the main OPE results are:

| Dataset | Behavior Policy | Estimator | Estimate | True Reward | Abs. Error | Match Rate | ESS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | uniform | direct method | 1.067 | 1.090 | 0.023 | 0.160 | 48.0 |
| SciFact | uniform | IPS | 1.122 | 1.090 | 0.031 | 0.160 | 48.0 |
| SciFact | uniform | SNIPS | 1.169 | 1.090 | 0.078 | 0.160 | 48.0 |
| SciFact | uniform | doubly robust | 1.103 | 1.090 | 0.013 | 0.160 | 48.0 |
| NFCorpus | uniform | direct method | 0.355 | 0.407 | 0.052 | 0.160 | 48.0 |
| NFCorpus | uniform | IPS | 0.402 | 0.407 | 0.005 | 0.160 | 48.0 |
| NFCorpus | uniform | SNIPS | 0.419 | 0.407 | 0.012 | 0.160 | 48.0 |
| NFCorpus | uniform | doubly robust | 0.383 | 0.407 | 0.024 | 0.160 | 48.0 |

The estimator behavior is useful rather than uniformly flattering. Under
uniform logging, SciFact's doubly robust estimate is closest for the selective
policy, while NFCorpus's IPS estimate is closest. Under biased train-best or
heuristic-epsilon logging, effective sample size can collapse even when the
raw match rate looks moderate. For example, SciFact selective-policy OPE under
train-best-epsilon logging has match rate 0.267 but ESS only 14.6; the doubly
robust estimate undershoots the true reward by 0.109. Under heuristic-epsilon
logging, IPS overshoots SciFact selective reward by 0.449. This supports an
important offline-RL caveat: logged-policy coverage matters as much as the
choice of estimator.

Because the logged action sample is stochastic, I also repeated the OPE
simulation over 10 logging seeds. The table below reports mean absolute error
for the selective retrieval policy.

| Dataset | Behavior Policy | DM Error | IPS Error | SNIPS Error | DR Error | Mean ESS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | uniform | 0.023 | 0.127 | 0.037 | 0.058 | 49.8 |
| SciFact | train-best-epsilon | 0.023 | 0.268 | 0.122 | 0.093 | 13.4 |
| SciFact | heuristic-epsilon | 0.023 | 0.244 | 0.109 | 0.096 | 13.6 |
| NFCorpus | uniform | 0.052 | 0.052 | 0.044 | 0.038 | 51.5 |
| NFCorpus | train-best-epsilon | 0.052 | 0.066 | 0.095 | 0.068 | 15.0 |
| NFCorpus | heuristic-epsilon | 0.052 | 0.075 | 0.056 | 0.045 | 22.2 |

The repeated-seed view changes how the single-seed table should be read. On
SciFact, the direct-method reward model is the most stable estimator for the
selective policy in this setup, while IPS has high variance when the behavior
policy is biased away from the target actions. On NFCorpus, doubly robust is
best under uniform and heuristic-epsilon logging, but train-best-epsilon still
has low effective sample size for the learned policy and no estimator fully
removes the coverage problem. This gives the project a concrete offline-RL
analysis point rather than only reporting retrieval reward.

### NFCorpus Feature-Set Ablation

This ablation uses the same NFCorpus full-corpus setup with 300 training and
150 held-out test queries, but fixes the policy estimator to ridge regression so
the comparison isolates the input state features rather than changing the model.

| Feature Set | Selected Model | Policy Reward | Train-Best Reward | Oracle Reward | Policy Recall@5 | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `full` | `ridge_l2=1.0` | 0.386 | 0.340 | 0.439 | 0.130 | 1.000 |
| `no_query` | `ridge_l2=1.0` | 0.389 | 0.340 | 0.439 | 0.131 | 1.000 |
| `no_retrieval` | `ridge_l2=1.0` | 0.377 | 0.340 | 0.439 | 0.120 | 1.000 |
| `no_wh` | `ridge_l2=1.0` | 0.392 | 0.340 | 0.439 | 0.132 | 1.000 |
| `retrieval_only` | `ridge_l2=1.0` | 0.389 | 0.340 | 0.439 | 0.131 | 1.000 |

### Qualitative Policy Cases

The project also exports qualitative examples from the detailed held-out
prediction files. These rows show cases where the policy beats the train-best
fixed action, uses dense or hybrid retrieval, or avoids hybrid retrieval while
matching or improving reward.

| Dataset | Case | QID | Selected | Train-Best | Policy Reward | Train-Best Reward |
| --- | --- | --- | --- | --- | ---: | ---: |
| SciFact | beats train-best | 1150 | `bm25_keyword` | `hybrid_keyword` | 1.500 | 1.470 |
| SciFact | avoids hybrid | 171 | `bm25_keep` | `hybrid_keyword` | 1.500 | 1.470 |
| SciFact | uses dense | 431 | `dense_keep` | `hybrid_keyword` | 0.000 | -0.030 |
| NFCorpus | uses dense | PLAIN-1983 | `dense_keyword` | `dense_keep` | 0.656 | 0.656 |
| NFCorpus | beats train-best | PLAIN-850 | `bm25_keyword` | `dense_keep` | 0.588 | 0.535 |
| NFCorpus | beats train-best | PLAIN-3053 | `bm25_keep` | `dense_keep` | 1.167 | 0.833 |

### Policy Regret Diagnostics

The diagnostics file measures per-query regret against the oracle retrieval
action, whether the policy beats the train-best fixed action, whether it
selects the same action as the oracle, and how sharply separated the oracle
action is from the second-best candidate action. This complements mean reward by
showing whether the policy has learned useful per-query switching behavior and
whether the action labels have enough margin to be easy to learn.

| Dataset | Mean Regret | Median Regret | Zero-Regret Rate | Beats Train-Best | Matches Oracle Action | Oracle-Second Gap | Oracle Tie Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 0.116 | 0.015 | 50.0% | 55.0% | 21.3% | 0.036 | 2.97 |
| NFCorpus | 0.057 | 0.000 | 69.3% | 15.3% | 34.0% | 0.010 | 2.71 |
| NFCorpus semantic pilot | 0.059 | 0.000 | 74.0% | 4.0% | 26.0% | 0.006 | 2.66 |

### Paired Bootstrap Diagnostics

The project also reports paired bootstrap confidence intervals over held-out
queries. Each bootstrap sample resamples query IDs with replacement and measures
the selective policy minus the train-best fixed action. This gives an uncertainty
check for whether the mean gains are stable under query resampling.

| Dataset | Metric | Queries | Mean Delta | 95% CI | P(Delta > 0) |
| --- | --- | ---: | ---: | --- | ---: |
| SciFact | Reward | 300 | 0.034 | [0.009, 0.058] | 0.997 |
| SciFact | Recall@5 | 300 | 0.011 | [-0.008, 0.031] | 0.860 |
| NFCorpus | Reward | 300 | 0.030 | [0.005, 0.055] | 0.995 |
| NFCorpus | Recall@5 | 300 | 0.013 | [0.001, 0.028] | 0.981 |
| NFCorpus semantic pilot | Reward | 50 | 0.015 | [-0.003, 0.045] | 0.827 |
| NFCorpus semantic pilot | Recall@5 | 50 | 0.004 | [0.000, 0.011] | 0.866 |

### Embedding Workload Preflight

Before running a larger Vertex semantic-feature experiment, the project now
estimates how many `gemini-embedding-001` calls would miss the local JSONL cache.
This preflight uses the same BM25 top-5 passages that semantic state features
would embed, but it does not call Vertex AI.

| Dataset | Split | Examples | Query Misses | Document Misses | Total New Embeddings |
| --- | --- | ---: | ---: | ---: | ---: |
| NFCorpus | train | 300 | 300 | 1,023 | 1,323 |
| NFCorpus | test | 150 | 150 | 537 | 687 |
| NFCorpus | combined | 450 | 450 | 1,347 | 1,797 |
| SciFact | train | 300 | 300 | 1,147 | 1,447 |
| SciFact | test | 150 | 150 | 650 | 800 |
| SciFact | combined | 450 | 450 | 1,557 | 2,007 |

The semantic state depth is now configurable rather than fixed to top-5. The
reported semantic tables remain at depth 5 for compatibility, but `--semantic-depth`
can stage deeper top-k rank profiles up to depth 8. A no-API NFCorpus 20/20
depth-8 preflight against the existing pilot cache found all 40 query embeddings
already cached and 93 missing document embeddings out of 344 unique
query/document embeddings. This makes depth-8 a practical next Vertex smoke
run, but still not free enough to launch large repeated grids without preflight.

The project now has a dedicated semantic-depth sweep runner that launches the
same policy-model grid at multiple depths and writes paired depth effects. A
cached Vertex NFCorpus 20/20 depth-5 versus depth-8 smoke shows that deeper
rank profiles are not a clear win yet: for `full/ridge`, depth 8 improves
held-out reward by only 0.002 and validation reward by 0.0006, while for
`full/auto` it reduces held-out reward by 0.0083. The `no_semantic` controls
are unchanged across depths, as expected. The predictive diagnostics are also
mixed: depth-8 semantic rank-profile test R2 for dense advantage is 0.062
versus 0.060 at depth 5, but semantic rank-agreement test R2 for dense
advantage drops from 0.141 to 0.028. This makes depth a tunable research axis,
not a solved improvement.

The depth sweep now also exports a paired predictive-effects table, so these
feature-level changes are reproducible instead of hand-computed from separate
CSV files. In that table, depth 8 adds six active rank-profile columns, but the
dense-advantage test R2 delta is only +0.002 and the test-correlation delta is
-0.020. For semantic rank-agreement, dense-advantage test R2 drops by 0.113 and
test correlation drops by 0.237. The evidence therefore argues against simply
increasing embedding depth as the next main strategy.

The selection diagnostic is now depth-aware: when a grid contains multiple
semantic depths, `semantic_depth` is part of the configuration key instead of
collapsing depth 5 and depth 8 into the same policy label. On the cached Vertex
single-split depth sweep, validation and held-out reward both select
`depth=5/no_semantic/auto/extra_trees`, not a depth-8 semantic configuration.
That selected configuration still trails the train-best fixed action by 0.0083
reward, reinforcing that the single-split depth sweep is not enough to justify
a deeper semantic policy.

A repeated semantic-depth runner now wraps the same comparison across multiple
seeds and exports cross-seed stability summaries for both policy reward deltas
and predictive-diagnostic deltas. The checked-in run for this repeated-depth
path is a fake-embedder 2-seed smoke, so it is pipeline evidence rather than a
Vertex result; it confirms that future cached Vertex depth comparisons can
report depth-effect means, standard deviations, and win rates across seeds.

After validating that repeated-depth path, a cached Vertex 2-seed NFCorpus
20/20 ridge-only repeated depth smoke was run. The depth-8 preflight estimated
94 missing document embeddings and zero missing query embeddings across seeds
41 and 42. In the repeated run, depth 8 versus depth 5 improved `full/ridge`
held-out reward by 0.0211 on average across seeds, with a 1.0 win rate, and
Recall@5 by 0.0156, while retrieval calls increased by 0.025. Validation reward
delta was only 0.0003 with a 0.5 win rate, and `no_semantic/ridge` remained
unchanged. The predictive diagnostics are target-dependent: semantic
rank-profile dense-advantage test R2 increased by 0.0055 with a 1.0 positive
rate but its test-correlation delta was near zero, while semantic
rank-agreement improved oracle-reward test R2 and correlation. This makes depth
8 a promising but still unproven policy feature rather than a settled upgrade.
The new depth-selection stability export makes the validation issue explicit:
held-out reward prefers depth 8 in both seeds, but validation selects depth 8
in only one of the two seeds. A validation-selected depth would leave an average
0.0201 held-out reward delta on the table relative to the held-out-best depth.

A larger cached Vertex repeated-depth grid was then staged with a no-API
preflight before making new embedding calls. For NFCorpus seeds 41, 42, and 43
with 30 train and 30 held-out queries per seed at depth 8, the preflight
estimated 533 missing embeddings across 1,310 unique query/document texts. The
run compared depth 5 versus depth 8 for `full` and `no_semantic` feature sets
under both ridge and `auto` policy families. This larger grid reverses the
small 20/20 signal: `full/auto` has mean held-out reward delta -0.0060 for
depth 8, with win rate 0.33, while `full/ridge` is only +0.0017 with win rate
0.67. The stability artifact now also reports paired bootstrap intervals over
the three seeds: `full/auto` reward delta has interval [-0.019894, 0.008454],
and `full/ridge` has interval [-0.008071, 0.010826]. Both cross zero, so depth
8 is not a statistically stable improvement in this pilot. Both full semantic settings
are validation-negative on average;
`full/auto` uses 0.0667 fewer calls while `full/ridge` uses 0.0667 more calls.
Depth selection remains unstable: validation matches the held-out-best depth in
zero of three seeds for both `full/auto` and `full/ridge`; the mean
depth-selection reward gap is 0.0145 for `full/auto` and 0.0093 for
`full/ridge`. The Wilson interval for both 0/3 match rates is [0.0000,
0.5615], so this should be read as weak pilot evidence against the current
validation rule rather than proof that validation can never select depth
correctly. The new per-seed depth-selection diagnostics show the mismatch
directly: seed 41 validation keeps depth 5 while held-out prefers depth 8,
seeds 42 and 43 validation select depth 8 for `full/auto` while held-out
prefers depth 5, and `full/ridge` alternates the same failure pattern. This is
stronger evidence that depth 8 should remain a tunable hypothesis, not the
default final policy.

### NFCorpus Semantic Feature Pilot

After the preflight, a small API-controlled NFCorpus pilot was run with 50 train
queries and 50 test queries against the full 3,633-document corpus. The policy
uses Vertex AI `gemini-embedding-001` semantic rank-shape features plus
top-k rank-aware similarity profiles, BM25/semantic rank-agreement signals,
compact direct embedding projections, and lexical-semantic interaction terms,
expanding the reported state vector to 56 features. The current code extends
this representation with four additional semantic score-shape features, so
rerunning the same command now produces a 60-feature state and enables
score-shape-specific ablations. The main pilot fixes the
direct-method estimator to ridge regression to isolate the semantic-feature
change.

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 keyword | 0.148 | 0.517 | 0.344 | 0.407 | 0.000 | 1.000 |
| Dense original | 0.128 | 0.543 | 0.351 | 0.400 | 0.000 | 1.000 |
| Hybrid keyword | 0.151 | 0.534 | 0.353 | 0.388 | 0.030 | 2.000 |
| Train-best retrieval action | 0.148 | 0.517 | 0.344 | 0.407 | 0.000 | 1.000 |
| Selective retrieval policy | 0.148 | 0.482 | 0.338 | 0.386 | 0.003 | 1.100 |
| Oracle retrieval action | 0.169 | 0.639 | 0.432 | 0.488 | 0.001 | 1.020 |

### NFCorpus Semantic Feature Ablation

To check whether the pilot improvement actually comes from the Vertex embedding
features, the same 50/50 NFCorpus split was rerun with semantic-specific feature
masks. `no_semantic` removes the semantic and interaction dimensions,
`no_profile` removes only the top-k rank-aware similarity profile,
`profile_only` keeps only that profile, `no_rank_agreement` removes the
BM25/semantic rank-agreement features, `rank_agreement_only` keeps only those
rank-agreement features, `no_projection` removes only the direct embedding
projection features, `projection_only` keeps only those projection features,
`no_interactions` removes only the lexical-semantic products, `semantic_only`
keeps the semantic summary, rank-profile, and rank-agreement dimensions, and
`interactions_only` keeps only the six product features.
The implementation now also supports `no_score_shape` and `score_shape_only`;
the cached rerun below adds those rows without requiring new Vertex embeddings.

| Feature Set | Selected Model | Policy Reward | Train-Best Reward | Oracle Reward | Policy Recall@5 | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `full` | `ridge_l2=1.0` | 0.386 | 0.407 | 0.488 | 0.148 | 1.100 |
| `no_semantic` | `ridge_l2=1.0` | 0.429 | 0.407 | 0.488 | 0.152 | 1.000 |
| `no_profile` | `ridge_l2=1.0` | 0.394 | 0.407 | 0.488 | 0.148 | 1.080 |
| `profile_only` | `ridge_l2=1.0` | 0.407 | 0.407 | 0.488 | 0.148 | 1.000 |
| `no_rank_agreement` | `ridge_l2=1.0` | 0.422 | 0.407 | 0.488 | 0.152 | 1.020 |
| `rank_agreement_only` | `ridge_l2=1.0` | 0.355 | 0.407 | 0.488 | 0.129 | 1.140 |
| `no_projection` | `ridge_l2=1.0` | 0.386 | 0.407 | 0.488 | 0.148 | 1.100 |
| `projection_only` | `ridge_l2=1.0` | 0.407 | 0.407 | 0.488 | 0.148 | 1.000 |
| `no_interactions` | `ridge_l2=1.0` | 0.386 | 0.407 | 0.488 | 0.148 | 1.080 |
| `semantic_only` | `ridge_l2=1.0` | 0.368 | 0.407 | 0.488 | 0.139 | 1.160 |
| `interactions_only` | `ridge_l2=1.0` | 0.400 | 0.407 | 0.488 | 0.141 | 1.000 |

The score-shape rerun first confirmed zero cache misses for the same 50/50
NFCorpus split: all 100 query embeddings and 401 unique document embeddings
were already cached. With the new 60-feature state, `full` remains at 0.386
reward, `no_score_shape` is identical to `full`, and `score_shape_only` reaches
only the train-best fixed reward of 0.407. Against the `no_semantic` baseline
at 0.429 reward, `full` and `no_score_shape` both lose 0.0428 reward while
using 0.10 more retrieval calls; `score_shape_only` loses 0.0221 reward with no
call increase. This makes score-shape a negative control rather than evidence
that the embedding state should be selected.

### NFCorpus Semantic Policy-Model Sweep

The same cached 50/50 semantic-feature split was also used to compare policy
estimators while holding the retrieval evaluations fixed. This checks whether
the negative semantic ablation is simply caused by ridge regression being too
weak to exploit the semantic representation.

| Policy Model | Selected Model | Validation Reward | Policy Reward | Train-Best Reward | Oracle Reward | Policy Recall@5 | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ridge` | `ridge_l2=1.0` | 0.376 | 0.386 | 0.407 | 0.488 | 0.148 | 1.100 |
| `ridge_sweep` | `ridge_l2=10.0` | 0.385 | 0.388 | 0.407 | 0.488 | 0.139 | 1.020 |
| `margin_ridge` | `margin_ridge_l2=1.0` | 0.388 | 0.390 | 0.407 | 0.488 | 0.139 | 1.040 |
| `extra_trees` | `extra_trees` | 0.361 | 0.398 | 0.407 | 0.488 | 0.149 | 1.160 |
| `random_forest` | `random_forest` | 0.374 | 0.417 | 0.407 | 0.488 | 0.155 | 1.200 |
| `mlp` | `mlp` | 0.365 | 0.394 | 0.407 | 0.488 | 0.137 | 1.200 |
| `auto` | `knn_k=3` | 0.399 | 0.383 | 0.407 | 0.488 | 0.137 | 1.020 |

### NFCorpus Semantic Feature-Model Grid

Because feature ablations and policy-model sweeps answer different questions,
the same cached split was also evaluated as a feature-set by policy-model grid.
This checks whether a stronger estimator rescues a specific semantic feature
subset, and whether validation reward reliably selects the best held-out
configuration.

| Feature Set | Policy Model | Selected Model | Validation Reward | Policy Reward | Train-Best Reward | Calls |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `full` | `ridge` | `ridge_l2=1.0` | 0.376 | 0.386 | 0.407 | 1.100 |
| `full` | `mlp` | `mlp` | 0.357 | 0.408 | 0.407 | 1.380 |
| `no_rank_agreement` | `ridge` | `ridge_l2=1.0` | 0.392 | 0.422 | 0.407 | 1.020 |
| `no_rank_agreement` | `auto` | `knn_k=1` | 0.393 | 0.400 | 0.407 | 1.000 |
| `no_profile` | `ridge` | `ridge_l2=1.0` | 0.377 | 0.394 | 0.407 | 1.080 |
| `no_profile` | `auto` | `knn_k=3` | 0.399 | 0.399 | 0.407 | 1.100 |
| `no_semantic` | `ridge` | `ridge_l2=1.0` | 0.395 | 0.429 | 0.407 | 1.000 |
| `no_semantic` | `auto` | `knn_k=9` | 0.400 | 0.413 | 0.407 | 1.000 |
| `profile_only` | `mlp` | `mlp` | 0.367 | 0.439 | 0.407 | 1.020 |
| `profile_only` | `ridge` | `ridge_l2=1.0` | 0.376 | 0.407 | 0.407 | 1.000 |

The validation-selection diagnostic summarizes the same grid as a model
selection problem. Selecting the highest validation reward chooses
`no_semantic/auto/knn_k=9`, which has held-out reward 0.413 and ranks fourth by
held-out reward. The held-out best configuration is
`profile_only/mlp/mlp` at 0.439 reward. The selection gap is therefore 0.0260
reward, and the Spearman correlation between validation reward and held-out
reward across the 15 grid configurations is only 0.204. The stronger
ranking-stability diagnostics tell the same story: validation and held-out
rankings have zero top-1 and top-2 overlap, with only one shared item in the
top-3 sets. The validation winner is 0.006 reward above train-best on held-out
and 0.020 above the train-best fixed action on validation; the held-out winner
is 0.032 above train-best on held-out but requires 0.02 more retrieval calls.
The new selection-stability exporter wraps the same diagnostics for multiple
grid CSVs so future repeated-seed runs can report modal validation choices,
unique selected configurations, mean selection gaps, validation-side fixed
baseline gaps, and mean top-k overlap instead of relying on one split.

A no-Vertex repeated-selection smoke run now exercises that path on two
NFCorpus seeds with fake embeddings, 20 training queries, 20 held-out queries,
and a small `full`/`no_query` by ridge/auto grid. It is an orchestration check,
not final semantic evidence. Even in this small run, validation chose two
different configurations across the two seeds, the held-out-best configuration
also changed across seeds, validation matched held-out best in 50% of runs, and
the mean selection reward gap was 0.0132. This confirms that the reporting code
can surface cross-seed selection instability before running a more expensive
cached Vertex semantic grid.

The paired repeated-embedding preflight estimates that the same two-seed
NFCorpus smoke setting would touch 80 unique query texts and 377 unique BM25
top-5 passage texts across seeds. Against the current local
`nfcorpus_pilot_vertex_embeddings.jsonl` cache, 42 query embeddings and 91
document embeddings are already cached, leaving 324 total missing embeddings.
This gives a concrete cache-miss budget before enabling Vertex semantic
features in repeated policy-model selection.

After the preflight, the same two-seed repeated-selection run was executed with
Vertex semantic features and a small `full` versus `no_semantic` grid. The run
completed end to end and still showed unstable model selection: validation chose
two different configurations across seeds, held-out best also changed across
seeds, validation matched held-out best in 50% of runs, and the mean selection
reward gap increased to 0.0348. Seed 41 selected `full/auto/knn_k=9`, which was
also held-out best for that seed at 0.550 reward. Seed 42 selected
`full/auto/mlp`, but held-out best was `no_semantic/auto/knn_k=15` at 0.628
reward. This is stronger evidence than the fake-embedding smoke because it uses
the actual 56-dimensional Vertex semantic state, but it remains a tiny pilot and
should not be used as the final policy-selection result.

The feature-effect aggregation makes the same point directly by pairing `full`
against `no_semantic` within each seed and policy-model family. Across the four
paired comparisons, the full semantic state has a small positive validation
delta (+0.0013 reward) but a negative held-out reward delta (-0.0222), a
negative Recall@5 delta (-0.0069), and requires 0.10 more retrieval calls on
average. The `auto` policy family shows positive validation deltas in both
seeds, but only wins held-out reward in one of two seeds. This is evidence that
the current semantic representation can look attractive on validation while
still failing to improve held-out retrieval behavior.

The repeated-selection workflow was then rerun on a larger cached Vertex
setting with seeds 41, 42, and 43, 30 training queries and 30 held-out queries
per seed, depth-5 semantic features, and a ridge/`auto` by `full`/`no_semantic`
grid. A preflight showed zero missing embeddings across 939 unique cached
query/document texts, so this run did not require new Vertex embedding calls.
The larger run now shows cleaner model-selection agreement: validation matches
the held-out-best configuration in all three seeds, the mean selection reward
gap is 0.0000, and the mean validation-vs-held-out Spearman correlation is
0.867. However, agreement is not the same as beating the baseline. The
validation-selected and held-out-best configurations are the same rows, and
they remain 0.0039 reward below the best fixed action on average while also
falling 0.0013 below the train-best fixed action on validation. The selection
stability report includes best-fixed beat rates, which makes the guardrail
explicit: selected policies beat the train-best fixed action in only one of
three seeds and require 0.0667 more retrieval calls on average. Combining
reward and cost into a dominance check gives the clearest guardrail: in two of
three seeds, the selected policy is dominated by train-best fixed, meaning it
has no higher reward and no lower retrieval-call cost. The stability export now
turns that diagnosis into a simple deployment guardrail: when
validation-selected is dominated, fall back to train-best fixed. On this 3-seed
grid the guardrail falls back in two of three seeds, raises mean reward by
0.0064 versus the raw validation-selected policy, and reduces retrieval calls
by 0.0111 on average. The repeated-selection runner now also exports a
per-seed diagnostics table, which shows why the aggregate guardrail fires:
seed 41 selects `no_semantic/ridge` and loses 0.0192 reward while using 0.0333
more calls than train-best fixed, seed 42 selects `no_semantic/auto` and ties
train-best fixed reward and calls, and seed 43 selects `full/auto/knn_k=3` with
a +0.0074 reward gap but 0.1667 more calls. Paired feature effects against
`no_semantic` also remain weak: `full/auto` loses 0.0013 held-out reward while
using 0.1333 more
retrieval calls, `full/ridge` loses 0.0191 reward, and the aggregate `full`
effect is -0.0102 reward despite a near-zero validation delta of +0.0009.
This larger repeated-selection result supports using the semantic feature grid
as an analysis tool, but not as the final selected policy.

The selection-protocol summary turns the repeated diagnostics into a final
decision table. For the NFCorpus 30/30x3 Vertex setting, the policy-model layer
gets `fallback_to_train_best_fixed`: the learned validation-selected policy is
below train-best fixed by 0.0039 reward on average, below it by 0.0013 on
validation, uses 0.0667 more calls, and triggers the fallback guardrail in
two of three seeds. Its machine-readable reason is
`fallback_rate=0.667; mean_reward_gap=-0.003939; mean_call_gap=0.066667`.
Because this is still a three-seed run, the summary also reports Wilson
support-rate intervals: the learned-policy support rate is 0.333 with a
0.061-0.792 interval, so the recommendation is conservative rather than a
claim that the learned policy is conclusively worse under all resamplings.
The row is therefore marked `evidence_strength=pilot_low_n` with
`n_runs=3; support_ci_width=0.731`.
The semantic-depth layer gets `do_not_select_depth_by_validation` for
`full/auto` and `full/ridge` because validation never selects the held-out-best
depth across the three seeds, with mean depth-selection reward gaps of 0.0145
and 0.0093. Their reasons are `match_rate=0.000` with positive selection gaps;
the Wilson upper bound for that zero-of-three match rate is still 0.562, so the
right interpretation is "insufficient support to deploy", not a proof that
depth validation can never work.
The only positive depth-selection verdict is the trivial `no_semantic/ridge`
row, where depth does not change the active state. This makes the practical
protocol explicit: keep the non-semantic train-best fixed checkpoint as the
deployment-safe baseline, and treat semantic depth and learned model selection
as analysis axes until a stronger validation protocol reverses these
diagnostics. The companion deployment-decision artifact records this as
`recommended_runtime_policy=train_best_fixed_retrieval_action`,
`learned_policy_status=analysis_only_guardrailed`,
`semantic_depth_strategy=do_not_select_full_depth_by_validation`, and
`semantic_feature_status=analysis_only`, with
`decision_confidence=pilot_low_n`.

To reduce avoidable model-selection noise in these small pilots, the `auto`
policy sweep now supports an explicit candidate-family list. A no-MLP NFCorpus
smoke run with `knn,ridge,extra_trees,random_forest` records that candidate set
in the grid CSV and selects `ridge_l2=1.0` for both `full` and `no_semantic`
feature sets, without invoking the high-variance MLP candidate. This keeps the
MLP available for deliberate larger runs while making small repeated-selection
diagnostics cheaper and easier to interpret.

The semantic-feature calibration probe adds `semantic_zscore`, which preserves
the lexical/query features but standardizes the semantic summary, rank profile,
rank-agreement, direct projection, and interaction dimensions using training-set
statistics. The current transformer also covers score-shape dimensions when
they are present. On the cached NFCorpus 20/20 Vertex smoke with no-MLP `auto`,
this does not rescue the semantic state: against `no_semantic`, `semantic_zscore`
has a mean held-out reward delta of -0.0097, a validation reward delta of
-0.0474, and uses 0.10 more retrieval calls on average across ridge and auto.
This suggests that the current semantic underperformance is not only a raw
feature-scale problem.

The retrieval-contrast extension adds eight state features that summarize
BM25/dense/hybrid top-k agreement for the original and keyword-compressed
queries, plus the dense and hybrid new-document rates relative to BM25. These
features are optional and can be ablated with `no_contrast` and
`contrast_only`. A no-Vertex NFCorpus 20/20 smoke run with fake dense embeddings
verifies the pipeline and records 22-dimensional states when contrast is
enabled. In that smoke, `full`, `no_contrast`, and `contrast_only` all select
`ridge_l2=1.0` and have identical held-out reward, so this is not positive
evidence yet; it is a controlled path for testing whether retrieval-system
disagreement helps larger dense/semantic runs.

The sweep now also exports feature-group diagnostics. On the same no-Vertex
contrast smoke, the retrieval-contrast group is not degenerate: all 8 contrast
columns are active on train and test, the nonzero rate is about 0.82, and the
mean column standard deviation is about 0.10. Therefore the neutral reward
result is not because the contrast features are constant in this smoke; it is
because the current small fake-embedding split does not convert that variation
into better action choices.

The feature-reward diagnostics make this more precise by correlating each
feature group with oracle reward, oracle margin, dense advantage over BM25, and
hybrid advantage over BM25. In the contrast smoke, the retrieval-contrast group
has a test-split mean absolute correlation of 0.265 with dense advantage and a
maximum absolute correlation of 0.534, plus a 0.197 mean absolute correlation
with hybrid advantage. The train split correlations are weaker. This supports a
conservative interpretation: the contrast features carry some action-relevant
signal, but the current small training split and fake dense embedding do not
turn it into a better learned policy.

The newest predictive diagnostic adds a held-out check beyond single-feature
correlation. For each feature group, it trains a ridge predictor on the training
split to predict oracle reward, oracle margin, dense advantage over BM25, and
hybrid advantage over BM25, then reports test R2 and correlation. On the same
20/20 fake-embedding contrast smoke, retrieval-contrast does not generalize:
test R2 is -0.284 for dense advantage and -0.184 for hybrid advantage. The best
small positive held-out result in this smoke is the retrieval-confidence group
on hybrid advantage, with test R2 0.010 and test correlation 0.443. This makes
the current conclusion stricter: retrieval-contrast is implemented and
measurable, but this split does not show that it is learnable enough for policy
selection.

### Natural Questions Single-Hop Retrieval

This first NQ run is the BM25-only rewrite policy. It verifies that the same
rewrite decision setup also runs on a single-hop dataset, but the action space is
thin because all actions use the same lexical retriever.

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla BM25 | 0.955 | 0.903 | 0.916 | 1.407 | 0.000 | 1.000 |
| Rewrite-all keyword | 0.970 | 0.936 | 0.945 | 1.438 | 0.000 | 1.000 |
| Selective bandit | 0.970 | 0.933 | 0.942 | 1.437 | 0.000 | 1.000 |
| Oracle best action | 0.970 | 0.940 | 0.948 | 1.440 | 0.000 | 1.000 |

### Natural Questions Retrieval-Action Policy

This stronger NQ run uses the same retrieval-action space as HotpotQA and
SciFact: BM25, dense, and hybrid retrieval with original or keyword-compressed
queries. The `auto` policy selected a ridge direct-method reward predictor
(`ridge_l2=1.0`) by 5-fold validation.

| Method | Recall@5 | MRR | nDCG@5 | Reward | Cost | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla BM25 | 0.955 | 0.901 | 0.914 | 1.405 | 0.000 | 1.000 |
| BM25 keyword | 0.970 | 0.934 | 0.943 | 1.437 | 0.000 | 1.000 |
| Dense original | 0.990 | 0.974 | 0.978 | 1.477 | 0.000 | 1.000 |
| Dense keyword | 0.990 | 0.974 | 0.978 | 1.477 | 0.000 | 1.000 |
| Hybrid original | 0.990 | 0.961 | 0.968 | 1.440 | 0.030 | 2.000 |
| Hybrid keyword | 0.995 | 0.967 | 0.974 | 1.449 | 0.030 | 2.000 |
| Train-best retrieval action | 0.990 | 0.974 | 0.978 | 1.477 | 0.000 | 1.000 |
| Selective retrieval policy | 0.990 | 0.976 | 0.980 | 1.478 | 0.000 | 1.000 |
| Oracle retrieval action | 0.995 | 0.980 | 0.984 | 1.485 | 0.000 | 1.000 |

## Analysis

Query rewriting improves retrieval over vanilla BM25 on both HotpotQA and Natural
Questions. The strongest simple baseline is keyword compression, which is hard
to beat in this toy setup. The selective bandit improves over vanilla BM25 but
does not consistently outperform the train-best fixed rewrite. The ablation
shows that the current keep action is too conservative on HotpotQA, while
removing the cost penalty increases retrieval calls and lowers cost-aware reward.

The oracle rows show substantial headroom. In HotpotQA one-step retrieval, the
oracle reaches 0.884 Recall@5, and the two-step oracle reaches 0.912 Recall@5.
This indicates that the action space is useful, but the current lightweight
state representation is not yet expressive enough to reliably choose the best
rewrite action per question.

The Gemini baseline shows that an external LLM can provide a stronger
decomposition-style rewrite on multi-hop questions, but it requires API calls and
has much higher rewrite cost. This supports the motivation for selective
rewriting: a production RAG system should learn when the additional rewrite
expense is justified.

When Gemini rewrite/decomposition is exposed as a selectable policy action, the
small HotpotQA run shows that the policy does not blindly call the LLM. Gemini
decomposition improves Recall@5 over BM25 on this sample, but the explicit API
cost makes its reward low. The selective policy reaches the same Recall@5 as the
best hybrid baseline while using fewer calls on average, 1.250 versus 2.000, and
it avoids expensive Gemini actions unless their expected reward is competitive.

The dense baseline addresses the main weakness of a BM25-only evaluation. Dense
retrieval improves semantic matching, and the hybrid BM25+dense variant is the
strongest non-LLM baseline in this report. Hybrid keyword retrieval reaches
0.887 Recall@5 on its held-out split, showing that the project is no longer only
compared against a thin lexical baseline. Because hybrid retrieval makes two
retrieval calls, the policy problem remains relevant: the learned system should
eventually decide when the additional dense pass is worth the extra retrieval
latency.

The retrieval-action policy makes that trade-off explicit. It nearly matches the
fixed hybrid-keyword action on Recall@5, improves cost-aware reward from 1.301
to 1.320, and reduces average retrieval calls from 2.000 to 1.242. The oracle
retrieval-action row reaches 0.950 Recall@5 with only 1.033 average calls, so
most remaining headroom is in better state features or a stronger policy model
rather than in adding more baseline actions.

A Vertex embedding feature probe on the same 300-example HotpotQA retrieval
policy split reached 1.304 reward. This is above the fixed hybrid-keyword
baseline at 1.301 but below the reported non-semantic policy at 1.320. The
semantic featurizer now records nine query-to-passage similarity statistics,
six BM25/semantic rank-agreement signals, 12 direct embedding projection
features, and six lexical-semantic interaction features. The current result
still suggests that semantic confidence needs stronger representation learning
or a larger validation setting before it should replace the non-semantic state
representation.

Tree-based and MLP reward predictors were also evaluated as stronger
alternatives to KNN. On the same HotpotQA split, `auto` selected RandomForest by
validation reward, but its held-out reward was 1.311, still below the current
KNN policy at 1.320. On full-corpus BEIR, `auto` considered the MLP candidate
but selected ExtraTrees for SciFact and ridge regression for NFCorpus. This is a
useful negative result: a larger policy estimator alone does not guarantee a
better policy when the state representation is still compact.

Using SciFact's official train/test qrels and full corpus makes the
cross-dataset result more realistic. The selective policy beats the best fixed
cost-aware action in reward, 1.090 versus 1.057, and improves Recall@5 from
0.763 to 0.774 while reducing average retrieval calls from 2.000 to 1.413. The
heuristic router is a useful adaptive baseline because it reduces calls to
1.053, but its reward is 1.048, below both train-best and the learned policy.
This supports the central claim across both multi-hop and scientific-claim
retrieval: the policy can learn when an expensive retrieval action is worth its
cost better than a simple confidence-rule router.

NFCorpus gives a second full-corpus BEIR check in a different domain. The
selective policy improves cost-aware reward over the strongest fixed baselines:
0.407 versus 0.385 for BM25 keyword and 0.377 for dense original. It also
improves Recall@5 to 0.136 while keeping average retrieval calls at 1.000,
meaning the learned policy finds better one-call retrieval choices rather than
paying for hybrid retrieval on most queries. The heuristic router reaches 0.390
reward, so it is stronger than the train-best fixed dense action but still below
the learned selective policy. The oracle reward of 0.464 still leaves meaningful
headroom for better state features or richer rewrite actions.

The action-distribution diagnostics show that the policy is not just a renamed
fixed baseline. On SciFact, the test policy uses all six base retrieval actions,
including hybrid keyword on 28.0% of queries and dense original on 20.0%. On
NFCorpus, the policy chooses among four one-call actions and avoids hybrid calls,
which explains why it improves reward without increasing average retrieval
calls. The NFCorpus learning curve adds another check: policy reward increases
from 0.362 with 50 training examples to 0.377 with 200 examples on the same
150-query test set, suggesting that there is a real training signal even though
the offline bandit trains quickly.

The difficulty-bucket diagnostics make that training signal more concrete. On
SciFact, low BM25-gap queries are the clearest adaptive case: the policy gains
0.054 reward over the train-best fixed action and 0.091 over the heuristic
router, because those queries benefit from switching among BM25, dense, and
hybrid methods rather than always using the fixed hybrid-keyword action. On
NFCorpus, the largest gains appear in high action-reward-dispersion bins and
mid BM25-top-score bins, where the policy improves reward without extra calls.
The same table also exposes failure modes, such as NFCorpus mid-length queries,
where the policy underperforms both train-best and heuristic. This is a better
research framing than only reporting a global average: adaptive retrieval helps
specific difficulty regimes, and those regimes can be audited.

The transfer matrix adds another robustness check. Training on SciFact and
testing on NFCorpus reaches 0.386 reward, above the NFCorpus target train-best
fixed action at 0.377 but slightly below the NFCorpus heuristic at 0.390 while
using 1.597 calls. Training on NFCorpus and testing on SciFact reaches 1.064
reward, narrowly above SciFact target train-best at 1.057 and heuristic at
1.048 while using only one call. The asymmetry is important: policy knowledge
does transfer, but source-domain cost preferences can be mismatched to the
target domain, especially when a source policy has learned to use hybrid
retrieval more often.

The confidence-gate sweep is mostly a guardrail diagnostic, not a replacement
for the learned policy. On SciFact, a margin of 0.001 is the only threshold that
slightly improves reward, 1.091 versus 1.090, while still reducing calls against
the train-best hybrid baseline. On NFCorpus, the same margin still beats
train-best but loses to the ungated learned policy. Larger thresholds quickly
turn the system back into the fixed train-best baseline, so the reported main
policy keeps the learned action unless a deployment setting specifically wants a
very low-threshold fallback.

The feature-set ablation gives a more direct check on whether the policy is
learning from retrieval state rather than only simple query text features. With
a fixed ridge estimator, removing retrieval statistics drops NFCorpus reward
from 0.386 to 0.377 and recall from 0.130 to 0.120. In contrast, removing the
WH-type features does not hurt this dataset, and a retrieval-only state remains
competitive. This suggests the current BEIR policy signal is driven mainly by
retrieval-confidence features, while the current query-form features are weak
for short biomedical and nutrition queries.

The qualitative exports make the same behavior inspectable at the example
level. SciFact contains cases where the policy avoids the train-best hybrid
action and keeps the same gold document at lower cost, such as QID 1150 and
QID 171. NFCorpus contains cases where the policy switches away from dense
retrieval and improves reward, such as PLAIN-850 and PLAIN-3053. These examples
are not a substitute for aggregate metrics, but they help diagnose whether the
policy choices are plausible rather than arbitrary.

The regret diagnostics add another view of policy quality. On SciFact, the
policy beats the train-best fixed action on 55.0% of held-out queries and
achieves zero oracle regret on 50.0%, even though exact oracle-action matching is
only 21.3%. This means the policy often reaches the same reward through a
different action, usually by avoiding unnecessary hybrid cost. On NFCorpus, the
policy has zero regret on 69.3% of queries and mean regret of 0.057, but beats
the train-best action on only 15.3%; the main gain there is avoiding bad action
choices while staying close to the oracle rather than frequently improving over
the fixed dense baseline. The new reward-gap columns also show why the offline
bandit can train quickly without being a trivial task: the median oracle margin
over the second-best candidate is zero on both BEIR datasets, and the mean
number of oracle-tied actions is about 2.7 to 3.0. Many labels are therefore
near-ties, so better feature alignment is more important than simply making the
policy estimator larger.

The paired bootstrap diagnostics make the aggregate BEIR gains more defensible.
For SciFact, the cost-aware reward gain over the train-best action is 0.034 with
a 95% bootstrap interval of [0.009, 0.058]. For NFCorpus, the reward gain is
0.030 with interval [0.005, 0.055], and the Recall@5 interval is also above zero.
In contrast, the 50-query semantic pilot reward interval crosses zero, so that
run should remain a pilot and not be treated as a statistically stable semantic
feature result.

The embedding preflight explains why the next semantic run should be staged
rather than launched blindly. The existing Vertex cache has no BEIR cache hits
for the current text hashes, so a 300/150 NFCorpus run would need 1,797 new
embeddings and the comparable SciFact run would need 2,007. A smaller NFCorpus
pilot with 50 train and 50 test queries needs 501 new embeddings, which is a
more controlled next API run before committing to a full semantic-feature table.
The new configurable-depth preflight provides the same control for deeper
embedding profiles; on the cached NFCorpus 20/20 depth-8 smoke, only 93 new
document embeddings are missing because the 40 query embeddings are already
cached. The paired depth sweep then showed that depth 8 is mixed rather than
strictly better: `full/ridge` gains 0.002 held-out reward, while `full/auto`
loses 0.0083 reward against the same depth-5 baseline.

The repeated cached Vertex depth smoke is more encouraging but still small:
with two NFCorpus 20/20 seeds and ridge-only policy learning, depth 8 improves
`full/ridge` held-out reward by 0.0211 and Recall@5 by 0.0156 on average over
depth 5, winning held-out reward in both seeds. The corresponding validation
gain is only 0.0003 and wins in one of two seeds, so the result supports
running a larger repeated depth study but not yet selecting depth 8 purely by
validation reward. The depth-selection stability table quantifies that gap:
validation matches the held-out-best depth in only 50% of runs, and the mean
held-out reward gap from validation-based depth selection is 0.0201.
This repeated result also disagrees with the depth-aware single-split selection
diagnostic, which chooses depth 5 with `no_semantic/auto`, so the next larger
run should test both model-family selection and semantic-depth selection rather
than treating either as fixed.

That larger repeated run has now been executed. With three seeds, 30/30 splits,
and both ridge and `auto`, depth 8 no longer looks like a reliable improvement:
`full/auto` drops held-out reward by 0.0060 on average, `full/ridge` gains only
0.0017, both full policies are validation-negative on average, and both paired
bootstrap reward intervals cross zero. Validation also never selects the
held-out-best depth for the full feature set, with Wilson match-rate upper
bound 0.5615 for the 0/3 pilot. The practical conclusion is that deeper
semantic profiles are useful for analysis but should not replace the
depth-5/non-semantic checkpoint without a stronger validation protocol.

The NFCorpus semantic pilot is a controlled result from that staging plan. On
the 50/50 full-corpus split, the 60-dimensional Vertex-feature ridge policy
drops reward from the train-best fixed action's 0.407 to 0.386 and uses 1.10
retrieval calls on average after adding BM25/semantic rank-agreement signals.
This is a useful negative result: more embedding-derived state is not
automatically better under a small validation split and ridge reward model.

The semantic-specific ablation qualifies that result. Removing all Vertex
semantic dimensions increases reward back to 0.429 on this split, and removing
only the new rank-agreement block recovers the prior 0.422 pilot reward.
Rank-agreement-only drops to 0.355, semantic-only drops to 0.368, and
interaction-only remains below train-best at 0.400. The current conclusion is
therefore not that the embedding features already solve policy selection; it is
that the pipeline can run cached Vertex semantic summaries, top-k similarity
profiles, BM25/semantic rank-agreement signals, compact direct embedding
projections, and interaction features, while the present semantic representation
is not yet carrying reliable additional signal beyond the existing query and
retrieval-confidence features.

The policy-model sweep adds a second check on that conclusion. With the full
semantic state, RandomForest reaches the best held-out reward
among the full-feature policy estimators at 0.417, but it still falls below the
`no_semantic` ridge ablation at 0.429 and pays 1.20 calls on average. Ridge
regularization tuning, margin-weighted ridge, MLP, and `auto` all remain below
train-best reward on the full semantic state. This suggests that richer
estimators do not rescue the current embedding representation before it is
better aligned with the action-selection reward.

The feature-model grid makes that point more explicit. The best held-out
configuration is `profile_only`+MLP at 0.439 reward and 1.02 retrieval calls on
average; `no_semantic`+ridge reaches 0.429 reward with one call. The
highest validation configuration in the grid is `no_semantic`+auto at 0.400
validation reward, but it reaches only 0.413 held-out reward and ranks fourth by
held-out reward. The validation versus held-out Spearman correlation is only
0.204, and the validation and held-out top-3 sets overlap on only one
configuration. This is a useful failure mode: the current 50-query validation
split is not reliable enough to jointly select semantic feature subsets and
policy estimators, so the next semantic step should either use a larger
validation split or calibrate the semantic features before using them in model
selection.

Natural Questions is useful as a third dataset, but it is easier than HotpotQA
and full-corpus BEIR datasets under the current sampled-negative setup. Dense
retrieval already reaches 0.990 Recall@5, so the learned policy has little room
to improve. The retrieval-action policy still slightly improves cost-aware
reward over the train-best dense action, 1.478 versus 1.477, and stays close to
the oracle at 1.485. This should be presented as a sanity check for generality,
not as the main evidence of technical depth.

## Checkpoints

The following trained policy artifacts are produced by the experiment scripts:

- `outputs/checkpoints/hotpot_bandit_policy.pkl`
- `outputs/checkpoints/nq_bandit_policy.pkl`
- `outputs/checkpoints/hotpot_retrieval_policy.pkl`
- `outputs/checkpoints/scifact_retrieval_policy.pkl`
- `outputs/checkpoints/nfcorpus_retrieval_policy.pkl`
- `outputs/checkpoints/nq_retrieval_policy.pkl`
- `outputs/checkpoints/hotpot_llm_retrieval_policy.pkl`
- `outputs/checkpoints/hotpot_fqi_q0.pkl`
- `outputs/checkpoints/hotpot_fqi_q1.pkl`

The current NFCorpus Vertex deployment-decision record is
`outputs/results/nfcorpus_vertex_deployment_decision.csv`. It recommends the
train-best fixed retrieval action for runtime use and marks the learned policy,
semantic features, and semantic-depth selection as analysis-only until stronger
validation evidence is available. Its evidence summary now carries the
depth-8-vs-depth-5 reward-effect intervals for the full semantic policies, so
the machine-readable decision record directly reflects the same uncertainty
reported in the text. Its confidence field is `pilot_low_n`, so it should be
read as a conservative pilot decision rather than a final large-sample claim.

The final evidence trail is indexed in
`outputs/results/final_artifact_index.csv`. That file records each core
artifact's relative path, existence status, byte size, short SHA256 digest,
role in the report, and producer command. This keeps the main claims tied to
specific generated CSVs and checkpoints instead of relying on a manual scan of
the output directory.

The project also writes `outputs/results/final_evidence_consistency.csv`, which
checks that the final-report deployment confidence, runtime recommendation, and
full semantic depth-effect confidence intervals match the protocol summary,
deployment decision, and artifact index. It is a lightweight guard against the
report drifting away from the machine-readable evidence.

For presentation defense, the project writes
`outputs/results/final_claims_matrix.csv`. This table links the main claims to
their metrics, baseline values, deltas, confidence intervals when available,
artifact IDs, file paths, and producer commands. It is the fastest way to audit
the central claims: SciFact policy reward improves by 0.033711 over the
train-best fixed action with CI [0.008852, 0.058341], NFCorpus improves by
0.029942 with CI [0.005428, 0.054799], the SciFact lambda=0.03 constrained
utility delta is 0.047350 with CI [0.014315, 0.082247], and the NFCorpus
lambda=0.03 constrained utility delta is 0.032656 with CI [0.010587, 0.058431].
The companion `FINAL_PRESENTATION_OUTLINE.md` turns those claim rows into a
slide sequence and likely defense answers.

The final paper-ready assets are generated by
`scripts/run_final_paper_assets.py`. The exporter writes
`outputs/results/final_main_results_table.csv` and
`outputs/results/final_main_results_table.tex`, then creates four figures under
`outputs/figures/`: `final_reward_delta_ci.png`,
`final_cost_reward_frontier.png`, `final_ope_estimator_error.png`, and
`final_linucb_comparison.png`. These artifacts are derived from the existing
retrieval-policy, bootstrap, constrained-policy, OPE, and selected-action
baseline result CSVs; they do not rerun retrieval or call external APIs.

The OPE values used in the final defense materials are also tied to the claims
matrix. For SciFact, IPS mean absolute error is 0.127173 under uniform logging
and 0.267959 under sparse train-best-epsilon logging. For NFCorpus, doubly
robust uniform-log error is 0.038419 versus 0.052150 for the direct-method
estimate.

Saved policies are also summarized in
`outputs/results/final_checkpoint_manifest.csv`. The manifest records the model
class, selected policy family, action space, inferred feature width, training
size, test size, and semantic-feature setting for each checkpoint. For example,
the current full-corpus checkpoints include 6-action retrieval policies with
14-dimensional non-semantic state features for SciFact and NFCorpus, while the
HotpotQA LLM-action checkpoint expands to 8 actions.

## Limitations

- No reader model is included, so EM/F1 answer quality is not reported.
- Natural Questions is evaluated as title retrieval with sampled negatives, not
  full open-domain retrieval.
- HotpotQA is evaluated against each example's distractor context rather than a
  full Wikipedia index.
- The learned policy is lightweight and interpretable but weaker than the best
  fixed rewrite on HotpotQA.
- Dense and hybrid retrieval are reported on a separate HotpotQA sample size
  from the main one-step BM25/RL experiment.
- Gemini results are currently from a small cached sample because Vertex API
  latency is high for full split evaluation.
- Vertex embedding state features are implemented, but the first full HotpotQA
  probe did not improve over the current non-semantic policy, and the NFCorpus
  semantic ablation shows no extra reward from the current semantic summary,
  rank-aware profile, BM25/semantic rank-agreement, direct projection, or
  lexical-semantic interaction dimensions; the score-shape rerun also matches
  or underperforms the non-semantic baseline.
- Cross-dataset transfer is promising but asymmetric: the NFCorpus-trained
  policy transfers to SciFact with a small gain and one retrieval call, while
  the SciFact-trained policy transfers to NFCorpus with extra hybrid calls and
  slightly trails the target-domain heuristic router.
- The selected-action bandit baselines are simulated offline replays from
  full-information action tables, not production online logging policies. They
  are useful as stronger RL-flavored baselines, but they do not replace a real
  deployment study with logged propensities and delayed user feedback.
- The OPE diagnostics are also simulated from full-information retrieval tables.
  They test estimator behavior under controlled logging policies, but they do
  not prove that real user traffic would have the same propensity coverage,
  delayed feedback quality, or reward noise.
- The constrained-policy sweep is a Lagrangian offline replay over the existing
  action table; it demonstrates reward/call tradeoffs but does not solve a
  production constrained MDP with hard per-user latency limits.
- Stronger policy estimators, ridge regularization tuning, and margin-aware
  reward weighting do not rescue the current semantic pilot; RandomForest is
  best on the full semantic state, but `no_semantic` fixed ridge
  still has the better held-out reward.
- Joint feature-set and policy-model selection is unstable on the 50-query
  semantic pilot: the highest validation configuration is not the best held-out
  configuration, with a 0.0260 reward selection gap, held-out rank 4 of 15, and
  only one shared item between the validation and held-out top-3 configurations.
- The checked-in no-Vertex repeated-selection smoke run confirms the cross-seed
  stability exporter, and the tiny Vertex semantic repeated-selection smoke
  runs end to end, but the semantic run is still only 2 seeds with 20/20
  examples per seed; it should not be treated as the final cross-seed result.
- A larger cached Vertex repeated-selection run with 3 seeds and 30/30 splits
  improves validation-held-out agreement to 3/3 seeds, but the selected
  configurations remain slightly below best fixed reward on average and full
  semantic features are still negative overall against `no_semantic`; the new
  best-fixed beat-rate and call-gap columns show selected policies beat
  train-best fixed in only 1/3 seeds while requiring more retrieval calls, and
  dominance rates show train-best fixed dominates selected policies in 2/3
  seeds. The best-fixed fallback guardrail also falls back in 2/3 seeds and
  improves mean reward while reducing retrieval calls. A per-seed diagnostics
  artifact now records the exact selected configuration, dominance decision,
  and guardrail delta for each seed.
- The repeated-embedding preflight reports cache misses before a repeated
  Vertex grid; before the tiny Vertex smoke it estimated 324 missing embeddings,
  so larger repeated grids should still be staged deliberately.
- The repeated feature-effect table shows that `full` semantic features are
  slightly positive on validation but negative on held-out reward and Recall@5
  against `no_semantic` in the tiny Vertex smoke.
- A cached Vertex repeated semantic-depth smoke now shows depth 8 beating depth
  5 for `full/ridge` held-out reward across two NFCorpus 20/20 seeds, but the
  validation delta is near zero and the run is still too small for a final
  semantic-depth claim.
- The larger cached Vertex 3-seed 30/30 repeated-depth grid weakens the depth-8
  case: `full/auto` is negative on held-out reward, `full/ridge` is nearly flat
  and validation-negative, and the new per-seed diagnostics show validation
  depth selection misses the held-out-best depth for the full feature set in
  all three seeds.
- Semantic-depth selection diagnostics now include depth in the configuration
  key; the single-split Vertex depth sweep selects depth 5 with `no_semantic`
  `auto`, while the tiny repeated ridge-only smoke favors depth 8 on held-out
  reward, so depth choice remains unstable.
- Repeated semantic-depth runs now export depth-selection stability; on the
  tiny Vertex repeated smoke, validation matches the held-out-best depth in
  only one of two seeds.
- Auto policy sweeps now record and can restrict candidate families, so small
  pilots can exclude MLP while larger runs can still include it deliberately.
- The `semantic_zscore` calibration feature set is implemented, but the cached
  20/20 NFCorpus smoke remains negative versus `no_semantic`, so feature scaling
  alone does not explain the semantic-feature gap.
- Retrieval-contrast features are implemented and ablatable, but the first
  fake-embedder 20/20 NFCorpus smoke shows no reward difference; larger dense
  or Vertex-backed runs are needed before claiming value.
- Feature-group diagnostics are now exported from policy sweeps; the contrast
  smoke confirms contrast features have nonzero variance even though reward is
  unchanged.
- Feature-reward diagnostics show retrieval-contrast features correlate with
  dense/hybrid advantage on the test split, but that signal is not yet converted
  into better learned action choices.
- Held-out feature-predictive diagnostics are now exported from policy sweeps;
  on the fake-embedder contrast smoke, retrieval-contrast has negative test R2
  for dense and hybrid advantage, so the current contrast signal should be
  treated as unstable until larger dense or Vertex-backed runs validate it.
- Tree-based direct-method policies are implemented, but the first HotpotQA
  `auto` run selected RandomForest and still underperformed the current KNN
  policy on held-out reward.
- Current query-form features are still weak on NFCorpus; the feature ablation
  suggests retrieval-confidence features carry most of the useful signal.

## Next Steps

- Add a small reader and evaluate EM/F1.
- Use the new semantic-depth sweep on larger cached NFCorpus splits; the first
  single-split 20/20 depth-8 smoke is mixed, while the 2-seed ridge-only Vertex
  repeated smoke is positive on held-out reward but nearly flat on validation.
  The 3-seed 30/30 ridge+auto Vertex grid is weaker for depth 8, so future work
  should focus on validation protocol and state representation rather than
  simply increasing semantic rank-profile depth.
- Improve the validation protocol before using validation reward to select
  semantic feature sets and policy estimators; the larger cached Vertex
  repeated-selection run reduces selection noise but still does not beat the
  best fixed action on average. The selection-protocol summary therefore
  recommends falling back to train-best fixed for deployment and avoiding
  validation-based semantic-depth selection for the full semantic state. This
  guardrail is a conservative interim deployment rule, not a substitute for a
  stronger validation split.
- Tune the keep-action threshold using validation data.
- Evaluate on larger Natural Questions and HotpotQA splits.
