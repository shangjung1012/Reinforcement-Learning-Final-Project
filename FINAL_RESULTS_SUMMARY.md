# Final Results Summary

## One-Sentence Claim

This project formulates retrieval-stage RAG control as an offline contextual
bandit and shows that a lightweight learned policy can improve cost-aware
evidence retrieval over fixed retrieval actions, while exposing the tradeoffs
through selected-action replay, constrained utility, OPE, and bootstrap
intervals.

## Main Result

| Dataset | Method | Reward / Utility | Delta vs Train-Best | 95% CI | Calls |
| --- | --- | ---: | ---: | ---: | ---: |
| SciFact | Selective retrieval policy | 1.090489 | +0.033711 | [0.008852, 0.058341] | 1.413333 |
| SciFact | Constrained policy lambda=0.03 | 1.089378 | +0.047350 | [0.014315, 0.082247] | 1.273333 |
| NFCorpus | Selective retrieval policy | 0.407095 | +0.029942 | [0.005428, 0.054799] | 1.000000 |
| NFCorpus | Constrained policy lambda=0.03 | 0.409809 | +0.032656 | [0.010587, 0.058431] | 1.000000 |

Evidence:

- `outputs/results/final_main_results_table.csv`
- `outputs/results/final_claims_matrix.csv`
- `outputs/figures/final_reward_delta_ci.png`
- `outputs/figures/final_cost_reward_frontier.png`

## Bandit Baseline

The main summary reports only the best selected-action baseline, chosen from
LinUCB, epsilon-greedy, and linear Thompson replay. The individual baseline rows
remain in the evidence CSVs for defense and appendix use.

| Dataset | Best Selected-Feedback Baseline | Reward | Train-Best Reward | Delta | Calls |
| --- | --- | ---: | ---: | ---: | ---: |
| SciFact | Linear Thompson | 1.062139 | 1.056778 | +0.005361 | 1.033333 |
| NFCorpus | Linear Thompson | 0.404025 | 0.377153 | +0.026873 | 1.136667 |

Evidence:

- `outputs/results/scifact_linucb_baseline_summary.csv`
- `outputs/results/nfcorpus_linucb_baseline_summary.csv`
- `outputs/figures/final_linucb_comparison.png`

## Repeated-Seed Robustness

The main table remains the primary seed-42 result. As a robustness check, I
reran the full-corpus SciFact and NFCorpus retrieval-action experiments over
three seeds with 300 train and 150 test examples per seed. The learned policies
do not win every seed, but the mean reward deltas remain positive on both
datasets.

| Dataset | Seeds | Selective Win Rate | Selective Mean Delta | Constrained Win Rate | Constrained Mean Delta | Best Bandit Win Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 3 | 0.667 | +0.021304 | 0.667 | +0.025856 | 1.000 |
| NFCorpus | 3 | 0.667 | +0.024816 | 1.000 | +0.031224 | 0.667 |

Evidence:

- `outputs/repeated_main_runs/results/repeated_main_robustness_per_seed.csv`
- `outputs/repeated_main_runs/results/repeated_main_robustness_aggregate.csv`

## OPE Finding

The OPE diagnostics show why offline contextual-bandit evaluation needs logging
coverage. On SciFact, IPS error is much worse under sparse train-best-epsilon
logging than under uniform logging. On NFCorpus, the doubly robust estimator
improves uniform-log absolute error relative to the direct-method estimate.

Key rows:

- SciFact IPS coverage warning: error increases from 0.127173 to 0.267959.
- NFCorpus doubly robust stability: error decreases from 0.052150 to 0.038419.

Evidence:

- `outputs/results/scifact_ope_stability.csv`
- `outputs/results/nfcorpus_ope_stability.csv`
- `outputs/figures/final_ope_estimator_error.png`

## What To Claim

Claim:

> A lightweight offline contextual-bandit policy can improve cost-aware
> retrieval-stage RAG performance over strong fixed-action baselines on SciFact
> and NFCorpus, and the improvement remains visible under paired bootstrap
> uncertainty checks.

Do not claim:

- A new full RL-RAG architecture.
- Answer-generation improvement.
- LLM fine-tuning.
- A production-ready semantic embedding policy.

## Current Limitations

- The project evaluates evidence retrieval, not generated-answer EM/F1.
- Direct-method policy learning uses replayable full-information action tables;
  selected-action replay is included to show bandit-feedback behavior.
- Semantic embedding features remain analysis-only because the cached pilot
  evidence did not provide stable enough validation gains.
- OPE is simulated from the full-information retrieval table rather than
  collected from real online logs.
