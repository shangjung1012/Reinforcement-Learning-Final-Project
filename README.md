# Selective and Cost-Aware RL Query Rewriting for RAG

Final project code for the reinforcement learning course project. The current
implementation focuses on retrieval-stage RAG optimization: a lightweight policy
selects deterministic query rewrite actions and is evaluated by evidence
retrieval quality and rewrite/retrieval cost.

## What Is Included

- HotpotQA multi-hop toy retrieval environment.
- Natural Questions single-hop title retrieval environment.
- BEIR SciFact scientific-claim retrieval environment with qrels.
- BEIR NFCorpus biomedical/nutrition retrieval environment with qrels.
- BM25 retriever baseline.
- Dense sentence-transformer retrieval and BM25+dense hybrid baselines.
- Rewrite-all baselines for keyword compression and entity expansion.
- Train-best fixed rewrite baseline.
- Heuristic adaptive retrieval router baseline using deployable query and
  retrieval-confidence features.
- Selective contextual-bandit policy.
- Cost-aware retrieval-action policy over BM25, dense, and hybrid retrieval.
- Selective Gemini rewrite/decomposition actions as expensive retrieval experts.
- Literature-grounded generated-query actions for HyDE-style pseudo-documents,
  multi-query retrieval, and hybrid LLM decomposition with explicit generation
  and retrieval-call costs.
- Cross-validated policy estimator choices: KNN direct method, ridge direct
  method, ExtraTrees direct method, RandomForest direct method, and MLP neural
  direct method.
- Confidence-gated policy fallback based on learned action-score margins, not
  realized rewards.
- A compact selected-action contextual-bandit baseline in the final table,
  chosen from LinUCB, epsilon-greedy, and linear Thompson sampling runs that
  train from chosen-action feedback instead of full-information direct-method
  labels.
- Retrieval-call budget curves that select the best non-oracle method under
  fixed expected call budgets.
- Lagrangian constrained-policy sweeps that retrain a direct-method policy
  under different retrieval-call penalties.
- Paired bootstrap intervals for constrained-policy utility and call deltas.
- Offline contextual-bandit OPE diagnostics with direct method, IPS, SNIPS, and
  doubly robust estimators under simulated logged behavior policies.
- Repeated-seed OPE stability summaries reporting estimator error, uncertainty
  intervals, match rate, and effective sample size.
- Repeated-seed main retrieval robustness summaries for SciFact and NFCorpus
  that rerun full-corpus retrieval-action policies over multiple random seeds.
- Policy action-distribution, reward-gap diagnostics, query-difficulty bucket
  diagnostics, and NFCorpus learning-curve analysis.
- SciFact/NFCorpus cross-dataset policy-transfer matrix for testing whether a
  source-trained retrieval router generalizes across domains.
- Machine-readable final artifact index with existence, size, short SHA256, role,
  and producer command for the core report evidence.
- Machine-readable checkpoint manifest summarizing saved policy classes, action
  spaces, feature widths, training sizes, and selected model families.
- Machine-readable evidence-consistency check tying final-report claims to the
  protocol summary, deployment decision, and artifact index.
- Machine-readable final claims matrix linking the core presentation claims to
  exact metrics, deltas, confidence intervals, commands, and evidence artifacts.
- Paper-ready final result table, LaTeX table, and four presentation figures
  for reward CIs, constrained frontier, OPE estimator error, and compact
  selected-action bandit comparison.
- Final presentation outline with defense answers for novelty, RL framing,
  baselines, and limitations.
- Feature-set ablation for policy state inputs.
- Policy-model sweep diagnostics for semantic feature experiments.
- Feature-set by policy-model grid sweeps and ranking-stability diagnostics for
  semantic calibration checks.
- Paired bootstrap confidence intervals for policy-vs-baseline comparisons.
- Qualitative case export for questions where the policy changes action choice.
- Two-step fitted-Q extension for feedback-aware refinement.
- Ablations for keep action, cost penalty, and retrieval-only reward.
- Vertex AI Gemini rewrite/decomposition baselines with local JSONL caching.
- Optional Vertex AI `gemini-embedding-001` semantic state features with
  similarity summaries, top-k rank-aware similarity profiles, semantic
  score-shape features, BM25/semantic rank-agreement signals, compact direct
  embedding projections, lexical-semantic interaction features, and local JSONL
  caching.
- Saved policy checkpoints for one-step bandit, retrieval-action bandit, and
  two-step FQI models.

Raw datasets are expected under `data/raw/` and are intentionally ignored by git.

## Setup

```bash
uv sync
```

## Final Reproduction Quickstart

For a raw-data-free reviewer check, run:

```bash
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke
```

This smoke path uses a synthetic HotpotQA-style fixture, a fake dense embedder,
targeted pytest, retrieval-policy evaluation, and OPE diagnostics. It does not
use raw datasets, Vertex/Gemini, paid APIs, or model downloads. See
`docs/FINAL_REPRODUCTION.md` for smoke, full-data, and external-API
reproduction details.

For a lightweight downstream QA metric smoke, run:

```bash
uv run python scripts/run_reader_eval.py --dataset toy --output-dir outputs/codex_reader_smoke
```

This uses a deterministic lexical-overlap reader and reports SQuAD-style EM/F1.
It is an evaluation-plumbing check, not final answer-generation evidence. See
`docs/READER_EXTENSION.md`.

For the precise RL interpretation and claim boundary, see `docs/RL_FRAMING.md`.
For model-selection guardrails and reward/cost interpretation, see
`docs/VALIDATION_PROTOCOL.md` and `docs/COST_MODEL.md`.
For optional Vertex/Gemini setup and bounded preflight commands, see
`docs/API_EXPERIMENTS.md`.

## Expected Data Layout

```text
data/raw/HotpotQA/hotpot_dev_distractor_v1.json
data/raw/HotpotQA/hotpot_dev_fullwiki_v1.json
data/raw/HotpotQA/hotpot_train_v1.1.json
data/raw/natural-questions/default/validation-00000-of-00007.parquet
data/raw/scifact/corpus.jsonl
data/raw/scifact/queries.jsonl
data/raw/scifact/qrels/train.tsv
data/raw/scifact/qrels/test.tsv
data/raw/nfcorpus/corpus.jsonl
data/raw/nfcorpus/queries.jsonl
data/raw/nfcorpus/qrels/train.tsv
data/raw/nfcorpus/qrels/dev.tsv
data/raw/nfcorpus/qrels/test.tsv
```

The current scripts use the HotpotQA distractor dev file and one Natural
Questions validation shard by default. The SciFact script uses the BEIR official
archive layout with qrels and supports `--full-corpus` evaluation over all 5,183
SciFact corpus documents. NFCorpus is included as an additional small BEIR
full-corpus benchmark for biomedical/nutrition retrieval.

## Run

```bash
uv run pytest
uv run python scripts/run_hotpot_toy.py --num-examples 1000 --seed 42
uv run python scripts/run_hotpot_ablation.py --num-examples 1000 --seed 42
uv run python scripts/run_multistep_hotpot.py --num-examples 600 --seed 42
uv run python scripts/run_nq_toy.py --num-examples 500 --seed 42 --pool-size 50
uv run python scripts/run_retrieval_policy_nq.py --num-examples 500 --seed 42 --pool-size 50 --policy-model auto
uv run python scripts/run_dense_hotpot.py --num-examples 300 --seed 42
uv run python scripts/run_retrieval_policy_hotpot.py --num-examples 300 --seed 42
uv run python scripts/run_retrieval_policy_hotpot.py --num-examples 300 --seed 42 --policy-model auto
uv run python scripts/run_retrieval_policy_hotpot.py --num-examples 300 --seed 42 --semantic-features vertex --semantic-cache-path outputs/cache/vertex_embeddings.jsonl
uv run python scripts/run_llm_retrieval_policy_hotpot.py --num-examples 20 --seed 42 --policy-model auto
uv run python scripts/run_retrieval_policy_scifact.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --policy-model auto
uv run python scripts/run_retrieval_policy_nfcorpus.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --policy-model auto
uv run python scripts/run_bandit_baselines.py --dataset scifact --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --alpha 1.0
uv run python scripts/run_bandit_baselines.py --dataset nfcorpus --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --alpha 1.0
uv run python scripts/run_repeated_main_robustness.py --datasets scifact,nfcorpus --seeds 41,42,43 --num-train-examples 300 --num-test-examples 150 --full-corpus --policy-model auto --knn-k-candidates 1,3,5,7,9 --tuning-folds 5
uv run python scripts/run_budget_curve.py --dataset scifact_main --summary-csv outputs/results/scifact_retrieval_policy_summary.csv --output-csv outputs/results/scifact_budget_curve.csv --budgets 1.0,1.25,1.5,2.0
uv run python scripts/run_budget_curve.py --dataset nfcorpus_main --summary-csv outputs/results/nfcorpus_retrieval_policy_summary.csv --output-csv outputs/results/nfcorpus_budget_curve.csv --budgets 1.0,1.25,1.5,2.0
uv run python scripts/run_constrained_policy_sweep.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_constrained_policy_sweep.csv --call-penalties 0,0.01,0.03,0.06,0.1,0.2
uv run python scripts/run_constrained_policy_sweep.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_constrained_policy_sweep.csv --call-penalties 0,0.01,0.03,0.06,0.1,0.2
uv run python scripts/run_constrained_policy_bootstrap.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_constrained_policy_bootstrap.csv --call-penalties 0,0.01,0.03,0.06,0.1,0.2 --bootstrap-samples 1000 --seed 42
uv run python scripts/run_constrained_policy_bootstrap.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_constrained_policy_bootstrap.csv --call-penalties 0,0.01,0.03,0.06,0.1,0.2 --bootstrap-samples 1000 --seed 42
uv run python scripts/run_ope_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_ope_diagnostics.csv --seed 42
uv run python scripts/run_ope_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_ope_diagnostics.csv --seed 42
uv run python scripts/run_ope_stability.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_ope_stability.csv --seeds 1,2,3,4,5,6,7,8,9,10
uv run python scripts/run_ope_stability.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_ope_stability.csv --seeds 1,2,3,4,5,6,7,8,9,10
uv run python scripts/run_retrieval_policy_scifact.py --num-train-examples 100 --num-test-examples 100 --seed 42 --full-corpus --policy-model auto --generated-actions --llm-cache-path outputs/cache/scifact_generated_actions.jsonl
uv run python scripts/run_retrieval_policy_nfcorpus.py --num-train-examples 100 --num-test-examples 100 --seed 42 --full-corpus --policy-model auto --generated-actions --llm-cache-path outputs/cache/nfcorpus_generated_actions.jsonl
uv run python scripts/run_confidence_gate_sweep.py --dataset scifact_main --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_confidence_gate_sweep.csv --margins 0,0.001,0.005,0.01,0.02,0.05,0.1
uv run python scripts/run_confidence_gate_sweep.py --dataset nfcorpus_main --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_confidence_gate_sweep.csv --margins 0,0.001,0.005,0.01,0.02,0.05,0.1
uv run python scripts/run_complexity_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_complexity_buckets.csv --action-distribution-csv outputs/results/scifact_complexity_action_distribution.csv
uv run python scripts/run_complexity_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_complexity_buckets.csv --action-distribution-csv outputs/results/nfcorpus_complexity_action_distribution.csv
uv run python scripts/run_cross_dataset_transfer.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus
uv run python scripts/run_retrieval_policy_nfcorpus.py --output-dir outputs/confidence_gate_smoke --num-train-examples 12 --num-test-examples 8 --pool-size 20 --embedder fake --policy-model ridge --tuning-folds 2 --knn-k-candidates 1 --confidence-gate-margin 999
uv run python scripts/run_confidence_gate_sweep.py --dataset nfcorpus_confidence_gate_smoke --detailed-csv outputs/confidence_gate_smoke/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_confidence_gate_smoke_sweep.csv --margins 0,0.001,0.005,0.01,0.02,0.05,0.1,999
uv run python scripts/run_policy_learning_curve.py --dataset nfcorpus --train-sizes 50,100,200 --num-test-examples 150 --seed 42 --full-corpus --policy-model auto
uv run python scripts/run_feature_ablation.py --dataset nfcorpus --feature-sets full,no_query,no_retrieval,no_wh,retrieval_only --num-train-examples 300 --num-test-examples 150 --seed 42 --full-corpus --policy-model ridge
uv run python scripts/export_qualitative_examples.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_qualitative_examples.csv
uv run python scripts/export_qualitative_examples.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_qualitative_examples.csv
uv run python scripts/run_policy_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_policy_diagnostics.csv
uv run python scripts/run_policy_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_policy_diagnostics.csv
uv run python scripts/run_policy_diagnostics.py --dataset nfcorpus_semantic_projection_pilot --detailed-csv outputs/semantic_nfcorpus_projection_pilot/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_semantic_projection_policy_diagnostics.csv
uv run python scripts/run_statistical_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_bootstrap_diagnostics.csv
uv run python scripts/run_statistical_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_bootstrap_diagnostics.csv
uv run python scripts/run_embedding_preflight.py --dataset nfcorpus --num-train-examples 300 --num-test-examples 150 --seed 42 --full-corpus --cache-path outputs/cache/vertex_embeddings.jsonl
uv run python scripts/run_embedding_preflight.py --dataset scifact --num-train-examples 300 --num-test-examples 150 --seed 42 --full-corpus --cache-path outputs/cache/vertex_embeddings.jsonl
uv run python scripts/run_embedding_preflight.py --dataset nfcorpus --num-train-examples 20 --num-test-examples 20 --semantic-depth 8 --cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-csv outputs/results/nfcorpus_semantic_depth8_embedding_preflight_smoke.csv
uv run python scripts/run_semantic_depth_sweep.py --dataset nfcorpus --semantic-depths 5,8 --policy-models ridge,auto --feature-sets full,no_semantic --num-train-examples 20 --num-test-examples 20 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --knn-k-candidates 1,3 --auto-candidate-models knn,ridge,extra_trees,random_forest --output-dir outputs/semantic_depth_sweep_runs/nfcorpus_vertex_depth_smoke
uv run python scripts/run_repeated_semantic_depth_sweep.py --dataset nfcorpus --seeds 41,42 --semantic-depths 5,8 --policy-models ridge --feature-sets full,no_semantic --num-train-examples 4 --num-test-examples 4 --embedder fake --semantic-features none --knn-k-candidates 1 --tuning-folds 2 --output-dir outputs/repeated_semantic_depth_runs/nfcorpus_repeated_depth_smoke
uv run python scripts/run_repeated_embedding_preflight.py --dataset nfcorpus --seeds 41,42 --num-train-examples 20 --num-test-examples 20 --semantic-depth 8 --cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-csv outputs/results/nfcorpus_vertex_repeated_semantic_depth8_embedding_preflight.csv
uv run python scripts/run_repeated_semantic_depth_sweep.py --dataset nfcorpus --seeds 41,42 --semantic-depths 5,8 --policy-models ridge --feature-sets full,no_semantic --num-train-examples 20 --num-test-examples 20 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --knn-k-candidates 1,3 --tuning-folds 5 --output-dir outputs/repeated_semantic_depth_runs/nfcorpus_vertex_repeated_depth_smoke
uv run python scripts/run_repeated_embedding_preflight.py --dataset nfcorpus --seeds 41,42,43 --num-train-examples 30 --num-test-examples 30 --semantic-depth 8 --cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-csv outputs/results/nfcorpus_vertex_repeated_semantic_depth8_30x30x3_embedding_preflight.csv
uv run python scripts/run_repeated_semantic_depth_sweep.py --dataset nfcorpus --seeds 41,42,43 --semantic-depths 5,8 --policy-models ridge,auto --feature-sets full,no_semantic --num-train-examples 30 --num-test-examples 30 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --knn-k-candidates 1,3 --tuning-folds 5 --auto-candidate-models knn,ridge,extra_trees,random_forest --output-dir outputs/repeated_semantic_depth_runs/nfcorpus_vertex_repeated_depth_30x30x3
uv run python scripts/run_retrieval_policy_nfcorpus.py --num-train-examples 50 --num-test-examples 50 --seed 42 --full-corpus --policy-model ridge --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-dir outputs/semantic_nfcorpus_projection_pilot
uv run python scripts/run_feature_ablation.py --dataset nfcorpus --feature-sets full,no_semantic,no_profile,profile_only,no_score_shape,score_shape_only,no_rank_agreement,rank_agreement_only,no_projection,projection_only,no_interactions,semantic_only,interactions_only --num-train-examples 50 --num-test-examples 50 --seed 42 --full-corpus --policy-model ridge --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-dir outputs/feature_ablation_runs/semantic_nfcorpus_projection
uv run python scripts/run_policy_model_sweep.py --dataset nfcorpus --policy-models ridge,ridge_sweep,margin_ridge,extra_trees,random_forest,mlp,auto --feature-set full --num-train-examples 50 --num-test-examples 50 --seed 42 --full-corpus --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-dir outputs/policy_model_sweep_runs/semantic_nfcorpus_projection
uv run python scripts/run_policy_model_sweep.py --dataset nfcorpus --policy-models ridge,mlp,auto --feature-sets full,no_rank_agreement,no_profile,no_semantic,profile_only --num-train-examples 50 --num-test-examples 50 --seed 42 --full-corpus --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-dir outputs/policy_model_sweep_runs/semantic_nfcorpus_feature_grid
uv run python scripts/run_selection_diagnostics.py --dataset nfcorpus_semantic_feature_model_grid --grid-csv outputs/results/nfcorpus_semantic_feature_model_grid.csv --output-csv outputs/results/nfcorpus_semantic_feature_model_selection_diagnostics.csv
uv run python scripts/run_selection_stability.py --dataset nfcorpus_semantic_feature_model_grid --grid-csv outputs/results/nfcorpus_semantic_feature_model_grid.csv --output-csv outputs/results/nfcorpus_semantic_feature_model_selection_stability.csv
uv run python scripts/run_repeated_embedding_preflight.py --dataset nfcorpus --seeds 41,42 --num-train-examples 20 --num-test-examples 20 --cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-csv outputs/results/nfcorpus_repeated_embedding_preflight_smoke.csv
uv run python scripts/run_repeated_selection.py --dataset nfcorpus --seeds 41,42 --policy-models ridge,auto --feature-sets full,no_query --num-train-examples 20 --num-test-examples 20 --embedder fake --output-dir outputs/repeated_selection_runs/nfcorpus_smoke
uv run python scripts/run_repeated_selection.py --dataset nfcorpus --seeds 41,42 --policy-models ridge,auto --feature-sets full,no_semantic --num-train-examples 20 --num-test-examples 20 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-dir outputs/repeated_selection_runs/nfcorpus_vertex_smoke
uv run python scripts/run_feature_effects.py --dataset nfcorpus_vertex_repeated_smoke --grid-csv outputs/results/nfcorpus_vertex_repeated_selection_seed41_grid.csv --grid-csv outputs/results/nfcorpus_vertex_repeated_selection_seed42_grid.csv --baseline-feature-set no_semantic --output-csv outputs/results/nfcorpus_vertex_repeated_feature_effects.csv
uv run python scripts/run_repeated_embedding_preflight.py --dataset nfcorpus --seeds 41,42,43 --num-train-examples 30 --num-test-examples 30 --semantic-depth 5 --cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-csv outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_embedding_preflight.csv
uv run python scripts/run_repeated_selection.py --dataset nfcorpus --seeds 41,42,43 --policy-models ridge,auto --feature-sets full,no_semantic --num-train-examples 30 --num-test-examples 30 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --semantic-depth 5 --knn-k-candidates 1,3 --tuning-folds 5 --auto-candidate-models knn,ridge,extra_trees,random_forest --output-dir outputs/repeated_selection_runs/nfcorpus_vertex_30x30x3
uv run python scripts/run_feature_effects.py --dataset nfcorpus_vertex_repeated_selection_30x30x3 --grid-csv outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_seed41_grid.csv --grid-csv outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_seed42_grid.csv --grid-csv outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_seed43_grid.csv --baseline-feature-set no_semantic --output-csv outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_feature_effects.csv
uv run python scripts/run_selection_protocol_summary.py --dataset nfcorpus_vertex_30x30x3 --policy-diagnostics-csv outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_diagnostics.csv --depth-selection-diagnostics-csv outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_selection_diagnostics.csv --depth-stability-csv outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_stability.csv --output-csv outputs/results/nfcorpus_vertex_selection_protocol_summary.csv --deployment-decision-csv outputs/results/nfcorpus_vertex_deployment_decision.csv
uv run python scripts/run_checkpoint_manifest.py --output-csv outputs/results/final_checkpoint_manifest.csv
uv run python scripts/run_evidence_consistency.py --output-csv outputs/results/final_evidence_consistency.csv
uv run python scripts/run_final_claims_matrix.py --output-csv outputs/results/final_claims_matrix.csv
uv run python scripts/run_final_paper_assets.py --results-dir outputs/results --figures-dir outputs/figures
uv run python scripts/run_markdown_consistency.py --output-csv outputs/results/final_markdown_consistency.csv
uv run python scripts/run_artifact_index.py --output-csv outputs/results/final_artifact_index.csv
uv run python scripts/run_embedding_preflight.py --dataset nfcorpus --num-train-examples 50 --num-test-examples 50 --seed 42 --full-corpus --semantic-depth 5 --cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-csv outputs/results/nfcorpus_score_shape_50x50_embedding_preflight.csv
uv run python scripts/run_feature_ablation.py --dataset nfcorpus --feature-sets full,no_semantic,no_score_shape,score_shape_only,no_profile,profile_only,no_rank_agreement,rank_agreement_only --num-train-examples 50 --num-test-examples 50 --seed 42 --full-corpus --policy-model ridge --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --output-dir outputs/feature_ablation_runs/semantic_nfcorpus_score_shape
cp outputs/feature_ablation_runs/semantic_nfcorpus_score_shape/results/nfcorpus_feature_ablation.csv outputs/results/nfcorpus_score_shape_feature_ablation.csv
uv run python scripts/run_feature_effects.py --dataset nfcorpus_score_shape_ablation --grid-csv outputs/results/nfcorpus_score_shape_feature_ablation.csv --baseline-feature-set no_semantic --output-csv outputs/results/nfcorpus_score_shape_feature_effects.csv
uv run python scripts/run_policy_model_sweep.py --dataset nfcorpus --policy-models auto --feature-sets full,no_semantic --num-train-examples 20 --num-test-examples 20 --embedder fake --knn-k-candidates 1,3 --auto-candidate-models knn,ridge,extra_trees,random_forest --output-dir outputs/policy_model_sweep_runs/nfcorpus_auto_no_mlp_smoke
uv run python scripts/run_policy_model_sweep.py --dataset nfcorpus --policy-models ridge,auto --feature-sets full,semantic_zscore,no_semantic --num-train-examples 20 --num-test-examples 20 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --knn-k-candidates 1,3 --auto-candidate-models knn,ridge,extra_trees,random_forest --output-dir outputs/policy_model_sweep_runs/nfcorpus_semantic_zscore_smoke
uv run python scripts/run_feature_effects.py --dataset nfcorpus_semantic_zscore_smoke --grid-csv outputs/results/nfcorpus_semantic_zscore_smoke_grid.csv --baseline-feature-set no_semantic --output-csv outputs/results/nfcorpus_semantic_zscore_feature_effects.csv
uv run python scripts/run_policy_model_sweep.py --dataset nfcorpus --policy-models ridge,auto --feature-sets full,no_contrast,contrast_only --num-train-examples 20 --num-test-examples 20 --embedder fake --knn-k-candidates 1,3 --auto-candidate-models knn,ridge,extra_trees,random_forest --retrieval-contrast-features --output-dir outputs/policy_model_sweep_runs/nfcorpus_retrieval_contrast_smoke
uv run python scripts/run_feature_effects.py --dataset nfcorpus_retrieval_contrast_smoke --grid-csv outputs/results/nfcorpus_retrieval_contrast_smoke_grid.csv --baseline-feature-set no_contrast --output-csv outputs/results/nfcorpus_retrieval_contrast_feature_effects.csv
uv run python scripts/run_policy_model_sweep.py --dataset nfcorpus --policy-models ridge,auto --feature-sets full,no_contrast,contrast_only --num-train-examples 20 --num-test-examples 20 --embedder fake --knn-k-candidates 1,3 --auto-candidate-models knn,ridge,extra_trees,random_forest --retrieval-contrast-features --output-dir outputs/policy_model_sweep_runs/nfcorpus_retrieval_contrast_diagnostics_smoke
uv run python scripts/run_feature_effects.py --dataset nfcorpus_feature_reward_diagnostics_smoke --grid-csv outputs/results/nfcorpus_feature_reward_diagnostics_smoke_grid.csv --baseline-feature-set no_contrast --output-csv outputs/results/nfcorpus_feature_reward_diagnostics_feature_effects.csv
uv run python scripts/run_policy_model_sweep.py --dataset nfcorpus --policy-models ridge,auto --feature-sets full,no_contrast,contrast_only --num-train-examples 20 --num-test-examples 20 --embedder fake --knn-k-candidates 1,3 --auto-candidate-models knn,ridge,extra_trees,random_forest --retrieval-contrast-features --output-dir outputs/policy_model_sweep_runs/nfcorpus_feature_predictive_diagnostics_smoke
uv run python scripts/run_feature_effects.py --dataset nfcorpus_feature_predictive_diagnostics_smoke --grid-csv outputs/results/nfcorpus_feature_predictive_diagnostics_smoke_grid.csv --baseline-feature-set no_contrast --output-csv outputs/results/nfcorpus_feature_predictive_diagnostics_feature_effects.csv
uv run python scripts/run_gemini_baseline.py --num-examples 20 --seed 42
```

## Final Defense Assets

Use the Markdown files below as the source for the final presentation and later
LaTeX conversion:

- `FINAL_RESULTS_SUMMARY.md`: one-page result summary with the main numbers,
  confidence intervals, evidence files, and limitations.
- `FINAL_SLIDES.md`: 13-slide Markdown draft with slide message, figure/table
  references, key numbers, and defense points.
- `FINAL_DEFENSE_QA.md`: prepared answers for likely questions about RL
  framing, supervised reward modeling, baselines, related work, cost-aware
  rewards, OPE, semantic features, and limitations.
- `FINAL_PRESENTATION_OUTLINE.md`: shorter outline version.
- `FINAL_REPORT.md`: full written report.

The generated result assets used by those Markdown files are:

- `outputs/results/final_main_results_table.csv`
- `outputs/results/final_claims_matrix.csv`
- `outputs/figures/final_reward_delta_ci.png`
- `outputs/figures/final_cost_reward_frontier.png`
- `outputs/figures/final_ope_estimator_error.png`
- `outputs/figures/final_linucb_comparison.png`
- `outputs/repeated_main_runs/results/repeated_main_robustness_aggregate.csv`
- `outputs/repeated_main_runs/results/repeated_main_robustness_per_seed.csv`

Outputs are written to:

```text
outputs/results/
outputs/figures/
outputs/checkpoints/
outputs/cache/
```

## Current Scope

This version does not include a neural reader or final generated-answer EM/F1
benchmark. A deterministic lexical-overlap reader smoke path is available for
testing downstream QA metrics, but the final scope remains retrieval-stage
selective query rewriting, using Recall@5, MRR, nDCG@5, retrieval calls, and
rewrite cost as the main evaluation signals.
Vertex semantic features are implemented as an optional extension and cached
under `outputs/cache/`. They include query-to-top-passage similarity summaries
and top-k rank-aware similarity profiles, score-shape features over the head
and tail of the semantic similarity list, BM25/semantic rank-agreement signals,
compact projections of the query embedding, top-passage centroid, and
query-centroid delta, plus lexical-semantic interaction features;
the main reported tables keep the stronger non-semantic retrieval-policy
checkpoint unless a semantic-feature run improves validation and held-out reward.
The semantic profile depth is configurable with `--semantic-depth` from 3 to 8;
the default remains 5 for compatibility with the reported tables, while depth-8
preflight can estimate additional Vertex cache misses before a deeper embedding
run. `run_semantic_depth_sweep.py` compares multiple depths with paired
validation, held-out reward, recall, retrieval-call, and feature-predictive
diagnostic deltas, and its selection diagnostic treats semantic depth as part
of the configuration key. `run_repeated_semantic_depth_sweep.py` repeats that
comparison across seeds and exports cross-seed stability summaries, including
paired bootstrap confidence intervals for depth-effect means, per-seed
depth-selection diagnostics, and Wilson intervals for whether
validation-selected depth matches the held-out-best depth. The checked
artifacts include both a
fake-embedder orchestration smoke and a cached Vertex 2-seed NFCorpus 20/20
ridge-only repeated-depth smoke, followed by a larger cached Vertex 3-seed
NFCorpus 30/30 ridge+auto repeated-depth grid.
Gemini rewrite/decomposition actions are implemented as
expensive optional actions in a separate HotpotQA experiment; cache files remain
under `outputs/cache/` and are ignored by git. The BEIR retrieval-policy scripts
also support `--generated-actions`, which expands the policy action space with
Gemini-generated HyDE pseudo-documents, multi-query retrieval, and hybrid
decomposition actions without overwriting the main non-generated result files.
These generated runs write `*_retrieval_policy_generated.*` outputs and should
be staged on smaller splits before a full-corpus API run. Feature ablation, policy-model
sweeps, feature-model grid sweeps, validation-selection ranking diagnostics,
qualitative exports, per-query regret diagnostics, paired bootstrap confidence
intervals, and embedding-cache preflight reports are included to show which
state inputs and estimators matter, where the policy is making different
action choices than the strongest fixed baseline, whether mean gains are stable
under query resampling, whether model selection remains stable across repeated
grid runs, how often selected policies beat the train-best fixed action, and
whether selected policies require more retrieval calls than that fixed-action
baseline, whether they beat the train-best fixed action on the validation split
before held-out evaluation, whether they are reward/cost dominated by that
baseline, how much a best-fixed fallback guardrail would change reward and
retrieval calls in each repeated seed and in aggregate, and how expensive a
semantic feature run will be before calling Vertex AI. The
checked-in repeated-selection smoke run uses fake embeddings and
small NFCorpus splits only to verify the cross-seed orchestration and stability
export path; it is not treated as semantic-feature evidence. The repeated
embedding preflight reports cross-seed cache hits and misses before launching
any Vertex-backed repeated grid. A tiny Vertex-backed repeated-selection smoke
run is also checked in as an initial end-to-end semantic stability result, but
a larger cached 3-seed NFCorpus 30/30 repeated-selection grid is now included
to test validation stability before selecting the final semantic policy.
The feature-effect aggregator compares semantic feature sets against a named
baseline across repeated grid CSVs so validation gains can be checked against
held-out reward and retrieval-call cost.
Retrieval-policy scripts also support `--confidence-gate-margin`, which adds a
confidence-gated policy row. The gate uses the learned model's top-vs-runner-up
predicted action-score margin and falls back to the train-best fixed action
when the policy is not confident; it does not use held-out realized rewards.
`run_confidence_gate_sweep.py` can then replay any detailed CSV with predicted
margins across multiple thresholds and export fallback rate, reward deltas, and
retrieval-call deltas without rerunning retrieval.
The main SciFact and NFCorpus full-corpus tables now include a heuristic
adaptive router as a stronger non-learned baseline. It uses only query length
and BM25 top-score/gap/entropy features available before action selection.
The full-corpus confidence-gate sweeps are also checked in, showing that very
small margins can act as a diagnostic guardrail, while high thresholds mostly
collapse the policy back to the train-best fixed action.
The full-corpus detailed outputs also include deployable query-difficulty
features, oracle reward margins, action reward dispersion, and
BM25/dense/hybrid overlap signals. `run_complexity_diagnostics.py` buckets
held-out queries by those features to show where the learned router gains,
where it loses, and whether the policy changes action mix on harder queries.
`run_cross_dataset_transfer.py` trains the retrieval policy on SciFact or
NFCorpus and evaluates it on both domains, providing an explicit transfer
check beyond same-dataset train/test reward.
`run_bandit_baselines.py` adds online-style selected-action comparisons by
replaying the full-information action table as bandit feedback: for each
training query, LinUCB, epsilon-greedy, and linear Thompson policies choose one
retrieval action, observe only that chosen action's reward, update that arm, and
are then evaluated on the held-out full action table. The final paper table uses
only the best of these selected-action baselines, while the per-policy rows stay
in the baseline summary CSVs. `run_budget_curve.py` turns the
summary tables into constrained retrieval-call curves by reporting the best
non-oracle method feasible under each expected-call budget.
`run_constrained_policy_sweep.py` goes one step further by retraining a
direct-method policy on Lagrangian utilities
`recall@5 + 0.5 * MRR - rewrite_cost - lambda * max(calls - 1, 0)` across
multiple call penalties, then reporting the learned policy's reward/call
frontier against train-selected fixed actions and per-query oracle utilities.
`run_constrained_policy_bootstrap.py` repeats the constrained-policy comparison
with paired bootstrap resampling over held-out queries, exporting confidence
intervals for utility deltas and retrieval-call deltas against the
train-selected fixed action.
`run_ope_diagnostics.py` converts the detailed full-information retrieval table
into simulated logged bandit feedback under uniform, train-best-epsilon, and
heuristic-epsilon behavior policies. It reports direct-method, IPS, SNIPS, and
doubly robust estimates against the known full-information policy value, making
coverage and estimator error explicit without requiring online deployment.
`run_ope_stability.py` repeats that logging simulation across seeds and exports
mean absolute error, standard deviation, 95% normal-approximation intervals,
match rate, and effective sample size for each behavior policy, target policy,
and estimator.
The selection-protocol summary combines the repeated policy and semantic-depth
diagnostics into an explicit recommendation table for whether to trust
validation selection or fall back to the train-best fixed action, including a
machine-readable `decision_reason` and Wilson support-rate interval for each
recommendation, semantic-depth reward-effect confidence intervals when a
repeated-depth stability artifact is supplied, plus an `evidence_strength`
caveat for low-seed pilots.
`nfcorpus_vertex_deployment_decision.csv` then condenses that table into the
runtime decision record used by the final report.
`final_checkpoint_manifest.csv` summarizes saved policy checkpoints with their
model classes, action spaces, feature widths, training/test sizes, selected
policy families, and semantic-feature settings.
`final_evidence_consistency.csv` checks that the final-report deployment
confidence, runtime-policy recommendation, full semantic depth-effect
confidence intervals, and required decision artifacts match the machine-readable
CSV evidence.
`final_claims_matrix.csv` is the presentation-facing evidence table: it links
the central claims to exact values, baselines, deltas, uncertainty intervals,
artifact IDs, paths, and producer commands. `FINAL_PRESENTATION_OUTLINE.md`
turns that matrix into a concise slide sequence and defense Q&A.
`final_main_results_table.csv`, `final_main_results_table.tex`, and the
`outputs/figures/final_*.png` figures are generated paper assets for the final
slides: main reward deltas with confidence intervals, constrained reward/call
frontiers, OPE estimator errors, and selected-action bandit versus direct-method
comparisons.
`final_markdown_consistency.csv` checks the final Markdown/report files against
the generated result values and verifies that referenced output artifacts exist.
`final_artifact_index.csv` records the core final-report artifacts, their
relative paths, existence status, byte size, short SHA256 digest, role, and
producer command so the evidence trail is auditable without manually searching
the output directory.
For small repeated-selection pilots, `--auto-candidate-models` can exclude MLP
from `auto` while still comparing KNN, ridge, and tree policies; the chosen
candidate set is written into sweep CSVs for reproducibility.
The `semantic_zscore` feature set standardizes semantic, projection, and
interaction dimensions using training-set statistics, then applies the same
transform to validation/test examples. The semantic ablation masks include
`no_score_shape` and `score_shape_only` so the policy can test whether these
head-vs-tail similarity-shape features carry useful signal beyond the rank
profile and rank-agreement groups. The cached 50/50 NFCorpus score-shape rerun
has zero embedding cache misses and shows `no_score_shape` matches `full`, while
`score_shape_only` only matches the train-best fixed action and remains below
`no_semantic`.
The optional retrieval-contrast state adds BM25/dense/hybrid top-k overlap and
new-document-rate features, with `no_contrast` and `contrast_only` masks for
ablation. Policy-model sweeps also write feature-group and feature-reward
diagnostics that report feature-group variance and correlations with oracle
reward, oracle margin, dense advantage, and hybrid advantage. The sweeps also
write held-out feature-predictive diagnostics that train a ridge predictor per
feature group and report whether the group can predict oracle reward or
dense/hybrid advantage on the test split.
