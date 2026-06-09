# Experiment Dashboard

This dashboard separates final benchmark evidence from smoke checks, API pilots,
and analysis-only artifacts. `claim_allowed=false` means the artifact can support
engineering confidence or future work, but should not be used as a final project
claim without stronger evidence.

## Evidence Level Counts

| evidence_level | count |
| --- | --- |
| analysis_only | 2 |
| api_pilot | 17 |
| final_claim | 20 |
| full_benchmark | 24 |
| smoke_synthetic | 3 |
| tiny_realdata | 23 |

## Artifact Evidence Levels

| artifact_id | dataset | experiment_type | evidence_level | claim_allowed | supports_final_claim | notes |
| --- | --- | --- | --- | --- | --- | --- |
| downstream_qa_gap_table | unknown | evidence_boundary | analysis_only | False | False | claim-boundary or gap-analysis artifact |
| downstream_qa_gap_table_markdown | unknown | evidence_boundary | analysis_only | False | False | claim-boundary or gap-analysis artifact |
| hotpot_gemini_pilot_summary | hotpot | api_pilot | api_pilot | False | False | API-backed pilot or semantic analysis only |
| hotpot_gemini_reader_pilot_summary | hotpot | api_pilot | api_pilot | False | False | API-backed pilot or semantic analysis only |
| hotpot_gemini_repeated_pilot_summary | hotpot | api_pilot | api_pilot | False | False | API-backed pilot or semantic analysis only |
| hotpot_llm_policy_summary | hotpot | main_result | api_pilot | False | False | API-backed pilot or semantic analysis only |
| hotpot_policy_gemini_reader_comparison_detailed | hotpot | api_pilot | api_pilot | False | False | API-backed pilot or semantic analysis only |
| hotpot_policy_gemini_reader_comparison_summary | hotpot | api_pilot | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nfcorpus_vertex_deployment_decision | nfcorpus | decision_record | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nfcorpus_vertex_protocol_summary | nfcorpus | decision_record | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nfcorpus_vertex_repeated_10x10_diagnostics | nfcorpus | selection_check | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nfcorpus_vertex_repeated_10x10_stability | nfcorpus | selection_check | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nfcorpus_vertex_repeated_selection_diagnostics | nfcorpus | selection_check | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nfcorpus_vertex_repeated_semantic_depth_diagnostics | nfcorpus | selection_check | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nfcorpus_vertex_repeated_semantic_depth_selection_stability | nfcorpus | selection_check | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nfcorpus_vertex_repeated_semantic_depth_stability | nfcorpus | selection_check | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nq_gemini_reader_pilot_summary | nq | api_pilot | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nq_policy_gemini_reader_comparison_detailed | nq | api_pilot | api_pilot | False | False | API-backed pilot or semantic analysis only |
| nq_policy_gemini_reader_comparison_summary | nq | api_pilot | api_pilot | False | False | API-backed pilot or semantic analysis only |
| final_main_results_table | multiple | paper_asset | final_claim | True | False | final report, figure, or claim artifact |
| nfcorpus_bandit_replay_regret_figure | nfcorpus | paper_asset | final_claim | True | False | final report, figure, or claim artifact |
| scifact_bandit_replay_regret_figure | scifact | paper_asset | final_claim | True | False | final report, figure, or claim artifact |
| experiment_dashboard_markdown | unknown | document | final_claim | True | False | final report, figure, or claim artifact |
| final_checkpoint_manifest | unknown | checkpoint_manifest | final_claim | True | False | final report, figure, or claim artifact |
| final_claims_matrix | unknown | defense_artifact | final_claim | True | False | final report, figure, or claim artifact |
| final_cost_reward_frontier_figure | unknown | paper_asset | final_claim | True | False | final report, figure, or claim artifact |
| final_defense_qa | unknown | document | final_claim | True | False | final report, figure, or claim artifact |
| final_evidence_consistency | unknown | consistency_check | final_claim | True | False | final report, figure, or claim artifact |
| final_linucb_comparison_figure | unknown | paper_asset | final_claim | True | False | final report, figure, or claim artifact |
| final_main_results_latex | unknown | paper_asset | final_claim | True | False | final report, figure, or claim artifact |
| final_markdown_consistency | unknown | consistency_check | final_claim | True | False | final report, figure, or claim artifact |
| final_ope_estimator_error_figure | unknown | paper_asset | final_claim | True | False | final report, figure, or claim artifact |
| final_presentation_outline | unknown | document | final_claim | True | False | final report, figure, or claim artifact |
| final_report | unknown | document | final_claim | True | False | final report, figure, or claim artifact |
| final_results_summary | unknown | document | final_claim | True | False | final report, figure, or claim artifact |
| final_reward_delta_ci_figure | unknown | paper_asset | final_claim | True | False | final report, figure, or claim artifact |
| final_slides_markdown | unknown | document | final_claim | True | False | final report, figure, or claim artifact |
| poster_claim_audit | unknown | document | final_claim | True | False | final report, figure, or claim artifact |
| readme | unknown | document | final_claim | True | False | final report, figure, or claim artifact |
| cross_dataset_transfer_summary | multiple | transfer_check | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| nfcorpus_bandit_replay_history | nfcorpus | bandit_diagnostic | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| nfcorpus_bandit_replay_summary | nfcorpus | bandit_diagnostic | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| nfcorpus_bootstrap | nfcorpus | statistical_check | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| nfcorpus_complexity_buckets | nfcorpus | diagnostic | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| nfcorpus_constrained_policy_bootstrap | nfcorpus | constrained_bandit | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| nfcorpus_learning_curve | nfcorpus | diagnostic | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| nfcorpus_linucb_baseline_history | nfcorpus | bandit_diagnostic | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| nfcorpus_linucb_baseline_summary | nfcorpus | bandit_baseline | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| nfcorpus_main_confidence_gate_sweep | nfcorpus | diagnostic | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| nfcorpus_ope_stability | nfcorpus | off_policy_evaluation | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| nfcorpus_policy_checkpoint | nfcorpus | checkpoint | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| nfcorpus_policy_summary | nfcorpus | main_result | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| scifact_bandit_replay_history | scifact | bandit_diagnostic | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| scifact_bandit_replay_summary | scifact | bandit_diagnostic | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| scifact_bootstrap | scifact | statistical_check | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| scifact_complexity_buckets | scifact | diagnostic | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| scifact_constrained_policy_bootstrap | scifact | constrained_bandit | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| scifact_linucb_baseline_history | scifact | bandit_diagnostic | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| scifact_linucb_baseline_summary | scifact | bandit_baseline | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| scifact_main_confidence_gate_sweep | scifact | diagnostic | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| scifact_ope_stability | scifact | off_policy_evaluation | full_benchmark | True | True | final retrieval-stage benchmark evidence |
| scifact_policy_checkpoint | scifact | checkpoint | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| scifact_policy_summary | scifact | main_result | full_benchmark | True | False | final retrieval-stage benchmark evidence |
| nfcorpus_confidence_gate_smoke | nfcorpus | diagnostic | smoke_synthetic | False | False | code-path smoke only, not benchmark evidence |
| nfcorpus_confidence_gate_sweep | nfcorpus | diagnostic | smoke_synthetic | False | False | code-path smoke only, not benchmark evidence |
| experiment_dashboard | unknown | evidence_dashboard | smoke_synthetic | False | False | code-path smoke only, not benchmark evidence |
| hotpot_fqi_diagnostics_summary | hotpot | rl_extension | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_fqi_trace_distribution | hotpot | rl_extension | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_multistep_action_traces | hotpot | paper_asset | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_multistep_fqi_detailed | hotpot | rl_extension | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_multistep_fqi_metadata | hotpot | rl_extension | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_multistep_fqi_summary | hotpot | rl_extension | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_multistep_metrics_figure | hotpot | paper_asset | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_policy_reader_comparison_detailed | hotpot | reader_smoke | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_policy_reader_comparison_summary | hotpot | reader_smoke | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_reader_realdata_200_summary | hotpot | reader_smoke | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_reader_realdata_summary | hotpot | reader_smoke | tiny_realdata | False | False | small or non-final real-data run |
| hotpot_retrieval_policy_summary | hotpot | main_result | tiny_realdata | False | False | small or non-final real-data run |
| nfcorpus_budget_curve | nfcorpus | budget_curve | tiny_realdata | False | False | small or non-final real-data run |
| nfcorpus_constrained_policy_sweep | nfcorpus | constrained_bandit | tiny_realdata | False | False | small or non-final real-data run |
| nfcorpus_ope_diagnostics | nfcorpus | off_policy_evaluation | tiny_realdata | False | False | small or non-final real-data run |
| nfcorpus_policy_diagnostics | nfcorpus | diagnostic | tiny_realdata | False | False | small or non-final real-data run |
| nq_policy_reader_comparison_detailed | nq | reader_smoke | tiny_realdata | False | False | small or non-final real-data run |
| nq_policy_reader_comparison_summary | nq | reader_smoke | tiny_realdata | False | False | small or non-final real-data run |
| nq_reader_realdata_summary | nq | reader_smoke | tiny_realdata | False | False | small or non-final real-data run |
| scifact_budget_curve | scifact | budget_curve | tiny_realdata | False | False | small or non-final real-data run |
| scifact_constrained_policy_sweep | scifact | constrained_bandit | tiny_realdata | False | False | small or non-final real-data run |
| scifact_ope_diagnostics | scifact | off_policy_evaluation | tiny_realdata | False | False | small or non-final real-data run |
| scifact_policy_diagnostics | scifact | diagnostic | tiny_realdata | False | False | small or non-final real-data run |
