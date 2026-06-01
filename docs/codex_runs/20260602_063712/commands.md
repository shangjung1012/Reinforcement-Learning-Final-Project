# Commands

This log records commands that matter for reproducibility. It intentionally omits `.env` values and credential content.

```powershell
git switch -c codex/overnight-improvements-20260602-0636
uv run pytest -q
uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight_overnight
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight_overnight
uv run python scripts/run_retrieval_policy_scifact.py --num-train-examples 30 --num-test-examples 30 --seed 42 --full-corpus --embedder fake --policy-model ridge --tuning-folds 2 --knn-k-candidates 1 --output-dir outputs/codex_realdata_smoke_overnight/scifact_fake
uv run python scripts/run_retrieval_policy_nfcorpus.py --num-train-examples 30 --num-test-examples 30 --seed 42 --full-corpus --embedder fake --policy-model ridge --tuning-folds 2 --knn-k-candidates 1 --output-dir outputs/codex_realdata_smoke_overnight/nfcorpus_fake
uv run python scripts/run_validation_guardrail.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/codex_diagnostics_overnight/scifact_validation_guardrail.csv
uv run python scripts/run_validation_guardrail.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/codex_diagnostics_overnight/nfcorpus_validation_guardrail.csv
uv run python scripts/run_cost_frontier_summary.py --dataset scifact --summary-csv outputs/results/scifact_retrieval_policy_summary.csv --output-csv outputs/codex_diagnostics_overnight/scifact_cost_frontier.csv --budgets 1.0,1.25,1.5,2.0
uv run python scripts/run_cost_frontier_summary.py --dataset nfcorpus --summary-csv outputs/results/nfcorpus_retrieval_policy_summary.csv --output-csv outputs/codex_diagnostics_overnight/nfcorpus_cost_frontier.csv --budgets 1.0,1.25,1.5,2.0
uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_overnight
uv run pytest tests/test_retrieval_policy_experiment.py::test_semantic_rank_agreement_handles_constant_scores_without_warning tests/test_retrieval_policy_experiment.py::test_semantic_state_features_can_use_deeper_rank_profile -q
uv run pytest tests/test_experiment_dashboard.py -q
uv run python scripts/run_experiment_dashboard.py --input outputs/results --input outputs/codex_smoke_overnight --input outputs/codex_api_preflight_overnight --input outputs/codex_realdata_smoke_overnight --input outputs/codex_reader_smoke_overnight --output-csv outputs/codex_diagnostics_overnight/experiment_dashboard.csv --output-md docs/EXPERIMENT_DASHBOARD.md
uv run pytest -q
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_code_hardening --pytest-mode targeted
```
