# Commands

Commands are recorded as the run proceeds. Results are summarized in
`status.md`.

```powershell
pwd
git status --short
git branch --show-current
git log --oneline -5
Get-ChildItem -File -Recurse -Depth 2 | Sort-Object FullName | ForEach-Object { $_.FullName.Substring((Get-Location).Path.Length + 1) } | Select-Object -First 200
git switch -c codex/autonomous-improvements-20260529-0436
git remote -v
uv sync
uv run pytest -q
uv run pytest tests/test_artifact_index.py::test_export_artifact_index_records_existing_and_missing_files -q
uv run pytest tests/test_core.py::test_hotpot_loader_reads_examples -q
uv run pytest -q
uv run pytest tests/test_final_smoke.py -q
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke --pytest-mode skip
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke
uv run pytest tests/test_answer_metrics.py tests/test_reader.py tests/test_reader_eval.py -q
uv run python scripts/run_reader_eval.py --help
uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke
uv run pytest tests/test_bandit_baselines.py::test_linucb_history_records_chosen_action_reward_not_oracle_reward tests/test_off_policy_evaluation.py::test_estimate_off_policy_value_reports_no_coverage_when_actions_never_match -q
uv run pytest tests/test_off_policy_evaluation.py tests/test_bandit_baselines.py -q
uv run pytest -q
git status --short
git log --oneline -6
```
