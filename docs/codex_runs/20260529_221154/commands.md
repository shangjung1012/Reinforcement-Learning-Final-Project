# Commands

Commands are recorded as the run proceeds. Results are summarized in
`status.md`.

```powershell
pwd
git status --short
git branch --show-current
git log --oneline -12
git fetch origin
git checkout codex/autonomous-improvements-20260529-0436
git pull --ff-only
git checkout -b codex/api-validation-improvements-20260529-2211
uv sync
uv run pytest -q
uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_second --pytest-mode targeted
uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_second
npx ctx7@latest library "Google Gen AI Python SDK" "Vertex AI Python client environment variables generate_content embed_content google-genai"
npx ctx7@latest docs /googleapis/python-genai "Vertex AI Python genai.Client environment variables GOOGLE_GENAI_USE_VERTEXAI GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION generate_content embed_content"
npx ctx7@latest docs /googleapis/python-genai "Python examples client.models.generate_content and client.models.embed_content EmbedContentConfig task_type"
uv run pytest -q
uv run pytest tests/test_api_preflight.py -q
uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight
$env:CODEX_ALLOW_API_CALLS='1'; uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight --allow-api --max-new-gemini-calls 1 --max-new-embedding-texts 1
uv run pytest -q
uv run pytest tests/test_data_preflight.py -q
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight
uv run pytest -q
uv run pytest tests/test_validation_guardrail.py -q
uv run python scripts/run_validation_guardrail.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_validation_guardrail.csv
uv run python scripts/run_validation_guardrail.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_validation_guardrail.csv
uv run pytest -q
uv run pytest tests/test_cost_frontier.py -q
uv run python scripts/run_cost_frontier_summary.py --dataset scifact --summary-csv outputs/results/scifact_retrieval_policy_summary.csv --output-csv outputs/results/scifact_cost_frontier_summary.csv --budgets 1.0,1.25,1.5,2.0
uv run python scripts/run_cost_frontier_summary.py --dataset nfcorpus --summary-csv outputs/results/nfcorpus_retrieval_policy_summary.csv --output-csv outputs/results/nfcorpus_cost_frontier_summary.csv --budgets 1.0,1.25,1.5,2.0
uv run pytest -q
uv run pytest tests/test_gemini_baseline.py -q
uv run python scripts/run_gemini_baseline.py --data-path outputs/codex_smoke_second/fixtures/synthetic_hotpot.json --num-examples 4 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_synthetic.jsonl --output-dir outputs/codex_gemini_pilot/synthetic_4 --dry-run
$env:CODEX_ALLOW_API_CALLS='1'; uv run python scripts/run_gemini_baseline.py --data-path outputs/codex_smoke_second/fixtures/synthetic_hotpot.json --num-examples 4 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_synthetic.jsonl --output-dir outputs/codex_gemini_pilot/synthetic_4 --allow-api --max-new-calls 4
uv run python scripts/run_gemini_baseline.py --data-path outputs/codex_smoke_second/fixtures/synthetic_hotpot.json --num-examples 4 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_synthetic.jsonl --output-dir outputs/codex_gemini_pilot/synthetic_4_cache_check --dry-run
uv run pytest -q
```
