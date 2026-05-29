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
```
