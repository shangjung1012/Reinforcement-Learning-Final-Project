# API Preflight Log

| Time | Provider | Mode | Allow API? | New calls/texts allowed | Estimated misses | Actual calls/texts | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-05-29 22:15 | Google GenAI | documentation check | no | 0 | 0 | 0 | Context7 docs checked; no project API call |
| 2026-05-29 22:27 | Gemini | dry-run preflight | no | 0 | 1 | 0 | `dry_run_no_api_call` |
| 2026-05-29 22:27 | Vertex embedding | dry-run preflight | no | 0 | 1 | 0 | `dry_run_no_api_call` |
| 2026-05-29 22:28 | Gemini | live preflight | yes | 1 | 1 | 1 | `api_call_succeeded` |
| 2026-05-29 22:28 | Vertex embedding | live preflight | yes | 1 | 1 | 1 | `api_call_succeeded` |
| 2026-05-29 23:24 | Gemini baseline | synthetic dry-run | no | 0 | 4 | 0 | budget estimate only |
| 2026-05-29 23:25 | Gemini baseline | synthetic live pilot | yes | 4 | 4 | 4 | succeeded |
| 2026-05-29 23:27 | Gemini baseline | synthetic cache check | no | 0 | 0 | 0 | cache-resumable |

## Environment Name Check

The local `.env` file exists. The following variable names were present:

- `GOOGLE_GENAI_USE_VERTEXAI`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GEMINI_MODEL`

Credential file existence was confirmed by basename only:
`application_default_credentials.json`.

## Local Output

The latest local preflight summary was written under
`outputs/codex_api_preflight/`, which is ignored by git.
