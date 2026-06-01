# API Preflight Log

| Time | Provider | Mode | Allow API? | New calls/texts allowed | Estimated misses | Actual calls/texts | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-06-02 06:40 | Gemini | dry-run preflight | false | 0 | 1 | 0 | `dry_run_no_api_call` |
| 2026-06-02 06:40 | Vertex embedding | dry-run preflight | false | 0 | 1 | 0 | `dry_run_no_api_call` |

## Environment Summary

- `.env` file exists.
- Required variable names are present: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`.
- Optional variable names are present: `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_LOCATION`, `GEMINI_MODEL`.
- Credential file exists; public logs record only the basename.
- `CODEX_ALLOW_API_CALLS` is not set, so this run made no live API calls.

## Interpretation

The local configuration is ready for a bounded API smoke from a file/variable-presence perspective. A live Gemini or Vertex call still requires an explicit allow flag and a strict budget. This dry-run does not support any model-quality claim.
