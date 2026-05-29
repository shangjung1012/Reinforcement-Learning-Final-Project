# Second Autonomous Run Initial Audit

## Branch And Commit

- Second-run branch: `codex/api-validation-improvements-20260529-2211`
- Continued from: `codex/autonomous-improvements-20260529-0436`
- Base commit: `809e372 docs: mark autonomous run complete`
- This is the second autonomous improvement run.

## Working Tree

- `git status --short` was clean before creating the second-run branch.
- Local `.env` exists and is ignored by git.
- Second-run smoke output directories are generated local artifacts and are now
  ignored by `.gitignore`.

## First-Run Artifacts Confirmed

- `scripts/run_final_smoke.py`
- `docs/FINAL_REPRODUCTION.md`
- `scripts/run_reader_eval.py`
- `src/selective_rag_rl/answer_metrics.py`
- `src/selective_rag_rl/reader.py`
- `docs/READER_EXTENSION.md`
- `docs/RL_FRAMING.md`
- `docs/VALIDATION_PROTOCOL.md`
- `docs/COST_MODEL.md`
- `docs/codex_runs/20260529_043651/final_report_for_human.md`

## Raw Data Availability

All expected raw-data paths were checked by existence only. No file contents
were printed.

| Path | Exists? |
| --- | --- |
| `data/raw/HotpotQA/hotpot_dev_distractor_v1.json` | no |
| `data/raw/HotpotQA/hotpot_dev_fullwiki_v1.json` | no |
| `data/raw/HotpotQA/hotpot_train_v1.1.json` | no |
| `data/raw/natural-questions/default/validation-00000-of-00007.parquet` | no |
| `data/raw/scifact/corpus.jsonl` | no |
| `data/raw/scifact/queries.jsonl` | no |
| `data/raw/scifact/qrels/train.tsv` | no |
| `data/raw/scifact/qrels/test.tsv` | no |
| `data/raw/nfcorpus/corpus.jsonl` | no |
| `data/raw/nfcorpus/queries.jsonl` | no |
| `data/raw/nfcorpus/qrels/train.tsv` | no |
| `data/raw/nfcorpus/qrels/dev.tsv` | no |
| `data/raw/nfcorpus/qrels/test.tsv` | no |

## Environment And API Availability

The `.env` file was checked by variable names only. Values were not printed.

| Variable | Status |
| --- | --- |
| `GOOGLE_GENAI_USE_VERTEXAI` | present |
| `GOOGLE_CLOUD_PROJECT` | present |
| `GOOGLE_CLOUD_LOCATION` | present |
| `GOOGLE_APPLICATION_CREDENTIALS` | present |
| `GEMINI_MODEL` | present |

Credential file existence was checked. The credential basename is
`application_default_credentials.json`; the file exists locally. The full path
and file contents were not written to this audit.

## Baseline Validation

- `uv sync` passed.
- `uv run pytest -q` passed with `151 passed, 1 warning`.
- `uv run python scripts/run_final_smoke.py --output-dir outputs/codex_smoke_second --pytest-mode targeted` passed with nested targeted pytest `18 passed`.
- `uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_second` passed.

## Dependency And Documentation Notes

- Google GenAI SDK documentation was checked with Context7 for Vertex AI
  environment variables, `genai.Client`, `models.generate_content`, and
  `models.embed_content`.
- Current repo code uses explicit `genai.Client(vertexai=True, project=...,
  location=...)`, which is consistent with the checked docs.

## Prioritized Backlog

1. P0: Implement no-cost API preflight with explicit allow flag and tests.
2. P3: Implement data preflight to record missing raw-data blockers.
3. P1: Implement validation guardrail utility and CLI.
4. P2: Implement cost frontier utility and CLI.
5. P4/P5: Run tightly bounded API pilots only after dry-run preflight passes
   and explicit `CODEX_ALLOW_API_CALLS=1` is used.
6. P7/P8: Dashboard/final report if time remains.

## API Budget Plan

- Default scripts must do zero API calls.
- First allowed API preflight may make at most one Gemini call and one embedding
  text request.
- Hard cap for this run remains 30 new Gemini calls and 300 new embedding texts.
- Generated outputs and caches must remain ignored by git.
