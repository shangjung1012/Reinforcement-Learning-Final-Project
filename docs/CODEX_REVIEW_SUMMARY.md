# Codex Review Summary

This document summarizes the cleanup and continuation work after the selective merge into `main`.

## What Is Now Stronger

- Reproducibility has a raw-data-free smoke path through `scripts/run_final_smoke.py`.
- API use is behind explicit preflight, allow flags, and budget controls.
- Local `.env` can be checked without printing secrets.
- SciFact and NFCorpus local data availability can be checked through `scripts/run_data_preflight.py`.
- Validation guardrail and cost frontier utilities are executable and tested.
- Tiny real-data smoke now verifies the local SciFact/NFCorpus pipeline without external APIs.

## Latest Overnight Evidence

| Evidence | Status | Claim Boundary |
| --- | --- | --- |
| Full pytest | 176 passed, 1 warning | Code health only. |
| API preflight | Dry-run passed, 0 live calls | Credentials/config are present; no quality claim. |
| Data preflight | SciFact/NFCorpus available, HotpotQA/NQ missing | Local data readiness only. |
| Tiny SciFact smoke | 30 train / 30 test completed | Integration evidence, not final benchmark. |
| Tiny NFCorpus smoke | 30 train / 30 test completed | Integration evidence, not final benchmark. |
| Toy reader smoke | Completed with lexical reader | Downstream metric plumbing only. |
| Guardrail diagnostics | Completed on final detailed CSVs | Analysis-only where no validation split exists. |
| Cost frontier diagnostics | Completed on final summary CSVs | Defense aid, not new benchmark evidence. |

## What To Claim

The conservative final claim remains:

> A lightweight offline contextual-bandit policy can improve cost-aware retrieval-stage RAG performance over strong fixed-action baselines on SciFact and NFCorpus, with bootstrap, OPE, selected-action baseline, constrained utility, and robustness diagnostics.

## What Not To Claim Yet

- Do not claim full RAG answer-generation improvement.
- Do not claim Gemini-generated rewrites improve final benchmarks from API preflight alone.
- Do not claim Vertex semantic features are final-best without repeated validation support.
- Do not claim online deployment or real logged OPE.

## Remaining Work

1. Decide whether to run a live one-call Gemini/Vertex smoke by setting an explicit allow flag and keeping budgets at 1.
2. Add HotpotQA and NQ raw files if downstream reader smoke on real QA datasets is needed.
3. Run repeated-seed full-data experiments only if final report evidence must be regenerated.
4. Review whether `FINAL_REPORT.md` should remain unchanged or receive a short appendix pointing to the new reproducibility/API docs.
5. Review and merge only the useful overnight branch changes; generated `outputs/codex_*` artifacts should stay local.
