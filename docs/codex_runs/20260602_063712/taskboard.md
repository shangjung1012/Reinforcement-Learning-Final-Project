# Overnight Taskboard

## Current Goal

Continue after the clean main merge by validating local data/API readiness and adding only narrow, tested improvements.

## Ordered Backlog

- [x] Re-audit branch, baseline, ignored artifacts, and current evidence.
- [x] Run no-cost API preflight and record whether live API calls are blocked or available.
- [x] Run data preflight and identify which real datasets are usable locally.
- [x] Run tiny real-data smoke experiments for locally available datasets.
- [x] Run validation guardrail and cost frontier diagnostics on existing artifacts if inputs exist.
- [x] Fill one high-value documentation or test gap found during the audit.
- [x] Run full pytest and final smoke.
- [ ] Commit and push coherent changes.

## Out Of Scope Tonight

- No broad rewrite of `FINAL_REPORT.md`.
- No live API spending unless `CODEX_ALLOW_API_CALLS=1` is already present and budgeted.
- No raw-data or cache commits.
- No production claim changes from tiny smoke or API preflight evidence.

## Human Decisions Remaining

- Whether to spend more Gemini/Vertex quota.
- Whether to merge the overnight branch after reviewing the diff.
- Whether final report claims should be updated after any future full-scale rerun.
