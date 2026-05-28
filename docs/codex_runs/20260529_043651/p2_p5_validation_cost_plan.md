# P2/P5 Validation And Cost Documentation Plan

## Goal

Clarify how a reviewer should interpret validation-selected policies, semantic
feature experiments, OPE diagnostics, and proxy cost-aware rewards without
changing the final benchmark claims.

## Planned Changes

1. Add `docs/VALIDATION_PROTOCOL.md` with data-split rules, fixed-action
   baselines, selection guardrails, semantic-feature caution, OPE diagnostics,
   and reporting boundaries.
2. Add `docs/COST_MODEL.md` with the retrieval-stage reward, rewrite cost,
   retrieval-call cost, constrained-policy interpretation, and smoke/full-data
   limitations.
3. Link both documents from the README quickstart/scope section.
4. Update the autonomous run logs and validate with the default test suite.

## Definition Of Done

- Docs exist and keep claims conservative.
- README points reviewers to the new interpretation guides.
- Full pytest passes after the docs-only change.
