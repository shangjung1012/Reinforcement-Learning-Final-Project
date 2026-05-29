# Experiment Dashboard

This dashboard classifies generated artifacts by evidence level so smoke
and API pilots are not confused with final benchmark evidence.

## Evidence Counts

| Evidence level | Count |
| --- | ---: |
| `api_pilot` | 5 |
| `api_preflight` | 2 |
| `blocked_budget_exceeded` | 0 |
| `blocked_missing_credentials` | 0 |
| `blocked_missing_data` | 2 |
| `final_claim` | 6 |
| `full_benchmark` | 181 |
| `smoke_synthetic` | 16 |
| `smoke_toy_reader` | 6 |
| `tiny_realdata` | 0 |

## Claim Boundary

Only rows marked `full_benchmark` or `final_claim` may support final
retrieval-stage claims. API pilot and smoke rows are integration evidence
only.
