# Gemini Pilot Report

## Scope

A tiny synthetic HotpotQA-style Gemini rewrite/decomposition pilot was run to
validate the budget gate and cache-resumable API path. This is an API pilot, not
benchmark evidence.

## Commands

Dry-run before live calls:

```powershell
uv run python scripts/run_gemini_baseline.py --data-path outputs/codex_smoke_second/fixtures/synthetic_hotpot.json --num-examples 4 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_synthetic.jsonl --output-dir outputs/codex_gemini_pilot/synthetic_4 --dry-run
```

Bounded live run:

```powershell
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_gemini_baseline.py --data-path outputs/codex_smoke_second/fixtures/synthetic_hotpot.json --num-examples 4 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_synthetic.jsonl --output-dir outputs/codex_gemini_pilot/synthetic_4 --allow-api --max-new-calls 4
```

Cache check after live run:

```powershell
uv run python scripts/run_gemini_baseline.py --data-path outputs/codex_smoke_second/fixtures/synthetic_hotpot.json --num-examples 4 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_synthetic.jsonl --output-dir outputs/codex_gemini_pilot/synthetic_4_cache_check --dry-run
```

## Results

| Step | Cache hits | Cache misses | Actual new Gemini calls | Result |
| --- | ---: | ---: | ---: | --- |
| dry-run before live | 0 | 4 | 0 | budget estimate only |
| bounded live run | 0 | 4 | 4 | succeeded |
| dry-run cache check | 4 | 0 | 0 | cache-resumable |

Live pilot summary:

| Method | Recall@5 | MRR | nDCG@5 | Reward | Rewrite cost | Retrieval calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini rewrite-all | 1.0 | 1.0 | 1.0 | 0.44 | 1.06 | 1.0 |
| Gemini decompose | 1.0 | 1.0 | 1.0 | 0.46 | 1.04 | 2.0 |

## Claim Boundary

This pilot used synthetic data and only two held-out examples. It validates API
access, budget enforcement, and cache reuse. It does not support any final claim
about Gemini-generated rewrites improving retrieval or answer quality.
