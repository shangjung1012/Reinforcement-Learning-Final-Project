# Real-Data Smoke Report

## Summary

`scripts/run_data_preflight.py` was run against the local repository. It found
that required raw-data files are absent, so no real-data smoke experiments were
launched in this milestone.

## Command

```powershell
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight
```

## Result

- `all_required_available`: false
- required missing paths: 11
- HotpotQA: blocked by missing required dev distractor file
- Natural Questions: blocked by missing validation parquet shard
- SciFact: blocked by missing corpus, queries, train qrels, and test qrels
- NFCorpus: blocked by missing corpus, queries, train/dev/test qrels

The local machine-readable outputs are under `outputs/codex_data_preflight/`,
which is ignored by git.

## Follow-Up

After raw data is downloaded under the README layout, run the small fake-embedder
real-data smoke commands from the second-run prompt before attempting any
expensive API-backed runs.
