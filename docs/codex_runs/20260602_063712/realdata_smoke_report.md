# Real-Data Smoke Report

## Evidence Level

`tiny_realdata`

These runs use real local SciFact and NFCorpus files, full corpus retrieval, fake embeddings, and only 30 train / 30 test examples. They verify integration and local data readiness. They are not final benchmark evidence.

## Data Availability

| Dataset | Status | Notes |
| --- | --- | --- |
| SciFact | available | corpus, queries, train qrels, and test qrels found. |
| NFCorpus | available | corpus, queries, train/dev/test qrels found. |
| HotpotQA | blocked_missing_data | required dev distractor JSON not found. |
| Natural Questions | blocked_missing_data | required validation parquet not found. |

## Commands Run

```powershell
uv run python scripts/run_retrieval_policy_scifact.py --num-train-examples 30 --num-test-examples 30 --seed 42 --full-corpus --embedder fake --policy-model ridge --tuning-folds 2 --knn-k-candidates 1 --output-dir outputs/codex_realdata_smoke_overnight/scifact_fake
uv run python scripts/run_retrieval_policy_nfcorpus.py --num-train-examples 30 --num-test-examples 30 --seed 42 --full-corpus --embedder fake --policy-model ridge --tuning-folds 2 --knn-k-candidates 1 --output-dir outputs/codex_realdata_smoke_overnight/nfcorpus_fake
```

## Key Summary Rows

| Dataset | Method | Recall@5 | MRR | Reward | Retrieval calls |
| --- | --- | ---: | ---: | ---: | ---: |
| SciFact | Train-best retrieval action | 0.738889 | 0.645556 | 1.061667 | 1.0 |
| SciFact | Selective retrieval policy | 0.738889 | 0.640000 | 1.058889 | 1.0 |
| NFCorpus | Train-best retrieval action | 0.149832 | 0.502778 | 0.401221 | 1.0 |
| NFCorpus | Selective retrieval policy | 0.149832 | 0.502778 | 0.401221 | 1.0 |

## Interpretation

The tiny smoke confirms that the checked-in scripts can read the local BEIR-format datasets and complete a full-corpus fake-embedder policy run. The sample is too small for final claims. On this smoke, the learned policy ties or nearly ties train-best fixed action, which is acceptable as an integration check but not a new result claim.
