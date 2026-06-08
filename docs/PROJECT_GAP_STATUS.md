# Project Gap Status

This file tracks the main remaining gaps after the code-quality and evidence
dashboard fixes.

## Fixed In This Pass

- Pytest warning removed: semantic rank-agreement features now avoid undefined
  Spearman correlation calls when semantic scores are constant.
- Evidence dashboard added: `scripts/run_experiment_dashboard.py` writes
  `outputs/results/experiment_dashboard.csv` and `docs/EXPERIMENT_DASHBOARD.md`.
- Poster claim audit added: `docs/POSTER_CLAIM_AUDIT.md` confirms the poster
  claim boundary is aligned with the retrieval-stage evidence.
- Bandit replay-regret diagnostics added:
  `scripts/run_bandit_replay_diagnostics.py` writes full-corpus
  selected-action replay summaries and regret figures for SciFact and NFCorpus,
  making the distinction between full-information direct-method learning and
  chosen-action feedback explicit.
- API live preflight now succeeds under the updated local `.env`: one bounded
  Gemini generation call and one bounded Vertex embedding text request both
  completed.
- A span heuristic reader was added alongside the lexical reader, plus
  `scripts/run_reader_comparison.py` for explicit toy/real-data reader
  comparisons.
- FQI post-hoc diagnostics were added:
  `scripts/run_fqi_diagnostics.py` writes the reward/call gap against
  train-best fixed trace and the selected trace distribution.
- A Windows-friendly missing raw-data downloader was added:
  `scripts/download_missing_raw_data.py` can fetch HotpotQA dev distractor from
  a Hugging Face parquet mirror and convert it into the existing JSON loader
  schema.
- HotpotQA real-data reader comparison and a bounded Gemini baseline pilot were
  run after the missing HotpotQA dev distractor file was restored locally.
- The Natural Questions validation parquet shard was restored from Hugging Face
  Hub through the same missing-data downloader.
- An answer-type heuristic reader was added, and larger HotpotQA plus small NQ
  reader comparisons were run as tiny real-data diagnostics.
- A repeated-seed Gemini baseline pilot and a tiny repeated-seed Vertex semantic
  pilot were run under explicit cache/budget gates.
- A bounded Gemini answer-reader pilot was run on HotpotQA and Natural
  Questions, providing the first non-deterministic downstream QA signal while
  keeping the evidence labeled as `api_pilot`.
- Policy-routed deterministic reader diagnostics were added for HotpotQA and
  Natural Questions. They compare BM25, train-best fixed retrieval, and learned
  retrieval-policy outputs under lexical/span/answer-type heuristic readers
  without API calls.

## Data Availability

Checked with:

```bash
uv run python scripts/run_data_preflight.py --output-dir outputs/codex_data_preflight
```

Current local status:

- SciFact is available: 5,183 corpus rows, 1,109 queries, 919 train qrels, 339
  test qrels.
- NFCorpus is available: 3,633 corpus rows, 3,237 queries, 110,575 train qrels,
  11,385 dev qrels, 12,334 test qrels.
- HotpotQA dev distractor is available: 7,405 rows. It was restored with:

  ```bash
  uv run python scripts/download_missing_raw_data.py --dataset hotpot-dev-distractor --prefer-hf --output-dir outputs/codex_data_download_hotpot_hf
  ```

- Natural Questions validation shard is available: 1,119 rows. It was restored
  with:

  ```bash
  uv run python scripts/download_missing_raw_data.py --dataset nq-validation-shard --prefer-hf --output-dir outputs/codex_nq_download
  ```

Small full-corpus smoke runs were completed for SciFact and NFCorpus with fake
embeddings and 30 train / 30 test examples. These runs validate real-data code
paths only; they are not final benchmark replacements.

## API Availability

Checked with:

```bash
uv run python scripts/run_api_preflight.py --provider all --output-dir outputs/codex_api_preflight
```

The dry-run preflight sees `.env`, the required variable names, optional Vertex
settings, and the credential file basename. It made zero Gemini calls and zero
Vertex embedding requests.

A bounded live API preflight was also attempted with both
`CODEX_ALLOW_API_CALLS=1` and `--allow-api`, capped at one Gemini call and one
Vertex embedding text. Under the updated local Google Cloud project, both
providers succeeded. This establishes API reachability only; it does not change
the final benchmark claim.

A bounded HotpotQA Gemini baseline pilot was run after HotpotQA became
available:

```bash
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_gemini_baseline.py --data-path data/raw/HotpotQA/hotpot_dev_distractor_v1.json --num-examples 10 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_realdata.jsonl --allow-api --max-new-calls 8 --output-dir outputs/codex_gemini_realdata_pilot
```

The dry-run estimated 8 cache misses, and the live pilot made 8 new Gemini
calls. On the 4 held-out examples, Gemini rewrite-all and decompose both reached
Recall@5 1.0 and MRR 1.0, but cost-aware reward was only 0.4175 and 0.4000 due
to high rewrite/retrieval cost. This is `api_pilot` evidence only.

A repeated-seed Gemini pilot was then run on seeds 41, 42, and 43 with 10
examples per seed. The live run used the existing cache: 24 cache hits and 0
new calls in this pass. Across 12 held-out examples, rewrite-all reached mean
Recall@5 0.75 and reward 0.085833; decompose reached mean Recall@5 0.833333
and reward 0.177083. This is still `api_pilot` evidence because the sample is
tiny and generated-action policy selection has not been validated.

Vertex semantic feature pilots were run on tiny SciFact/NFCorpus splits:

- NFCorpus 10/10: selective reward tied train-best fixed at 0.212549.
- SciFact 10/10: selective reward tied train-best fixed at 0.900000.
- NFCorpus 30/30: selective reward 0.410110 versus train-best fixed 0.401221.
- SciFact 30/30: selective reward 1.058889 versus train-best fixed 1.061667.

These are API pilots, not final evidence. They show the path runs and that the
signal is mixed at tiny scale.

A smaller repeated-seed Vertex semantic pilot was also run on NFCorpus with 10
train / 10 test examples, depth 3, ridge policy, and `full` versus
`no_semantic` feature sets. The run wrote 208 new cached embedding texts. The
validation-selected configuration matched heldout-best in all three seeds, but
the guardrail fallback rate was 1.0 because the selected policies were
dominated/tied by the train-best fixed action. This is a useful limitation
diagnostic, not semantic-feature win evidence.

## Reader Smoke

Checked with:

```bash
uv run python scripts/run_reader_eval.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_smoke_phase
```

The deterministic lexical reader smoke completed with exact match 0.0 and token
F1 0.490079. The span heuristic reader improves the synthetic toy fixture:

```bash
uv run python scripts/run_reader_comparison.py --dataset toy --num-examples 4 --output-dir outputs/codex_reader_comparison_api_run
```

Toy comparison result:

- lexical reader: exact match 0.0, token F1 0.490079;
- span reader: exact match 1.0, token F1 1.0.

This validates stronger downstream metric plumbing on a synthetic fixture only;
it is not evidence for answer-generation quality on HotpotQA or NQ.

After restoring HotpotQA, a tiny real-data reader comparison was run:

```bash
uv run python scripts/run_reader_comparison.py --dataset hotpot --num-examples 50 --readers lexical,span --output-dir outputs/codex_reader_hotpot_realdata_50
```

Result:

- lexical reader: exact match 0.0, token F1 0.046297, Recall@5 0.83;
- span reader: exact match 0.02, token F1 0.076500, Recall@5 0.83.

This is now real-data evidence, but it is still tiny and uses deterministic
heuristics. It should be treated as `tiny_realdata`, not final downstream RAG
answer-quality evidence.

A larger HotpotQA 200-example comparison and an NQ 50-example comparison now
exist:

```bash
uv run python scripts/run_reader_comparison.py --dataset hotpot --num-examples 200 --readers lexical,span,answer_type --output-dir outputs/codex_reader_hotpot_realdata_200
uv run python scripts/run_reader_comparison.py --dataset nq --num-examples 50 --readers lexical,span,answer_type --output-dir outputs/codex_reader_nq_realdata_50
```

HotpotQA 200 summary:

- lexical reader: exact match 0.0, token F1 0.050195, Recall@5 0.82;
- span reader: exact match 0.035, token F1 0.084691, Recall@5 0.82;
- answer-type reader: exact match 0.035, token F1 0.084691, Recall@5 0.82.

NQ 50 summary:

- lexical reader: exact match 0.0, token F1 0.038915, Recall@5 0.98;
- span reader: exact match 0.0, token F1 0.013333, Recall@5 0.98;
- answer-type reader: exact match 0.0, token F1 0.013333, Recall@5 0.98.

These results make the downstream gap more concrete: retrieval coverage can be
high while deterministic answer extraction remains weak. They do not support a
final downstream answer-quality claim.

A stronger bounded Gemini reader pilot was then run on the same downstream
metric path:

```bash
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_gemini_reader_eval.py --dataset hotpot --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_hotpot.jsonl --allow-api --max-new-calls 40 --output-dir outputs/codex_gemini_reader_hotpot_40
CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_gemini_reader_eval.py --dataset nq --num-examples 40 --cache-path outputs/cache/codex_gemini_reader_nq.jsonl --allow-api --max-new-calls 40 --output-dir outputs/codex_gemini_reader_nq_40
```

HotpotQA 40 summary:

- best deterministic baseline: exact match 0.0, token F1 0.062292;
- Gemini reader: exact match 0.575, token F1 0.628081.

NQ 40 summary:

- best deterministic baseline: exact match 0.0, token F1 0.029894;
- Gemini reader: exact match 0.225, token F1 0.265417.

This is a meaningful pilot improvement over deterministic readers, but it is
not yet a final answer-quality claim because it is single-seed, small, API
backed, BM25-only, and not compared against policy-routed retrieval.

The missing policy-routed deterministic reader comparison has now been run:

```bash
uv run python scripts/run_policy_reader_comparison.py --dataset hotpot --detailed-csv outputs/results/retrieval_policy_detailed.csv --num-examples 50 --readers lexical,span,answer_type --output-dir outputs/codex_policy_reader_hotpot_50 --publish-results
uv run python scripts/run_policy_reader_comparison.py --dataset nq --detailed-csv outputs/results/nq_retrieval_policy_detailed.csv --num-examples 50 --source-num-examples 500 --readers lexical,span,answer_type --output-dir outputs/codex_policy_reader_nq_50 --publish-results
```

HotpotQA 50 result:

- Vanilla BM25 + best deterministic reader: exact match 0.06, token F1
  0.120706, retrieval reward 1.249167.
- Train-best fixed retrieval + best deterministic reader: exact match 0.08,
  token F1 0.130706, retrieval reward 1.316167.
- Learned retrieval policy + best deterministic reader: exact match 0.06,
  token F1 0.120706, retrieval reward 1.346367.

NQ 50 result:

- Vanilla BM25 + answer-type reader: exact match 0.04, token F1 0.072857,
  retrieval reward 1.456667.
- Train-best fixed retrieval + answer-type reader: exact match 0.04, token F1
  0.072857, retrieval reward 1.495000.
- Learned retrieval policy + answer-type reader: exact match 0.04, token F1
  0.072857, retrieval reward 1.495000.

This is a useful diagnostic because it connects retrieval-policy outputs to
EM/F1 plumbing. It is still not an answer-quality claim: deterministic readers
remain too weak, and the policy-routed Gemini-reader comparison has not been
run.

## FQI Extension Status

Checked with:

```bash
uv run python scripts/run_fqi_diagnostics.py --dataset hotpot --detailed-csv outputs/results/multistep_detailed.csv --summary-csv outputs/results/hotpot_fqi_diagnostics_summary.csv --trace-csv outputs/results/hotpot_fqi_trace_distribution.csv --split test
```

The existing HotpotQA FQI artifact has 240 test examples. Multi-step FQI reward
is 1.222521, below the train-best fixed trace at 1.264444, while the two-step
oracle reaches 1.376264. FQI also averages 2.4375 retrieval calls versus 2.0 for
the train-best fixed trace. This is a useful RL extension and failure analysis,
but it should not be presented as a main win.

## Remaining Work

- Scale the Gemini reader or a locally cached extractive QA reader to larger
  HotpotQA/NQ splits with repeated seeds before making any answer-quality claim.
- Compare the stronger Gemini reader under fixed BM25, train-best fixed
  retrieval, and the learned retrieval-action policy before connecting
  downstream QA to the RL policy.
- Keep Gemini/Vertex pilots labeled as API pilots until larger repeated-seed
  guardrail checks support them against fixed-action and policy baselines.
- Keep the deterministic reader paths labeled as smoke unless a stronger reader
  experiment is run on real data with baselines.
- Treat FQI as an extension/limitation unless a stronger state representation or
  value model beats train-best fixed trace under validation.
- Use `docs/EXPERIMENT_DASHBOARD.md` before updating final reports or the
  poster; rows with `claim_allowed=false` should remain analysis-only.
