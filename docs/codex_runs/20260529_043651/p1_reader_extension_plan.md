# P1 Reader Extension Plan

## Goal

Add a tested, lightweight downstream QA evaluation path without changing the
main retrieval-stage claims.

## Design

- Add `src/selective_rag_rl/answer_metrics.py` with SQuAD-style normalization,
  exact match, and token F1.
- Add `src/selective_rag_rl/reader.py` with:
  - `ReaderPrediction`
  - `LexicalOverlapReader`, a deterministic local reader that selects the
    sentence with the highest lexical overlap with the question.
- Add `scripts/run_reader_eval.py` with:
  - `--dataset toy` default, requiring no raw data;
  - optional `hotpot` and `nq` modes that require local raw files;
  - BM25 retrieval and lexical reader evaluation;
  - detailed CSV and summary JSON/CSV outputs.
- Add `docs/READER_EXTENSION.md` that states the reader is a smoke downstream
  QA check, not evidence for replacing the retrieval-stage final claims.

## Tests First

1. `tests/test_answer_metrics.py`
   - normalization removes case, punctuation, and articles;
   - exact match handles normalized equivalence;
   - token F1 handles partial overlap and empty strings.
2. `tests/test_reader.py`
   - lexical reader selects a passage sentence containing overlapping evidence;
   - empty passages return an empty prediction.
3. `tests/test_reader_eval.py`
   - toy-mode script writes detailed and summary outputs with EM/F1 columns.

## Validation

- `uv run pytest tests/test_answer_metrics.py tests/test_reader.py tests/test_reader_eval.py -q`
- `uv run python scripts/run_reader_eval.py --dataset toy --output-dir outputs/codex_reader_smoke`
- `uv run pytest -q`

## Claim Boundary

This milestone adds evaluation plumbing only. It does not justify claiming
end-to-end generated-answer improvement unless full-data reader runs are later
executed and documented.
