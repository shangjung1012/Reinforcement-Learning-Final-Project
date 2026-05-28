# Literature-Grounded Generated Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HyDE, multi-query, and hybrid decomposition actions to the retrieval policy action space with cached Gemini generation and cost-aware evaluation.

**Architecture:** Extend the existing `RewriteProvider` and `GeminiCache` path instead of introducing a new LLM client. Add a generic generated-query action evaluator that can run BM25, dense, or hybrid retrieval over generated queries and keep existing base-action behavior stable.

**Tech Stack:** Python, pandas, pytest, uv, existing Vertex Gemini client, existing BM25/dense retrievers.

---

### Task 1: Generated Action Definitions

**Files:**
- Modify: `src/selective_rag_rl/retrieval_policy_experiment.py`
- Test: `tests/test_retrieval_policy_experiment.py`

- [ ] Add generated action constants for `hybrid_llm_decompose`, `bm25_hyde`, `dense_hyde`, `hybrid_hyde`, `bm25_multi_query`, and `hybrid_multi_query`.
- [ ] Extend method labels and LLM method order so summaries include generated actions.
- [ ] Update the LLM action test to expect the generated action family.
- [ ] Run `uv run pytest -q tests/test_retrieval_policy_experiment.py::test_run_llm_retrieval_policy_experiment_writes_cached_actions`.

### Task 2: Generated Retrieval Evaluator

**Files:**
- Modify: `src/selective_rag_rl/retrieval_policy_experiment.py`
- Test: `tests/test_retrieval_policy_experiment.py`

- [ ] Add a generic helper that accepts `mode` and `retriever_kind` and merges generated query results.
- [ ] Replace the BM25-only LLM helper calls with the generic helper for all generated actions.
- [ ] Add a unit test that calls `evaluate_retrieval_actions` with `bm25_hyde`, `dense_hyde`, `hybrid_hyde`, and `hybrid_multi_query`, using a fake provider.
- [ ] Verify hybrid generated actions have higher retrieval calls than BM25/dense generated actions.

### Task 3: Gemini Prompt Modes

**Files:**
- Modify: `src/selective_rag_rl/gemini_baseline.py`
- Test: `tests/test_gemini_baseline.py`

- [ ] Add prompts for `hyde` and `multi_query`.
- [ ] Update parsing so `rewrite` and `hyde` return one query, `decompose` returns up to two, and `multi_query` returns up to three.
- [ ] Add tests for parsing limits.

### Task 4: Smoke, Docs, and Verification

**Files:**
- Modify: `README.md`
- Optional outputs: `outputs/generated_action_smoke/`

- [ ] Add a README command for the generated-action Hotpot smoke.
- [ ] Run targeted tests.
- [ ] Run `uv run pytest -q && uv run python -m compileall -q src scripts && git diff --check`.
- [ ] Commit and push.

