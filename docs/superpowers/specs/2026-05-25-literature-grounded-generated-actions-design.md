# Literature-Grounded Generated Actions Design

## Goal

Expand the retrieval policy action space with non-trivial generated-query actions
that are grounded in RAG/query-rewriting literature and can be selected by the
existing cost-aware contextual bandit.

## Scope

This design adds a controlled action family rather than a large exhaustive
Cartesian product. The first implementation includes HyDE-style pseudo-document
retrieval, multi-query retrieval with rank fusion, and hybrid LLM decomposition.
Existing BM25/dense/hybrid keep and keyword actions remain unchanged.

## Action Set

The generated action family is:

- `bm25_llm_rewrite`
- `bm25_llm_decompose`
- `hybrid_llm_decompose`
- `bm25_hyde`
- `dense_hyde`
- `hybrid_hyde`
- `bm25_multi_query`
- `hybrid_multi_query`

These actions extend the existing six base retrieval actions. The resulting
generated action run has fourteen total actions.

## Retrieval Semantics

Generated query modes are produced through the existing `RewriteProvider`
interface and cached through `GeminiCache` by `(qid, mode)`.

- `rewrite`: one concise search query.
- `decompose`: up to two subqueries.
- `hyde`: one hypothetical evidence passage or pseudo-document.
- `multi_query`: up to three diverse search queries.

BM25 generated actions merge BM25 results across generated queries. Dense
generated actions merge dense results across generated queries. Hybrid generated
actions run both BM25 and dense for each generated query, then merge all results.

## Cost Model

Generated actions use:

```text
llm_base_cost + llm_token_cost * generated_query_tokens
+ retrieval_call_cost * max(retrieval_calls - 1, 0)
```

For hybrid generated actions, `retrieval_calls = 2 * number_of_generated_queries`.
For BM25/dense generated actions, `retrieval_calls = number_of_generated_queries`.

## Evaluation

The implementation must be testable without Vertex calls by injecting a fake
`rewrite_provider`. Smoke tests should verify that generated actions are cached,
appear in checkpoints, and compute different retrieval-call costs for BM25,
dense, and hybrid modes.

