from __future__ import annotations

import numpy as np

from selective_rag_rl.policies.heuristic_policy import heuristic_retrieval_action


def test_heuristic_retrieval_action_uses_bm25_when_initial_rank_is_confident() -> None:
    actions = ["bm25_keep", "dense_keep", "hybrid_keyword"]
    confident_bm25 = np.asarray([1.0, 0.2, 0.0, 0.8, 0.6, 0.1], dtype=float)

    assert heuristic_retrieval_action(confident_bm25, actions) == "bm25_keep"


def test_heuristic_retrieval_action_uses_hybrid_for_uncertain_long_query() -> None:
    actions = ["bm25_keep", "bm25_keyword", "dense_keyword", "hybrid_keyword"]
    uncertain_long_query = np.asarray([1.0, 0.8, 0.0, 0.1, 0.02, 0.9], dtype=float)

    assert heuristic_retrieval_action(uncertain_long_query, actions) == "hybrid_keyword"


def test_heuristic_retrieval_action_falls_back_to_available_dense_action() -> None:
    actions = ["bm25_keep", "dense_keep"]
    uncertain_short_query = np.asarray([1.0, 0.1, 0.0, 0.05, 0.01, 0.8], dtype=float)

    assert heuristic_retrieval_action(uncertain_short_query, actions) == "dense_keep"
