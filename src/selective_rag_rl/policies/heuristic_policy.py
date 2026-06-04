from __future__ import annotations

import numpy as np


def heuristic_retrieval_action(feature: np.ndarray, actions: list[str]) -> str:
    """Rule-based adaptive retrieval baseline using deployable state features."""
    if not actions:
        raise ValueError("actions must not be empty")
    question_length = float(feature[1]) if feature.size > 1 else 0.0
    bm25_top1 = float(feature[3]) if feature.size > 3 else 0.0
    bm25_gap = float(feature[4]) if feature.size > 4 else 0.0
    bm25_entropy = float(feature[5]) if feature.size > 5 else 0.0

    bm25_confident = bm25_top1 >= 0.35 and bm25_gap >= 0.20 and bm25_entropy <= 0.45
    long_query = question_length >= 0.35
    uncertain = bm25_top1 < 0.20 or bm25_gap < 0.05 or bm25_entropy >= 0.65

    if bm25_confident:
        return _first_available(actions, ["bm25_keep", "bm25_keyword"])
    if uncertain and long_query:
        return _first_available(actions, ["hybrid_keyword", "hybrid_keep", "dense_keyword", "dense_keep", "bm25_keyword"])
    if uncertain:
        return _first_available(actions, ["dense_keep", "dense_keyword", "hybrid_keep", "hybrid_keyword", "bm25_keep"])
    if long_query:
        return _first_available(actions, ["bm25_keyword", "hybrid_keyword", "dense_keyword", "bm25_keep"])
    return _first_available(actions, ["bm25_keep", "dense_keep", "hybrid_keep"])


def _first_available(actions: list[str], preferences: list[str]) -> str:
    for action in preferences:
        if action in actions:
            return action
    return actions[0]
