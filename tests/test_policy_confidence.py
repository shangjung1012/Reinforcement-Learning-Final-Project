from __future__ import annotations

import numpy as np

from selective_rag_rl.policies.bandit import DirectMethodBandit, KnnDirectMethodBandit, SklearnRegressorBandit
from selective_rag_rl.policies.policy_confidence import confidence_gated_action, prediction_margin


def test_direct_method_bandit_exposes_predicted_action_scores() -> None:
    actions = ["left", "right"]
    features = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=float)
    rewards = {
        "left": [1.0, 0.8, 0.2, 0.0],
        "right": [0.0, 0.2, 0.8, 1.0],
    }
    policy = DirectMethodBandit(actions=actions, l2=0.1)
    policy.fit(features, rewards)

    scores = policy.predict_scores(np.asarray([1.0, 3.0], dtype=float))

    assert set(scores) == set(actions)
    assert policy.predict(np.asarray([1.0, 3.0], dtype=float)) == max(
        actions,
        key=lambda action: (scores[action], -actions.index(action)),
    )


def test_knn_bandit_exposes_predicted_action_scores() -> None:
    actions = ["left", "right"]
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    rewards = {
        "left": [1.0, 0.5, 0.0],
        "right": [0.0, 0.5, 1.0],
    }
    policy = KnnDirectMethodBandit(actions=actions, k=1)
    policy.fit(features, rewards)

    scores = policy.predict_scores(np.asarray([2.0], dtype=float))

    assert set(scores) == set(actions)
    assert policy.predict(np.asarray([2.0], dtype=float)) == "right"
    assert scores["right"] > scores["left"]


def test_sklearn_bandit_exposes_predicted_action_scores() -> None:
    from sklearn.tree import DecisionTreeRegressor

    actions = ["left", "right"]
    features = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    rewards = {
        "left": [1.0, 0.5, 0.0],
        "right": [0.0, 0.5, 1.0],
    }
    policy = SklearnRegressorBandit(actions=actions, estimator_factory=lambda: DecisionTreeRegressor(random_state=0))
    policy.fit(features, rewards)

    scores = policy.predict_scores(np.asarray([2.0], dtype=float))

    assert set(scores) == set(actions)
    assert policy.predict(np.asarray([2.0], dtype=float)) == "right"
    assert scores["right"] > scores["left"]


def test_prediction_margin_uses_policy_scores_with_existing_tie_break() -> None:
    actions = ["keep", "rewrite", "hybrid"]
    selected, margin = prediction_margin(
        {"keep": 0.4, "rewrite": 0.7, "hybrid": 0.7},
        actions,
    )

    assert selected == "rewrite"
    assert margin == 0.0


def test_confidence_gated_action_falls_back_only_on_small_predicted_margin() -> None:
    actions = ["keep", "rewrite", "hybrid"]

    low_confidence = confidence_gated_action(
        scores={"keep": 0.55, "rewrite": 0.56, "hybrid": 0.2},
        actions=actions,
        fallback_action="keep",
        min_margin=0.02,
    )
    high_confidence = confidence_gated_action(
        scores={"keep": 0.55, "rewrite": 0.58, "hybrid": 0.2},
        actions=actions,
        fallback_action="keep",
        min_margin=0.02,
    )

    assert low_confidence.action == "keep"
    assert low_confidence.policy_action == "rewrite"
    assert low_confidence.fallback_used is True
    assert low_confidence.predicted_margin < 0.02
    assert high_confidence.action == "rewrite"
    assert high_confidence.fallback_used is False
    assert high_confidence.predicted_margin >= 0.02
