from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from selective_rag_rl.core.retriever import RetrievalResult
from selective_rag_rl.core.text import capitalized_spans, content_tokens, entropy, wh_word

WH_TO_INDEX = {
    "who": 0,
    "what": 1,
    "when": 2,
    "where": 3,
    "which": 4,
    "why": 5,
    "how": 6,
    "yesno": 7,
}


def state_features(
    question: str,
    initial_results: list[RetrievalResult],
    semantic_features: list[float] | None = None,
) -> np.ndarray:
    scores = [r.score for r in initial_results[:5]]
    top1 = scores[0] if scores else 0.0
    gap = scores[0] - scores[1] if len(scores) > 1 else top1
    features = [
        1.0,
        min(len(content_tokens(question)), 30) / 30.0,
        min(len(capitalized_spans(question)), 5) / 5.0,
        min(max(top1, 0.0), 20.0) / 20.0,
        min(max(gap, 0.0), 20.0) / 20.0,
        entropy(scores) / 2.0,
    ]
    wh = [0.0] * len(WH_TO_INDEX)
    wh[WH_TO_INDEX.get(wh_word(question), WH_TO_INDEX["what"])] = 1.0
    semantic = semantic_features or []
    interactions = _lexical_semantic_interactions(features, semantic)
    return np.asarray([*features, *wh, *semantic, *interactions], dtype=float)


def _lexical_semantic_interactions(base_features: list[float], semantic_features: list[float]) -> list[float]:
    if len(semantic_features) < 9:
        return []
    bm25_top1 = base_features[3]
    bm25_gap = base_features[4]
    bm25_entropy = base_features[5]
    semantic_top1 = semantic_features[0]
    semantic_margin = semantic_features[3]
    semantic_positive_rate = semantic_features[4]
    semantic_std = semantic_features[5]
    semantic_spread = semantic_features[8]
    return [
        bm25_top1 * semantic_top1,
        bm25_gap * semantic_margin,
        bm25_entropy * semantic_std,
        bm25_top1 * semantic_spread,
        bm25_gap * semantic_spread,
        bm25_entropy * semantic_positive_rate,
    ]


@dataclass
class DirectMethodBandit:
    actions: list[str]
    l2: float = 1.0

    def fit(self, features: np.ndarray, rewards: dict[str, list[float]]) -> None:
        xtx = features.T @ features
        reg = self.l2 * np.eye(features.shape[1])
        self.weights: dict[str, np.ndarray] = {}
        for action in self.actions:
            y = np.asarray(rewards[action], dtype=float)
            self.weights[action] = np.linalg.solve(xtx + reg, features.T @ y)

    def predict_scores(self, feature: np.ndarray) -> dict[str, float]:
        return {action: float(feature @ self.weights[action]) for action in self.actions}

    def predict(self, feature: np.ndarray) -> str:
        scores = self.predict_scores(feature)
        return max(self.actions, key=lambda action: (scores[action], -self.actions.index(action)))


@dataclass
class MarginWeightedDirectMethodBandit(DirectMethodBandit):
    margin_floor: float = 0.25

    def fit(self, features: np.ndarray, rewards: dict[str, list[float]]) -> None:
        weights = _reward_margin_weights(rewards, self.actions, self.margin_floor)
        self.sample_weights_ = weights
        sqrt_weights = np.sqrt(weights)
        weighted_features = features * sqrt_weights[:, None]
        xtx = weighted_features.T @ weighted_features
        reg = self.l2 * np.eye(features.shape[1])
        self.weights = {}
        for action in self.actions:
            y = np.asarray(rewards[action], dtype=float) * sqrt_weights
            self.weights[action] = np.linalg.solve(xtx + reg, weighted_features.T @ y)


def _reward_margin_weights(rewards: dict[str, list[float]], actions: list[str], margin_floor: float) -> np.ndarray:
    reward_matrix = np.vstack([np.asarray(rewards[action], dtype=float) for action in actions])
    if reward_matrix.shape[0] < 2:
        return np.ones(reward_matrix.shape[1], dtype=float)
    sorted_rewards = np.sort(reward_matrix, axis=0)
    margins = sorted_rewards[-1] - sorted_rewards[-2]
    max_margin = float(np.max(margins)) if margins.size else 0.0
    if max_margin <= 1e-12:
        return np.ones(reward_matrix.shape[1], dtype=float)
    floor = float(np.clip(margin_floor, 0.0, 1.0))
    return floor + (1.0 - floor) * (margins / max_margin)


@dataclass
class KnnDirectMethodBandit:
    actions: list[str]
    k: int = 7

    def fit(self, features: np.ndarray, rewards: dict[str, list[float]]) -> None:
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-6
        self.features = (features - self.mean) / self.std
        self.rewards = {action: np.asarray(rewards[action], dtype=float) for action in self.actions}

    def predict_scores(self, feature: np.ndarray) -> dict[str, float]:
        query = (feature - self.mean) / self.std
        distances = np.linalg.norm(self.features - query, axis=1)
        k = min(self.k, len(distances))
        idx = np.argpartition(distances, k - 1)[:k]
        weights = 1.0 / (distances[idx] + 1e-3)
        return {
            action: float(np.sum(weights * self.rewards[action][idx]) / np.sum(weights))
            for action in self.actions
        }

    def predict(self, feature: np.ndarray) -> str:
        scores = self.predict_scores(feature)
        return max(self.actions, key=lambda action: (scores[action], -self.actions.index(action)))


@dataclass
class SklearnRegressorBandit:
    actions: list[str]
    estimator_factory: Callable[[], object]

    def fit(self, features: np.ndarray, rewards: dict[str, list[float]]) -> None:
        self.models = {}
        for action in self.actions:
            model = self.estimator_factory()
            model.fit(features, np.asarray(rewards[action], dtype=float))
            self.models[action] = model

    def predict_scores(self, feature: np.ndarray) -> dict[str, float]:
        x = feature.reshape(1, -1)
        return {action: float(self.models[action].predict(x)[0]) for action in self.actions}

    def predict(self, feature: np.ndarray) -> str:
        scores = self.predict_scores(feature)
        return max(self.actions, key=lambda action: (scores[action], -self.actions.index(action)))


@dataclass(frozen=True)
class TreeRegressorFactory:
    model_type: str
    seed: int
    n_estimators: int = 200
    min_samples_leaf: int = 3
    max_features: str = "sqrt"

    def __call__(self) -> object:
        if self.model_type == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.seed,
                n_jobs=1,
            )
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.seed,
                n_jobs=1,
            )
        raise ValueError(f"Unknown tree model: {self.model_type}")


@dataclass(frozen=True)
class NeuralRegressorFactory:
    seed: int
    hidden_layer_sizes: tuple[int, ...] = (32, 16)
    alpha: float = 1e-3
    max_iter: int = 500

    def __call__(self) -> object:
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="lbfgs",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
