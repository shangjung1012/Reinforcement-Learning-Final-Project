from __future__ import annotations

import numpy as np


class RidgeQ:
    def __init__(self, actions: list[str], l2: float = 1.0) -> None:
        self.actions = actions
        self.l2 = l2
        self.weights: dict[str, np.ndarray] = {}

    def fit(self, rows: list[tuple[np.ndarray, str, float]]) -> None:
        dim = rows[0][0].shape[0]
        for action in self.actions:
            action_rows = [(x, y) for x, a, y in rows if a == action]
            if not action_rows:
                self.weights[action] = np.zeros(dim)
                continue
            xmat = np.vstack([x for x, _y in action_rows])
            yvec = np.asarray([y for _x, y in action_rows], dtype=float)
            reg = self.l2 * np.eye(dim)
            self.weights[action] = np.linalg.solve(xmat.T @ xmat + reg, xmat.T @ yvec)

    def value(self, feature: np.ndarray, action: str) -> float:
        return float(feature @ self.weights[action])

    def choose(self, feature: np.ndarray) -> str:
        return max(self.actions, key=lambda a: (self.value(feature, a), -self.actions.index(a)))


class KnnQ:
    def __init__(self, actions: list[str], k: int = 5) -> None:
        self.actions = actions
        self.k = k

    def fit(self, rows: list[tuple[np.ndarray, str, float]]) -> None:
        xmat = np.vstack([x for x, _a, _y in rows])
        self.mean = xmat.mean(axis=0)
        self.std = xmat.std(axis=0) + 1e-6
        self.by_action: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for action in self.actions:
            action_rows = [(x, y) for x, a, y in rows if a == action]
            if not action_rows:
                self.by_action[action] = (np.zeros((0, xmat.shape[1])), np.zeros(0))
                continue
            feats = np.vstack([(x - self.mean) / self.std for x, _y in action_rows])
            vals = np.asarray([y for _x, y in action_rows], dtype=float)
            self.by_action[action] = (feats, vals)

    def value(self, feature: np.ndarray, action: str) -> float:
        feats, vals = self.by_action[action]
        if vals.size == 0:
            return float("-inf")
        query = (feature - self.mean) / self.std
        dists = np.linalg.norm(feats - query, axis=1)
        k = min(self.k, vals.size)
        idx = np.argpartition(dists, k - 1)[:k]
        weights = 1.0 / (dists[idx] + 1e-3)
        return float(np.sum(weights * vals[idx]) / np.sum(weights))

    def choose(self, feature: np.ndarray) -> str:
        return max(self.actions, key=lambda a: (self.value(feature, a), -self.actions.index(a)))
