from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.data import load_beir_dataset
from selective_rag_rl.dense_experiment import FakeDenseEmbedder
from selective_rag_rl.dense_retriever import load_sentence_transformer
from selective_rag_rl.retrieval_policy_experiment import (
    BASE_RETRIEVAL_ACTIONS,
    SEMANTIC_DEPTH_DEFAULT,
    evaluate_retrieval_actions,
    fit_policy_feature_transform,
    _load_semantic_embedder,
    _prewarm_semantic_embeddings,
    _validate_semantic_depth,
)


@dataclass
class LinUCBPolicy:
    actions: list[str]
    alpha: float = 1.0
    l2: float = 1.0

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must not be empty")

    def fit(self, features: np.ndarray, rewards: dict[str, list[float]]) -> pd.DataFrame:
        self._init_params(features.shape[1])
        rows = []
        for step, feature in enumerate(features):
            scores = self.predict_scores(feature)
            action = max(self.actions, key=lambda item: (scores[item], -self.actions.index(item)))
            reward = float(rewards[action][step])
            oracle_reward = max(float(rewards[item][step]) for item in self.actions)
            self._update(action, feature, reward)
            rows.append(
                {
                    "step": step,
                    "chosen_action": action,
                    "reward": reward,
                    "oracle_reward": oracle_reward,
                    "regret": oracle_reward - reward,
                }
            )
        return pd.DataFrame(rows)

    def predict_scores(self, feature: np.ndarray) -> dict[str, float]:
        x = np.asarray(feature, dtype=float)
        scores = {}
        for action in self.actions:
            inv_a = np.linalg.inv(self.a[action])
            theta = inv_a @ self.b[action]
            mean = float(x @ theta)
            uncertainty = float(np.sqrt(max(x @ inv_a @ x, 0.0)))
            scores[action] = mean + self.alpha * uncertainty
        return scores

    def predict(self, feature: np.ndarray) -> str:
        scores = self.predict_scores(feature)
        return max(self.actions, key=lambda item: (scores[item], -self.actions.index(item)))

    def _init_params(self, feature_width: int) -> None:
        self.a = {action: self.l2 * np.eye(feature_width) for action in self.actions}
        self.b = {action: np.zeros(feature_width, dtype=float) for action in self.actions}

    def _update(self, action: str, feature: np.ndarray, reward: float) -> None:
        x = np.asarray(feature, dtype=float)
        self.a[action] += np.outer(x, x)
        self.b[action] += reward * x


@dataclass
class EpsilonGreedyPolicy:
    actions: list[str]
    epsilon: float = 0.1
    l2: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must not be empty")
        if self.epsilon < 0.0 or self.epsilon > 1.0:
            raise ValueError("epsilon must be in [0, 1]")

    def fit(self, features: np.ndarray, rewards: dict[str, list[float]]) -> pd.DataFrame:
        self._init_params(features.shape[1])
        rng = np.random.default_rng(self.seed)
        rows = []
        for step, feature in enumerate(features):
            if step < len(self.actions):
                action = self.actions[step]
            elif rng.random() < self.epsilon:
                action = str(rng.choice(self.actions))
            else:
                action = self.predict(feature)
            reward = float(rewards[action][step])
            oracle_reward = max(float(rewards[item][step]) for item in self.actions)
            self._update(action, feature, reward)
            rows.append(
                {
                    "step": step,
                    "chosen_action": action,
                    "reward": reward,
                    "oracle_reward": oracle_reward,
                    "regret": oracle_reward - reward,
                }
            )
        return pd.DataFrame(rows)

    def predict_scores(self, feature: np.ndarray) -> dict[str, float]:
        x = np.asarray(feature, dtype=float)
        scores = {}
        for action in self.actions:
            theta = np.linalg.solve(self.a[action], self.b[action])
            scores[action] = float(x @ theta)
        return scores

    def predict(self, feature: np.ndarray) -> str:
        scores = self.predict_scores(feature)
        return max(self.actions, key=lambda item: (scores[item], -self.actions.index(item)))

    def _init_params(self, feature_width: int) -> None:
        self.a = {action: self.l2 * np.eye(feature_width) for action in self.actions}
        self.b = {action: np.zeros(feature_width, dtype=float) for action in self.actions}

    def _update(self, action: str, feature: np.ndarray, reward: float) -> None:
        x = np.asarray(feature, dtype=float)
        self.a[action] += np.outer(x, x)
        self.b[action] += reward * x


@dataclass
class LinearThompsonSamplingPolicy:
    actions: list[str]
    posterior_scale: float = 1.0
    l2: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must not be empty")
        if self.posterior_scale < 0.0:
            raise ValueError("posterior_scale must be non-negative")

    def fit(self, features: np.ndarray, rewards: dict[str, list[float]]) -> pd.DataFrame:
        self._init_params(features.shape[1])
        rng = np.random.default_rng(self.seed)
        rows = []
        for step, feature in enumerate(features):
            if step < len(self.actions):
                action = self.actions[step]
            else:
                action = self._sample_action(feature, rng)
            reward = float(rewards[action][step])
            oracle_reward = max(float(rewards[item][step]) for item in self.actions)
            self._update(action, feature, reward)
            rows.append(
                {
                    "step": step,
                    "chosen_action": action,
                    "reward": reward,
                    "oracle_reward": oracle_reward,
                    "regret": oracle_reward - reward,
                }
            )
        return pd.DataFrame(rows)

    def predict_scores(self, feature: np.ndarray) -> dict[str, float]:
        x = np.asarray(feature, dtype=float)
        scores = {}
        for action in self.actions:
            theta = np.linalg.solve(self.a[action], self.b[action])
            scores[action] = float(x @ theta)
        return scores

    def predict(self, feature: np.ndarray) -> str:
        scores = self.predict_scores(feature)
        return max(self.actions, key=lambda item: (scores[item], -self.actions.index(item)))

    def _sample_action(self, feature: np.ndarray, rng: np.random.Generator) -> str:
        x = np.asarray(feature, dtype=float)
        sampled_scores = {}
        for action in self.actions:
            inv_a = np.linalg.inv(self.a[action])
            mean = inv_a @ self.b[action]
            covariance = (self.posterior_scale**2) * inv_a
            theta = rng.multivariate_normal(mean, covariance)
            sampled_scores[action] = float(x @ theta)
        return max(self.actions, key=lambda item: (sampled_scores[item], -self.actions.index(item)))

    def _init_params(self, feature_width: int) -> None:
        self.a = {action: self.l2 * np.eye(feature_width) for action in self.actions}
        self.b = {action: np.zeros(feature_width, dtype=float) for action in self.actions}

    def _update(self, action: str, feature: np.ndarray, reward: float) -> None:
        x = np.asarray(feature, dtype=float)
        self.a[action] += np.outer(x, x)
        self.b[action] += reward * x


def export_linucb_baseline_from_evals(
    train_evals: list[dict[str, object]],
    test_evals: list[dict[str, object]],
    actions: list[str],
    output_csv: Path,
    history_csv: Path,
    alpha: float = 1.0,
    l2: float = 1.0,
    epsilon: float = 0.1,
    posterior_scale: float = 1.0,
    seed: int = 42,
) -> Path:
    features = np.vstack([row["features"] for row in train_evals])
    rewards = {action: [float(row["actions"][action]["reward"]) for row in train_evals] for action in actions}
    policy = LinUCBPolicy(actions=actions, alpha=alpha, l2=l2)
    history = policy.fit(features, rewards)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_csv, index=False)

    best_fixed_action = max(actions, key=lambda action: (float(np.mean(rewards[action])), -actions.index(action)))
    linucb_actions = [policy.predict(np.asarray(row["features"], dtype=float)) for row in test_evals]
    epsilon_policy = EpsilonGreedyPolicy(actions=actions, epsilon=epsilon, l2=l2, seed=seed)
    epsilon_policy.fit(features, rewards)
    epsilon_actions = [epsilon_policy.predict(np.asarray(row["features"], dtype=float)) for row in test_evals]
    thompson_policy = LinearThompsonSamplingPolicy(
        actions=actions,
        posterior_scale=posterior_scale,
        l2=l2,
        seed=seed,
    )
    thompson_policy.fit(features, rewards)
    thompson_actions = [thompson_policy.predict(np.asarray(row["features"], dtype=float)) for row in test_evals]
    oracle_actions = [
        max(actions, key=lambda action: float(row["actions"][action]["reward"]))
        for row in test_evals
    ]
    summary = pd.DataFrame(
        [
            _summary_row("LinUCB retrieval policy", test_evals, linucb_actions),
            _summary_row("Epsilon-greedy retrieval policy", test_evals, epsilon_actions),
            _summary_row("Linear Thompson retrieval policy", test_evals, thompson_actions),
            _summary_row("Train-best retrieval action", test_evals, [best_fixed_action] * len(test_evals)),
            _summary_row("Oracle retrieval action", test_evals, oracle_actions),
        ]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return output_csv


def run_beir_linucb_baseline(
    dataset: str,
    data_path: Path,
    output_dir: Path,
    num_train_examples: int = 300,
    num_test_examples: int = 300,
    seed: int = 42,
    full_corpus: bool = False,
    k: int = 5,
    pool_size: int = 100,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_weight: float = 0.5,
    retrieval_call_cost: float = 0.03,
    alpha: float = 1.0,
    l2: float = 1.0,
    epsilon: float = 0.1,
    posterior_scale: float = 1.0,
    feature_set: str = "full",
    semantic_features: str = "none",
    semantic_cache_path: Path | None = None,
    semantic_allow_api: bool = False,
    semantic_max_new_texts: int = 0,
    semantic_dry_run: bool = False,
    semantic_depth: int = SEMANTIC_DEPTH_DEFAULT,
) -> Path:
    if dataset not in {"scifact", "nfcorpus"}:
        raise ValueError(f"Unsupported LinUCB dataset: {dataset}")
    semantic_depth = _validate_semantic_depth(semantic_depth)
    train_examples = load_beir_dataset(
        data_path,
        num_examples=num_train_examples,
        seed=seed,
        split="train",
        pool_size=pool_size,
        full_corpus=full_corpus,
        qtype=f"beir-{dataset}",
    )
    test_examples = load_beir_dataset(
        data_path,
        num_examples=num_test_examples,
        seed=seed,
        split="test",
        pool_size=pool_size,
        full_corpus=full_corpus,
        qtype=f"beir-{dataset}",
    )
    embedder = FakeDenseEmbedder() if embedder_name == "fake" else load_sentence_transformer(embedder_name)
    semantic_embedder = _load_semantic_embedder(
        semantic_features,
        output_dir,
        semantic_cache_path,
        semantic_allow_api=semantic_allow_api,
        semantic_max_new_texts=semantic_max_new_texts,
        semantic_dry_run=semantic_dry_run,
    )
    if semantic_embedder is not None:
        _prewarm_semantic_embeddings([*train_examples, *test_examples], semantic_embedder, semantic_depth)
    retriever_cache = {}
    train_evals = [
        evaluate_retrieval_actions(
            ex,
            embedder,
            k,
            dense_weight,
            retrieval_call_cost,
            semantic_embedder,
            actions=BASE_RETRIEVAL_ACTIONS,
            retriever_cache=retriever_cache,
            semantic_depth=semantic_depth,
        )
        for ex in tqdm(train_examples, desc=f"{dataset} LinUCB train actions")
    ]
    test_evals = [
        evaluate_retrieval_actions(
            ex,
            embedder,
            k,
            dense_weight,
            retrieval_call_cost,
            semantic_embedder,
            actions=BASE_RETRIEVAL_ACTIONS,
            retriever_cache=retriever_cache,
            semantic_depth=semantic_depth,
        )
        for ex in tqdm(test_examples, desc=f"{dataset} LinUCB test actions")
    ]
    feature_transform = fit_policy_feature_transform(np.vstack([row["features"] for row in train_evals]), feature_set)
    transformed_train = [_with_features(row, feature_transform.transform(row["features"])) for row in train_evals]
    transformed_test = [_with_features(row, feature_transform.transform(row["features"])) for row in test_evals]
    results_dir = output_dir / "results"
    return export_linucb_baseline_from_evals(
        train_evals=transformed_train,
        test_evals=transformed_test,
        actions=BASE_RETRIEVAL_ACTIONS,
        output_csv=results_dir / f"{dataset}_linucb_baseline_summary.csv",
        history_csv=results_dir / f"{dataset}_linucb_baseline_history.csv",
        alpha=alpha,
        l2=l2,
        epsilon=epsilon,
        posterior_scale=posterior_scale,
        seed=seed,
    )


def _summary_row(method: str, evals: list[dict[str, object]], actions: list[str]) -> dict[str, object]:
    return {
        "method": method,
        "recall_at_5": _metric(evals, actions, "recall_at_5"),
        "mrr": _metric(evals, actions, "mrr"),
        "ndcg_at_5": _metric(evals, actions, "ndcg_at_5"),
        "reward": _metric(evals, actions, "reward"),
        "rewrite_cost": _metric(evals, actions, "rewrite_cost"),
        "retrieval_calls": _metric(evals, actions, "retrieval_calls"),
    }


def _metric(evals: list[dict[str, object]], actions: list[str], metric: str) -> float:
    values = [float(row["actions"][action][metric]) for row, action in zip(evals, actions)]
    return float(np.mean(values)) if values else 0.0


def _with_features(row: dict[str, object], features: np.ndarray) -> dict[str, object]:
    updated = dict(row)
    updated["features"] = np.asarray(features, dtype=float)
    return updated
