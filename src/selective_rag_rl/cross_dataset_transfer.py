from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.bandit import DirectMethodBandit
from selective_rag_rl.data import QAExample, load_beir_dataset
from selective_rag_rl.dense_experiment import FakeDenseEmbedder
from selective_rag_rl.dense_retriever import load_sentence_transformer
from selective_rag_rl.heuristic_policy import heuristic_retrieval_action
from selective_rag_rl.retrieval_policy_experiment import (
    ACTION_TO_METHOD,
    RETRIEVAL_ACTIONS,
    evaluate_retrieval_actions,
    fit_policy_feature_transform,
    summarize_retrieval_policy,
)


TRANSFER_METHOD_ORDER = [
    "Target train-best action",
    "Heuristic retrieval router",
    "Transferred source policy",
    "Oracle retrieval action",
]


@dataclass(frozen=True)
class TransferInputs:
    source_dataset: str
    target_dataset: str
    source_train: list[dict[str, Any]]
    target_test: list[dict[str, Any]]
    actions: list[str]
    method_names: dict[str, str]
    output_dir: Path
    output_prefix: str = "cross_dataset_transfer"
    target_train: list[dict[str, Any]] | None = None


def run_transfer_from_evals(inputs: TransferInputs) -> dict[str, str]:
    actions = list(inputs.actions)
    if not actions:
        raise ValueError("actions must not be empty")
    if not inputs.source_train:
        raise ValueError("source_train must not be empty")

    raw_source_features = np.vstack([np.asarray(row["features"], dtype=float) for row in inputs.source_train])
    feature_transform = fit_policy_feature_transform(raw_source_features, "full")
    source_features = feature_transform.transform(raw_source_features)
    source_rewards = {action: [float(row["actions"][action]["reward"]) for row in inputs.source_train] for action in actions}

    policy = DirectMethodBandit(actions=actions, l2=1.0)
    policy.fit(source_features, source_rewards)

    target_train = inputs.target_train if inputs.target_train is not None else inputs.target_test
    best_fixed_action = _best_fixed_action(target_train, actions)
    method_order = [inputs.method_names.get(action, action) for action in actions]
    method_order.extend(TRANSFER_METHOD_ORDER)

    rows: list[dict[str, object]] = []
    for action_eval in inputs.target_test:
        for action in actions:
            rows.append(_transfer_row(inputs, action_eval, inputs.method_names.get(action, action), action))

        rows.append(_transfer_row(inputs, action_eval, "Target train-best action", best_fixed_action))

        heuristic_action = heuristic_retrieval_action(np.asarray(action_eval["features"], dtype=float), actions)
        rows.append(_transfer_row(inputs, action_eval, "Heuristic retrieval router", heuristic_action))

        policy_scores = policy.predict_scores(feature_transform.transform(np.asarray(action_eval["features"], dtype=float)))
        transferred_action = max(actions, key=lambda action: (policy_scores[action], -actions.index(action)))
        rows.append(
            _transfer_row(
                inputs,
                action_eval,
                "Transferred source policy",
                transferred_action,
                extra={"policy_action_score": policy_scores[transferred_action]},
            )
        )

        oracle_action = max(actions, key=lambda action: float(action_eval["actions"][action]["reward"]))
        rows.append(_transfer_row(inputs, action_eval, "Oracle retrieval action", oracle_action))

    results_dir = inputs.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = results_dir / f"{inputs.output_prefix}_detailed.csv"
    summary_csv = results_dir / f"{inputs.output_prefix}_summary.csv"

    detailed = pd.DataFrame(rows)
    detailed.to_csv(detailed_csv, index=False)
    summary = summarize_retrieval_policy(detailed, method_order=method_order)
    summary.insert(0, "source_dataset", inputs.source_dataset)
    summary.insert(1, "target_dataset", inputs.target_dataset)
    summary.to_csv(summary_csv, index=False)

    return {"detailed_csv": str(detailed_csv), "summary_csv": str(summary_csv)}


def run_beir_transfer_matrix(
    scifact_path: Path,
    nfcorpus_path: Path,
    output_dir: Path,
    num_train_examples: int = 600,
    num_test_examples: int = 300,
    seed: int = 42,
    full_corpus: bool = True,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, object]:
    dataset_paths = {"scifact": scifact_path, "nfcorpus": nfcorpus_path}
    embedder = FakeDenseEmbedder() if embedder_name == "fake" else load_sentence_transformer(embedder_name)
    retriever_cache: dict[tuple[str, ...], object] = {}

    train_evals: dict[str, list[dict[str, Any]]] = {}
    test_evals: dict[str, list[dict[str, Any]]] = {}
    for dataset, data_path in dataset_paths.items():
        train_examples = load_beir_dataset(
            data_path,
            num_examples=num_train_examples,
            seed=seed,
            split="train",
            full_corpus=full_corpus,
            qtype=f"beir-{dataset}",
        )
        test_examples = load_beir_dataset(
            data_path,
            num_examples=num_test_examples,
            seed=seed,
            split="test",
            full_corpus=full_corpus,
            qtype=f"beir-{dataset}",
        )
        train_evals[dataset] = _evaluate_examples(
            train_examples,
            embedder,
            retriever_cache,
            desc=f"{dataset} train transfer actions",
        )
        test_evals[dataset] = _evaluate_examples(
            test_examples,
            embedder,
            retriever_cache,
            desc=f"{dataset} test transfer actions",
        )

    pair_outputs = []
    for source_dataset, target_dataset in [
        ("scifact", "scifact"),
        ("scifact", "nfcorpus"),
        ("nfcorpus", "nfcorpus"),
        ("nfcorpus", "scifact"),
    ]:
        pair_outputs.append(
            run_transfer_from_evals(
                TransferInputs(
                    source_dataset=source_dataset,
                    target_dataset=target_dataset,
                    source_train=train_evals[source_dataset],
                    target_train=train_evals[target_dataset],
                    target_test=test_evals[target_dataset],
                    actions=list(RETRIEVAL_ACTIONS),
                    method_names={action: ACTION_TO_METHOD[action] for action in RETRIEVAL_ACTIONS},
                    output_dir=output_dir,
                    output_prefix=f"cross_dataset_transfer_{source_dataset}_to_{target_dataset}",
                )
            )
        )

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = results_dir / "cross_dataset_transfer_detailed.csv"
    summary_csv = results_dir / "cross_dataset_transfer_summary.csv"

    detailed = pd.concat([pd.read_csv(paths["detailed_csv"]) for paths in pair_outputs], ignore_index=True)
    summary = pd.concat([pd.read_csv(paths["summary_csv"]) for paths in pair_outputs], ignore_index=True)
    detailed.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    return {
        "detailed_csv": str(detailed_csv),
        "summary_csv": str(summary_csv),
        "pair_outputs": pair_outputs,
        "num_train_examples": num_train_examples,
        "num_test_examples": num_test_examples,
        "seed": seed,
        "full_corpus": full_corpus,
        "embedder": embedder_name,
    }


def _evaluate_examples(
    examples: list[QAExample],
    embedder: object,
    retriever_cache: dict[tuple[str, ...], object],
    desc: str,
) -> list[dict[str, Any]]:
    rows = []
    for ex in tqdm(examples, desc=desc):
        row = evaluate_retrieval_actions(
            ex,
            embedder,
            k=5,
            dense_weight=0.5,
            retrieval_call_cost=0.03,
            actions=list(RETRIEVAL_ACTIONS),
            retriever_cache=retriever_cache,
        )
        row["qid"] = ex.qid
        row["question"] = ex.question
        rows.append(row)
    return rows


def _best_fixed_action(evals: list[dict[str, Any]], actions: list[str]) -> str:
    if not evals:
        return actions[0]
    return max(
        actions,
        key=lambda action: (
            float(np.mean([float(row["actions"][action]["reward"]) for row in evals])),
            -actions.index(action),
        ),
    )


def _transfer_row(
    inputs: TransferInputs,
    action_eval: dict[str, Any],
    method: str,
    action: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "source_dataset": inputs.source_dataset,
        "target_dataset": inputs.target_dataset,
        "split": "test",
        "method": method,
        "action": action,
        "qid": str(action_eval.get("qid", "")),
        "question": str(action_eval.get("question", "")),
        **action_eval["actions"][action],
    }
    if extra:
        row.update(extra)
    return row
