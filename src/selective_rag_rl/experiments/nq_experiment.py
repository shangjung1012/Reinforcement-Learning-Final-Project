from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.policies.bandit import KnnDirectMethodBandit
from selective_rag_rl.core.data import load_natural_questions, split_examples
from selective_rag_rl.experiments.experiment import _evaluate_all_actions, _latex_table, _plot_actions, _plot_metrics, summarize
from selective_rag_rl.core.policy_io import save_checkpoint
from selective_rag_rl.core.rewrites import ACTIONS


def run_nq_experiment(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 500,
    seed: int = 42,
    pool_size: int = 50,
    k: int = 5,
) -> dict[str, object]:
    examples = load_natural_questions(data_path, num_examples=num_examples, seed=seed, pool_size=pool_size)
    train, test = split_examples(examples)

    train_evals = [_evaluate_all_actions(ex, k) for ex in tqdm(train, desc="nq train actions")]
    features = np.vstack([row["features"] for row in train_evals])
    rewards = {action: [row["actions"][action]["reward"] for row in train_evals] for action in ACTIONS}
    best_fixed_action = max(
        ACTIONS,
        key=lambda action: (float(np.mean(rewards[action])), -ACTIONS.index(action)),
    )

    policy = KnnDirectMethodBandit(actions=ACTIONS)
    policy.fit(features, rewards)

    rows: list[dict[str, object]] = []
    for split_name, split in [("train", train), ("test", test)]:
        for ex in tqdm(split, desc=f"nq {split_name} evaluation"):
            action_eval = _evaluate_all_actions(ex, k)
            selected = policy.predict(action_eval["features"])
            for method_name, action in [
                ("Vanilla BM25", "keep"),
                ("Rewrite-all keyword", "keyword_compress"),
                ("Rewrite-all entity expansion", "entity_expand"),
                ("Train-best fixed rewrite", best_fixed_action),
                ("Selective bandit", selected),
            ]:
                rows.append(
                    {
                        "split": split_name,
                        "method": method_name,
                        "action": action,
                        "qid": ex.qid,
                        "question": ex.question,
                        **action_eval["actions"][action],
                    }
                )
            oracle_action = max(ACTIONS, key=lambda a: action_eval["actions"][a]["reward"])
            rows.append(
                {
                    "split": split_name,
                    "method": "Oracle best action",
                    "action": oracle_action,
                    "qid": ex.qid,
                    "question": ex.question,
                    **action_eval["actions"][oracle_action],
                }
            )

    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    detailed_csv = results_dir / "nq_toy_detailed.csv"
    summary_csv = results_dir / "nq_toy_summary.csv"
    summary_json = results_dir / "nq_toy_summary.json"
    table_tex = results_dir / "nq_toy_table.tex"
    df.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    table_tex.write_text(_latex_table(summary), encoding="utf-8")
    _plot_metrics(summary, figures_dir)
    _plot_actions(df, figures_dir)
    checkpoint_path = checkpoints_dir / "nq_bandit_policy.pkl"
    save_checkpoint(
        checkpoint_path,
        policy,
        {
            "dataset": "Natural Questions",
            "num_examples": len(examples),
            "train_examples": len(train),
            "seed": seed,
            "pool_size": pool_size,
            "k": k,
            "actions": ACTIONS,
            "best_fixed_action": best_fixed_action,
        },
    )

    metadata = {
        "dataset": "Natural Questions single-hop title retrieval",
        "num_examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "seed": seed,
        "pool_size": pool_size,
        "k": k,
        "best_fixed_action": best_fixed_action,
        "outputs": {
            "detailed_csv": str(detailed_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "table_tex": str(table_tex),
            "checkpoint": str(checkpoint_path),
        },
    }
    (results_dir / "nq_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
