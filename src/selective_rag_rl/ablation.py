from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.bandit import KnnDirectMethodBandit
from selective_rag_rl.data import load_hotpotqa, split_examples
from selective_rag_rl.experiment import _evaluate_all_actions
from selective_rag_rl.rewrites import ACTIONS


@dataclass(frozen=True)
class PolicyVariant:
    name: str
    actions: list[str]
    reward_fn: Callable[[dict[str, object]], float]


def run_hotpot_ablation(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 600,
    seed: int = 42,
    k: int = 5,
) -> dict[str, object]:
    examples = load_hotpotqa(data_path, num_examples=num_examples, seed=seed)
    train, test = split_examples(examples)
    variants = [
        PolicyVariant("Selective bandit", ACTIONS, lambda row: float(row["reward"])),
        PolicyVariant("No keep action", [a for a in ACTIONS if a != "keep"], lambda row: float(row["reward"])),
        PolicyVariant(
            "No cost penalty",
            ACTIONS,
            lambda row: float(row["recall_at_5"]) + 0.5 * float(row["mrr"]),
        ),
        PolicyVariant("Retrieval-only reward", ACTIONS, lambda row: float(row["recall_at_5"])),
    ]

    train_evals = [_evaluate_all_actions(ex, k) for ex in tqdm(train, desc="ablation train actions")]
    test_evals = [(ex, _evaluate_all_actions(ex, k)) for ex in tqdm(test, desc="ablation test actions")]

    rows: list[dict[str, object]] = []
    for variant in variants:
        features = np.vstack([row["features"] for row in train_evals])
        rewards = {
            action: [variant.reward_fn(row["actions"][action]) for row in train_evals]
            for action in variant.actions
        }
        policy = KnnDirectMethodBandit(actions=variant.actions)
        policy.fit(features, rewards)
        for ex, action_eval in test_evals:
            selected = policy.predict(action_eval["features"])
            rows.append(
                {
                    "method": variant.name,
                    "action": selected,
                    "qid": ex.qid,
                    "question": ex.question,
                    **action_eval["actions"][selected],
                }
            )

    df = pd.DataFrame(rows)
    summary = _summarize_ablation(df)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = results_dir / "hotpot_ablation_detailed.csv"
    summary_csv = results_dir / "hotpot_ablation_summary.csv"
    summary_json = results_dir / "hotpot_ablation_summary.json"
    table_tex = results_dir / "hotpot_ablation_table.tex"
    df.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    table_tex.write_text(_latex_table(summary), encoding="utf-8")

    metadata = {
        "dataset": "HotpotQA ablation",
        "num_examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "seed": seed,
        "k": k,
        "outputs": {
            "detailed_csv": str(detailed_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "table_tex": str(table_tex),
        },
    }
    (results_dir / "hotpot_ablation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["Selective bandit", "No keep action", "No cost penalty", "Retrieval-only reward"]:
        part = df[df["method"] == method]
        rows.append(
            {
                "method": method,
                "recall_at_5": float(part["recall_at_5"].mean()),
                "mrr": float(part["mrr"].mean()),
                "ndcg_at_5": float(part["ndcg_at_5"].mean()),
                "reward": float(part["reward"].mean()),
                "rewrite_cost": float(part["rewrite_cost"].mean()),
                "retrieval_calls": float(part["retrieval_calls"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _latex_table(summary: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Recall@5 & MRR & nDCG@5 & Reward & Cost & Calls \\\\",
        "\\midrule",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"{row['method']} & {row['recall_at_5']:.3f} & {row['mrr']:.3f} & "
            f"{row['ndcg_at_5']:.3f} & {row['reward']:.3f} & {row['rewrite_cost']:.3f} & "
            f"{row['retrieval_calls']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)
