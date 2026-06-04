from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.policies.bandit import KnnDirectMethodBandit, state_features
from selective_rag_rl.core.data import QAExample, load_hotpotqa, split_examples
from selective_rag_rl.core.metrics import aggregate, mrr, ndcg_at_k, recall_at_k
from selective_rag_rl.core.policy_io import save_checkpoint
from selective_rag_rl.core.retriever import BM25Retriever, RetrievalResult
from selective_rag_rl.core.rewrites import ACTIONS, rewrite, rewrite_cost


def run_experiment(
    data_path: Path,
    output_dir: Path,
    num_examples: int = 300,
    seed: int = 42,
    k: int = 5,
) -> dict[str, object]:
    examples = load_hotpotqa(data_path, num_examples=num_examples, seed=seed)
    train, test = split_examples(examples)

    train_evals = [_evaluate_all_actions(ex, k) for ex in tqdm(train, desc="train actions")]
    features = np.vstack([row["features"] for row in train_evals])
    rewards = {action: [row["actions"][action]["reward"] for row in train_evals] for action in ACTIONS}
    best_fixed_action = max(
        ACTIONS,
        key=lambda action: (float(np.mean(rewards[action])), -ACTIONS.index(action)),
    )

    policy = KnnDirectMethodBandit(actions=ACTIONS)
    policy.fit(features, rewards)

    rows: list[dict[str, object]] = []
    for split_name, split_examples_ in [("train", train), ("test", test)]:
        for ex in tqdm(split_examples_, desc=f"{split_name} evaluation"):
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

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    detailed_csv = results_dir / "hotpot_toy_detailed.csv"
    summary_csv = results_dir / "hotpot_toy_summary.csv"
    summary_json = results_dir / "hotpot_toy_summary.json"
    table_tex = results_dir / "hotpot_toy_table.tex"
    df.to_csv(detailed_csv, index=False)

    summary = summarize(df)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    table_tex.write_text(_latex_table(summary), encoding="utf-8")
    _plot_metrics(summary, figures_dir)
    _plot_actions(df, figures_dir)
    _write_examples(df, results_dir)
    checkpoint_path = checkpoints_dir / "hotpot_bandit_policy.pkl"
    save_checkpoint(
        checkpoint_path,
        policy,
        {
            "dataset": "HotpotQA",
            "num_examples": len(examples),
            "train_examples": len(train),
            "seed": seed,
            "k": k,
            "actions": ACTIONS,
            "best_fixed_action": best_fixed_action,
        },
    )

    metadata = {
        "num_examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "seed": seed,
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
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"] == "test"]
    metrics = ["recall_at_5", "mrr", "ndcg_at_5", "reward", "rewrite_cost", "retrieval_calls"]
    rows = []
    order = [
        "Vanilla BM25",
        "Rewrite-all keyword",
        "Rewrite-all entity expansion",
        "Train-best fixed rewrite",
        "Selective bandit",
        "Oracle best action",
    ]
    for method in order:
        part = test[test["method"] == method]
        row = {"method": method}
        for metric in metrics:
            row[metric] = float(part[metric].mean()) if len(part) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_all_actions(ex: QAExample, k: int) -> dict[str, object]:
    retriever = BM25Retriever(ex.passages)
    initial = retriever.search(ex.question, k=k)
    output: dict[str, dict[str, object]] = {}
    for action in ACTIONS:
        rewritten = rewrite(ex.question, action)
        merged = _merge_results([retriever.search(q, k=k) for q in rewritten.queries], k=k)
        rec = recall_at_k(merged, ex.gold_doc_ids, k)
        rr = mrr(merged, ex.gold_doc_ids)
        ndcg = ndcg_at_k(merged, ex.gold_doc_ids, k)
        cost = rewrite_cost(ex.question, rewritten)
        output[action] = {
            "recall_at_5": rec,
            "mrr": rr,
            "ndcg_at_5": ndcg,
            "rewrite_cost": cost,
            "retrieval_calls": rewritten.retrieval_calls,
            "rewrite_tokens": sum(len(q.split()) for q in rewritten.queries),
            "reward": rec + 0.5 * rr - cost,
            "queries": " || ".join(rewritten.queries),
            "top_docs": " | ".join(r.doc_id for r in merged),
            "gold_docs": " | ".join(sorted(ex.gold_doc_ids)),
        }
    return {"features": state_features(ex.question, initial), "actions": output}


def _merge_results(result_lists: list[list[RetrievalResult]], k: int) -> list[RetrievalResult]:
    best: dict[str, float] = {}
    for results in result_lists:
        for result in results:
            best[result.doc_id] = max(best.get(result.doc_id, float("-inf")), result.score)
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [RetrievalResult(doc_id=doc_id, score=score, rank=i + 1) for i, (doc_id, score) in enumerate(ranked)]


def _latex_table(summary: pd.DataFrame) -> str:
    cols = ["method", "recall_at_5", "mrr", "ndcg_at_5", "reward", "rewrite_cost", "retrieval_calls"]
    display = summary[cols].copy()
    for col in cols[1:]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Recall@5 & MRR & nDCG@5 & Reward & Cost & Calls \\\\",
        "\\midrule",
    ]
    for row in display.to_dict(orient="records"):
        lines.append(
            f"{row['method']} & {row['recall_at_5']} & {row['mrr']} & {row['ndcg_at_5']} & "
            f"{row['reward']} & {row['rewrite_cost']} & {row['retrieval_calls']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _plot_metrics(summary: pd.DataFrame, figures_dir: Path) -> None:
    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, metric, label in [
        (-width, "recall_at_5", "Recall@5"),
        (0.0, "mrr", "MRR"),
        (width, "ndcg_at_5", "nDCG@5"),
    ]:
        ax.bar(x + offset, summary[metric], width, label=label)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "retrieval_metrics.png", dpi=180)
    plt.close(fig)


def _plot_actions(df: pd.DataFrame, figures_dir: Path) -> None:
    selected = df[(df["split"] == "test") & (df["method"] == "Selective bandit")]
    counts = selected["action"].value_counts().reindex(ACTIONS, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#4C78A8")
    ax.set_ylabel("Number of test questions")
    ax.set_xlabel("Selected action")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "bandit_actions.png", dpi=180)
    plt.close(fig)


def _write_examples(df: pd.DataFrame, results_dir: Path) -> None:
    test = df[df["split"] == "test"]
    pivot = test.pivot_table(index="qid", columns="method", values="recall_at_5", aggfunc="first")
    improved = pivot[
        (pivot.get("Selective bandit", 0) > pivot.get("Vanilla BM25", 0))
        & (pivot.get("Selective bandit", 0) >= pivot.get("Rewrite-all keyword", 0))
    ]
    rows = []
    for qid in improved.index[:5]:
        selected = test[(test["qid"] == qid) & (test["method"] == "Selective bandit")].iloc[0]
        vanilla = test[(test["qid"] == qid) & (test["method"] == "Vanilla BM25")].iloc[0]
        rows.append(
            {
                "qid": qid,
                "question": selected["question"],
                "selected_action": selected["action"],
                "vanilla_top_docs": vanilla["top_docs"],
                "selected_top_docs": selected["top_docs"],
                "gold_docs": selected["gold_docs"],
            }
        )
    (results_dir / "qualitative_examples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
