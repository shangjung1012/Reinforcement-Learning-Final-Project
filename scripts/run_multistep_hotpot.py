from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from selective_rag_rl.bandit import state_features
from selective_rag_rl.data import QAExample, load_hotpotqa, split_examples
from selective_rag_rl.fqi import KnnQ, RidgeQ
from selective_rag_rl.metrics import mrr, ndcg_at_k, recall_at_k
from selective_rag_rl.policy_io import save_checkpoint
from selective_rag_rl.retriever import BM25Retriever, RetrievalResult
from selective_rag_rl.rewrites import RewriteOutput, rewrite
from selective_rag_rl.text import content_tokens, unique_preserve_order

REWRITE_ACTIONS = ["keyword_compress", "entity_expand", "decompose", "feedback_expand", "title_bridge"]
ACTIONS = ["stop", *REWRITE_ACTIONS]


@dataclass(frozen=True)
class State:
    query: str
    results: list[RetrievalResult]
    step: int
    cost: float
    calls: int
    previous_action: str


@dataclass(frozen=True)
class CostConfig:
    retrieval: float
    token: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/HotpotQA/hotpot_dev_distractor_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--target-mode", choices=["bootstrapped", "exact"], default="bootstrapped")
    parser.add_argument("--model", choices=["ridge", "knn"], default="knn")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--retrieval-cost", type=float, default=0.02)
    parser.add_argument("--token-cost", type=float, default=0.005)
    args = parser.parse_args()

    metadata = run_multistep_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        seed=args.seed,
        k=args.k,
        target_mode=args.target_mode,
        model=args.model,
        knn_k=args.knn_k,
        cost_config=CostConfig(retrieval=args.retrieval_cost, token=args.token_cost),
    )
    print(pd.read_csv(metadata["outputs"]["summary_csv"]).to_string(index=False))


def run_multistep_experiment(
    data_path: Path,
    output_dir: Path,
    num_examples: int,
    seed: int,
    k: int,
    target_mode: str,
    model: str,
    knn_k: int,
    cost_config: CostConfig,
) -> dict[str, object]:
    examples = load_hotpotqa(data_path, num_examples=num_examples, seed=seed)
    train, test = split_examples(examples)
    best_fixed_trace = select_best_fixed_trace(train, k, cost_config)

    q1_rows: list[tuple[np.ndarray, str, float]] = []
    q0_rows: list[tuple[np.ndarray, str, float]] = []
    for ex in tqdm(train, desc="building FQI data"):
        retriever = BM25Retriever(ex.passages)
        s0 = initial_state(ex, retriever, k)
        for action in ACTIONS:
            q0_rows.append((features(s0), action, exact_two_step_value(ex, retriever, s0, action, k, cost_config)))
        for first_action in REWRITE_ACTIONS:
            s1 = transition(ex, retriever, s0, first_action, k, cost_config)
            for action in ACTIONS:
                q1_rows.append((features(s1), action, exact_one_step_value(ex, retriever, s1, action, k, cost_config)))

    q1 = make_q_model(model, knn_k)
    q1.fit(q1_rows)
    q0_training_rows = make_q0_rows(train, q1, k, target_mode, cost_config)
    q0 = make_q_model(model, knn_k)
    q0.fit(q0_training_rows)

    rows = []
    for split_name, split in [("train", train), ("test", test)]:
        for ex in tqdm(split, desc=f"evaluating {split_name}"):
            rows.extend(evaluate_example(ex, split_name, q0, q1, best_fixed_trace, k, cost_config))

    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    detailed_csv = results_dir / "multistep_detailed.csv"
    summary_csv = results_dir / "multistep_summary.csv"
    summary_json = results_dir / "multistep_summary.json"
    table_tex = results_dir / "multistep_table.tex"
    metadata_json = results_dir / "multistep_metadata.json"
    df.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    table_tex.write_text(latex_table(summary), encoding="utf-8")
    plot_summary(summary, figures_dir)
    plot_policy_actions(df, figures_dir)
    q0_checkpoint = checkpoints_dir / "hotpot_fqi_q0.pkl"
    q1_checkpoint = checkpoints_dir / "hotpot_fqi_q1.pkl"
    checkpoint_metadata = {
        "dataset": "HotpotQA",
        "num_examples": len(examples),
        "train_examples": len(train),
        "seed": seed,
        "k": k,
        "horizon": 2,
        "target_mode": target_mode,
        "model": model,
        "knn_k": knn_k,
        "actions": ACTIONS,
        "rewrite_actions": REWRITE_ACTIONS,
    }
    save_checkpoint(q0_checkpoint, q0, {**checkpoint_metadata, "stage": "q0"})
    save_checkpoint(q1_checkpoint, q1, {**checkpoint_metadata, "stage": "q1"})

    metadata: dict[str, object] = {
        "num_examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "seed": seed,
        "k": k,
        "horizon": 2,
        "target_mode": target_mode,
        "model": model,
        "knn_k": knn_k,
        "best_fixed_trace": " -> ".join(best_fixed_trace),
        "retrieval_cost": cost_config.retrieval,
        "token_cost": cost_config.token,
        "outputs": {
            "detailed_csv": str(detailed_csv),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "table_tex": str(table_tex),
            "q0_checkpoint": str(q0_checkpoint),
            "q1_checkpoint": str(q1_checkpoint),
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def make_q0_rows(
    train: list[QAExample],
    q1: RidgeQ | KnnQ,
    k: int,
    target_mode: str,
    cost_config: CostConfig,
) -> list[tuple[np.ndarray, str, float]]:
    rows: list[tuple[np.ndarray, str, float]] = []
    for ex in train:
        retriever = BM25Retriever(ex.passages)
        s0 = initial_state(ex, retriever, k)
        for action in ACTIONS:
            if target_mode == "exact":
                target = exact_two_step_value(ex, retriever, s0, action, k, cost_config)
            elif action == "stop":
                target = terminal_reward(ex, s0, k)
            else:
                s1 = transition(ex, retriever, s0, action, k, cost_config)
                second = q1.choose(features(s1))
                target = exact_one_step_value(ex, retriever, s1, second, k, cost_config)
            rows.append((features(s0), action, target))
    return rows


def make_q_model(model: str, knn_k: int) -> RidgeQ | KnnQ:
    return RidgeQ(ACTIONS) if model == "ridge" else KnnQ(ACTIONS, k=knn_k)


def initial_state(ex: QAExample, retriever: BM25Retriever, k: int) -> State:
    return State(ex.question, retriever.search(ex.question, k=k), 0, 0.0, 1, "start")


def transition(
    ex: QAExample,
    retriever: BM25Retriever,
    state: State,
    action: str,
    k: int,
    cost_config: CostConfig,
) -> State:
    if action == "stop":
        return state
    rewritten = apply_action(state, action)
    results = merge_results([retriever.search(q, k=k) for q in rewritten.queries], k)
    return State(
        query=rewritten.joined_query,
        results=results,
        step=state.step + 1,
        cost=state.cost + action_cost(state.query, rewritten.queries, cost_config),
        calls=state.calls + rewritten.retrieval_calls,
        previous_action=action,
    )


def apply_action(state: State, action: str) -> RewriteOutput:
    if action in {"keyword_compress", "entity_expand", "decompose"}:
        return rewrite(state.query, action)
    top_titles = [r.doc_id for r in state.results[:2]]
    content = content_tokens(state.query)
    if action == "feedback_expand":
        query = " ".join(unique_preserve_order([*top_titles, *content])) or state.query
        return RewriteOutput(action=action, queries=[query])
    if action == "title_bridge":
        if not top_titles:
            return RewriteOutput(action=action, queries=[state.query])
        tail = " ".join(content[:8])
        return RewriteOutput(action=action, queries=[f"{title} {tail}".strip() for title in top_titles])
    raise ValueError(f"Unknown action: {action}")


def action_cost(query: str, rewritten_queries: list[str], cost_config: CostConfig) -> float:
    old_len = max(1, len(content_tokens(query)))
    new_len = sum(len(content_tokens(q)) for q in rewritten_queries)
    return cost_config.retrieval * len(rewritten_queries) + cost_config.token * max(0, new_len - old_len)


def terminal_reward(ex: QAExample, state: State, k: int) -> float:
    return recall_at_k(state.results, ex.gold_doc_ids, k) + 0.5 * mrr(state.results, ex.gold_doc_ids) - state.cost


def exact_one_step_value(
    ex: QAExample, retriever: BM25Retriever, state: State, action: str, k: int, cost_config: CostConfig
) -> float:
    if action == "stop":
        return terminal_reward(ex, state, k)
    return terminal_reward(ex, transition(ex, retriever, state, action, k, cost_config), k)


def exact_two_step_value(
    ex: QAExample, retriever: BM25Retriever, state: State, action: str, k: int, cost_config: CostConfig
) -> float:
    if action == "stop":
        return terminal_reward(ex, state, k)
    s1 = transition(ex, retriever, state, action, k, cost_config)
    return max(exact_one_step_value(ex, retriever, s1, a, k, cost_config) for a in ACTIONS)


def features(state: State) -> np.ndarray:
    base = state_features(state.query, state.results)
    previous = np.zeros(len(ACTIONS) + 1, dtype=float)
    previous[["start", *ACTIONS].index(state.previous_action)] = 1.0
    extra = np.asarray([state.step / 2.0, min(state.cost, 1.0), min(state.calls, 5) / 5.0], dtype=float)
    return np.concatenate([base, extra, previous])


def evaluate_example(
    ex: QAExample,
    split_name: str,
    q0: RidgeQ | KnnQ,
    q1: RidgeQ | KnnQ,
    best_fixed_trace: tuple[str, ...],
    k: int,
    cost_config: CostConfig,
) -> list[dict[str, object]]:
    retriever = BM25Retriever(ex.passages)
    s0 = initial_state(ex, retriever, k)
    rows = [
        row(ex, split_name, "Vanilla BM25", s0, "stop", k),
        row(
            ex,
            split_name,
            "Rewrite-all keyword",
            transition(ex, retriever, s0, "keyword_compress", k, cost_config),
            "keyword_compress",
            k,
        ),
        row(
            ex,
            split_name,
            "Rewrite-all entity",
            transition(ex, retriever, s0, "entity_expand", k, cost_config),
            "entity_expand",
            k,
        ),
    ]
    fixed_state = apply_trace(ex, retriever, s0, best_fixed_trace, k, cost_config)
    rows.append(row(ex, split_name, "Train-best fixed trace", fixed_state, " -> ".join(best_fixed_trace), k))

    first = q0.choose(features(s0))
    if first == "stop":
        final = s0
        trace = "stop"
    else:
        s1 = transition(ex, retriever, s0, first, k, cost_config)
        second = q1.choose(features(s1))
        final = s1 if second == "stop" else transition(ex, retriever, s1, second, k, cost_config)
        trace = f"{first} -> {second}"
    rows.append(row(ex, split_name, "Multi-step FQI", final, trace, k))
    rows.append(row(ex, split_name, "Oracle two-step", *oracle_two_step(ex, retriever, s0, k, cost_config), k))
    return rows


def select_best_fixed_trace(
    train: list[QAExample],
    k: int,
    cost_config: CostConfig,
) -> tuple[str, ...]:
    candidates = fixed_trace_candidates()
    scores: dict[tuple[str, ...], list[float]] = {trace: [] for trace in candidates}
    for ex in train:
        retriever = BM25Retriever(ex.passages)
        s0 = initial_state(ex, retriever, k)
        for trace in candidates:
            final = apply_trace(ex, retriever, s0, trace, k, cost_config)
            scores[trace].append(terminal_reward(ex, final, k))
    return max(
        candidates,
        key=lambda trace: (float(np.mean(scores[trace])), -candidates.index(trace)),
    )


def fixed_trace_candidates() -> list[tuple[str, ...]]:
    candidates: list[tuple[str, ...]] = [("stop",)]
    for first in REWRITE_ACTIONS:
        candidates.append((first, "stop"))
        for second in REWRITE_ACTIONS:
            candidates.append((first, second))
    return candidates


def apply_trace(
    ex: QAExample,
    retriever: BM25Retriever,
    s0: State,
    trace: tuple[str, ...],
    k: int,
    cost_config: CostConfig,
) -> State:
    state = s0
    for action in trace:
        if action == "stop":
            break
        state = transition(ex, retriever, state, action, k, cost_config)
    return state


def oracle_two_step(
    ex: QAExample,
    retriever: BM25Retriever,
    s0: State,
    k: int,
    cost_config: CostConfig,
) -> tuple[State, str]:
    best_value = float("-inf")
    best_state = s0
    best_trace = "stop"
    for first_action in ACTIONS:
        if first_action == "stop":
            candidates = [("stop", s0)]
        else:
            s1 = transition(ex, retriever, s0, first_action, k, cost_config)
            candidates = [
                (
                    f"{first_action} -> {second}",
                    s1 if second == "stop" else transition(ex, retriever, s1, second, k, cost_config),
                )
                for second in ACTIONS
            ]
        for trace, candidate in candidates:
            value = terminal_reward(ex, candidate, k)
            if value > best_value:
                best_value = value
                best_state = candidate
                best_trace = trace
    return best_state, best_trace


def row(ex: QAExample, split_name: str, method: str, state: State, action_trace: str, k: int) -> dict[str, object]:
    return {
        "split": split_name,
        "method": method,
        "qid": ex.qid,
        "question": ex.question,
        "action_trace": action_trace,
        "recall_at_5": recall_at_k(state.results, ex.gold_doc_ids, k),
        "mrr": mrr(state.results, ex.gold_doc_ids),
        "ndcg_at_5": ndcg_at_k(state.results, ex.gold_doc_ids, k),
        "reward": terminal_reward(ex, state, k),
        "cost": state.cost,
        "retrieval_calls": state.calls,
        "top_docs": " | ".join(r.doc_id for r in state.results),
        "gold_docs": " | ".join(sorted(ex.gold_doc_ids)),
    }


def merge_results(result_lists: list[list[RetrievalResult]], k: int) -> list[RetrievalResult]:
    best: dict[str, float] = {}
    for results in result_lists:
        for result in results:
            best[result.doc_id] = max(best.get(result.doc_id, float("-inf")), result.score)
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [RetrievalResult(doc_id=doc_id, score=score, rank=i + 1) for i, (doc_id, score) in enumerate(ranked)]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"] == "test"]
    rows = []
    for method in [
        "Vanilla BM25",
        "Rewrite-all keyword",
        "Rewrite-all entity",
        "Train-best fixed trace",
        "Multi-step FQI",
        "Oracle two-step",
    ]:
        part = test[test["method"] == method]
        rows.append(
            {
                "method": method,
                "recall_at_5": float(part["recall_at_5"].mean()),
                "mrr": float(part["mrr"].mean()),
                "ndcg_at_5": float(part["ndcg_at_5"].mean()),
                "reward": float(part["reward"].mean()),
                "cost": float(part["cost"].mean()),
                "retrieval_calls": float(part["retrieval_calls"].mean()),
            }
        )
    return pd.DataFrame(rows)


def latex_table(summary: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Recall@5 & MRR & nDCG@5 & Reward & Cost & Calls \\\\",
        "\\midrule",
    ]
    for row_ in summary.to_dict(orient="records"):
        lines.append(
            f"{row_['method']} & {row_['recall_at_5']:.3f} & {row_['mrr']:.3f} & "
            f"{row_['ndcg_at_5']:.3f} & {row_['reward']:.3f} & {row_['cost']:.3f} & "
            f"{row_['retrieval_calls']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def plot_summary(summary: pd.DataFrame, figures_dir: Path) -> None:
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - 0.2, summary["recall_at_5"], 0.2, label="Recall@5")
    ax.bar(x, summary["mrr"], 0.2, label="MRR")
    ax.bar(x + 0.2, summary["ndcg_at_5"], 0.2, label="nDCG@5")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "multistep_metrics.png", dpi=180)
    plt.close(fig)


def plot_policy_actions(df: pd.DataFrame, figures_dir: Path) -> None:
    selected = df[(df["split"] == "test") & (df["method"] == "Multi-step FQI")]
    counts = selected["action_trace"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(counts.index, counts.values, color="#59A14F")
    ax.set_ylabel("Number of test questions")
    ax.set_xlabel("Action trace")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "multistep_action_traces.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
