from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from selective_rag_rl.experiments.cross_dataset_transfer import TransferInputs, run_transfer_from_evals


def test_run_transfer_from_evals_applies_source_policy_to_target(tmp_path: Path) -> None:
    actions = ["left", "right"]
    source_train = [
        _eval_row("s1", "source left", [1.0, 0.0], {"left": 1.0, "right": 0.0}),
        _eval_row("s2", "source right", [0.0, 1.0], {"left": 0.0, "right": 1.0}),
        _eval_row("s3", "source left again", [1.0, 0.0], {"left": 1.0, "right": 0.0}),
        _eval_row("s4", "source right again", [0.0, 1.0], {"left": 0.0, "right": 1.0}),
    ]
    target_train = [
        _eval_row("tt1", "target train right", [1.0, 0.0], {"left": 0.1, "right": 0.9}),
        _eval_row("tt2", "target train right again", [0.0, 1.0], {"left": 0.2, "right": 0.8}),
    ]
    target_test = [
        _eval_row("t1", "target left", [1.0, 0.0], {"left": 0.7, "right": 0.1}),
        _eval_row("t2", "target right", [0.0, 1.0], {"left": 0.2, "right": 0.8}),
    ]

    metadata = run_transfer_from_evals(
        TransferInputs(
            source_dataset="source",
            target_dataset="target",
            source_train=source_train,
            target_train=target_train,
            target_test=target_test,
            actions=actions,
            method_names={"left": "Left fixed action", "right": "Right fixed action"},
            output_dir=tmp_path,
        )
    )

    summary = pd.read_csv(metadata["summary_csv"])
    assert "Transferred source policy" in set(summary["method"])
    assert "Target train-best action" in set(summary["method"])
    assert "Oracle retrieval action" in set(summary["method"])
    assert set(summary["source_dataset"]) == {"source"}
    assert set(summary["target_dataset"]) == {"target"}

    detailed = pd.read_csv(metadata["detailed_csv"])
    transferred = detailed[detailed["method"] == "Transferred source policy"]
    assert len(transferred) == len(target_test)
    assert set(transferred["action"]) == {"left", "right"}
    assert set(transferred["source_dataset"]) == {"source"}
    assert set(transferred["target_dataset"]) == {"target"}

    heuristic = detailed[detailed["method"] == "Heuristic retrieval router"]
    assert len(heuristic) == len(target_test)
    assert set(heuristic["action"]) == {"left"}

    target_best = detailed[detailed["method"] == "Target train-best action"]
    assert len(target_best) == len(target_test)
    assert set(target_best["action"]) == {"right"}


def _eval_row(qid: str, question: str, features: list[float], rewards: dict[str, float]) -> dict[str, object]:
    return {
        "qid": qid,
        "question": question,
        "features": np.asarray(features, dtype=float),
        "actions": {
            action: {
                "recall_at_5": reward,
                "mrr": reward,
                "ndcg_at_5": reward,
                "rewrite_cost": 0.0,
                "retrieval_calls": 1,
                "rewrite_tokens": 1,
                "reward": reward,
                "queries": question,
                "top_docs": action,
                "gold_docs": action,
            }
            for action, reward in rewards.items()
        },
    }
