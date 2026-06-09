from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_rag_rl.diagnostics.fqi_diagnostics import export_fqi_diagnostics


def test_export_fqi_diagnostics_reports_reward_gaps_and_trace_rates(tmp_path: Path) -> None:
    detailed = pd.DataFrame(
        [
            _row("q1", "Train-best fixed trace", "keyword_compress -> stop", 1.0, 2.0),
            _row("q1", "Multi-step FQI", "stop", 0.8, 1.0),
            _row("q1", "Oracle two-step", "keyword_compress -> title_bridge", 1.2, 2.0),
            _row("q2", "Train-best fixed trace", "keyword_compress -> stop", 0.5, 2.0),
            _row("q2", "Multi-step FQI", "keyword_compress -> stop", 0.7, 2.0),
            _row("q2", "Oracle two-step", "keyword_compress -> stop", 0.7, 2.0),
        ]
    )
    detailed_csv = tmp_path / "multistep_detailed.csv"
    detailed.to_csv(detailed_csv, index=False)

    exported = export_fqi_diagnostics(
        detailed_csv=detailed_csv,
        summary_csv=tmp_path / "summary.csv",
        trace_csv=tmp_path / "traces.csv",
        dataset="hotpot",
        split="test",
    )

    summary = pd.read_csv(exported["summary_csv"]).iloc[0]
    traces = pd.read_csv(exported["trace_csv"]).set_index("action_trace")

    assert summary["dataset"] == "hotpot"
    assert summary["split"] == "test"
    assert summary["examples"] == 2
    assert summary["fqi_reward"] == 0.75
    assert summary["train_best_reward"] == 0.75
    assert summary["oracle_reward"] == 0.95
    assert summary["reward_gap_vs_train_best"] == 0.0
    assert summary["oracle_trace_match_rate"] == 0.5
    assert summary["stop_rate"] == 0.5
    assert traces.loc["stop", "count"] == 1
    assert traces.loc["keyword_compress -> stop", "rate"] == 0.5


def _row(qid: str, method: str, trace: str, reward: float, calls: float) -> dict[str, object]:
    return {
        "split": "test",
        "method": method,
        "qid": qid,
        "action_trace": trace,
        "recall_at_5": reward,
        "mrr": reward,
        "ndcg_at_5": reward,
        "reward": reward,
        "cost": 0.0,
        "retrieval_calls": calls,
    }
