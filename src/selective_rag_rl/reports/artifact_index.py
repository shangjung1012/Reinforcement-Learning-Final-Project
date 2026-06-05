from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd


ARTIFACT_INDEX_COLUMNS = [
    "artifact_id",
    "category",
    "path",
    "exists",
    "size_bytes",
    "sha256_12",
    "role",
    "producer_command",
]


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    category: str
    path: Path
    role: str
    producer_command: str


def export_artifact_index(
    specs: list[ArtifactSpec],
    *,
    output_csv: Path,
    root: Path | None = None,
) -> Path:
    root = (root or Path.cwd()).resolve()
    rows = [_artifact_row(spec, root=root) for spec in specs]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=ARTIFACT_INDEX_COLUMNS).to_csv(output_csv, index=False)
    return output_csv


def final_project_artifact_specs(root: Path | None = None) -> list[ArtifactSpec]:
    root = root or Path(".")

    def artifact(
        artifact_id: str,
        category: str,
        relative_path: str,
        role: str,
        producer_command: str,
    ) -> ArtifactSpec:
        return ArtifactSpec(
            artifact_id=artifact_id,
            category=category,
            path=root / relative_path,
            role=role,
            producer_command=producer_command,
        )

    return [
        artifact(
            "final_report",
            "document",
            "FINAL_REPORT.md",
            "Human-readable final project narrative and main result tables.",
            "manual report update from generated outputs",
        ),
        artifact(
            "readme",
            "document",
            "README.md",
            "Reproduction commands, scope, and project inventory.",
            "manual documentation update",
        ),
        artifact(
            "final_presentation_outline",
            "document",
            "FINAL_PRESENTATION_OUTLINE.md",
            "Presentation slide outline and defense answers grounded in generated project evidence.",
            "manual outline update from outputs/results/final_claims_matrix.csv",
        ),
        artifact(
            "final_slides_markdown",
            "document",
            "FINAL_SLIDES.md",
            "Markdown slide draft with per-slide message, evidence files, figures, and defense points.",
            "manual Markdown defense package update from final paper assets",
        ),
        artifact(
            "final_defense_qa",
            "document",
            "FINAL_DEFENSE_QA.md",
            "Markdown defense Q&A covering RL framing, baselines, related work, cost reward, OPE, and limitations.",
            "manual Markdown defense package update from final paper assets",
        ),
        artifact(
            "final_results_summary",
            "document",
            "FINAL_RESULTS_SUMMARY.md",
            "Markdown one-page result summary with main deltas, confidence intervals, evidence files, and limitations.",
            "manual Markdown defense package update from final paper assets",
        ),
        artifact(
            "hotpot_retrieval_policy_summary",
            "main_result",
            "outputs/results/retrieval_policy_summary.csv",
            "HotpotQA cost-aware retrieval-action policy summary.",
            "uv run python scripts/run_retrieval_policy_hotpot.py --num-examples 300 --seed 42",
        ),
        artifact(
            "hotpot_llm_policy_summary",
            "main_result",
            "outputs/results/llm_retrieval_policy_summary.csv",
            "HotpotQA retrieval-action policy with optional Vertex Gemini actions.",
            "uv run python scripts/run_llm_retrieval_policy_hotpot.py --num-examples 20 --seed 42 --policy-model auto",
        ),
        artifact(
            "hotpot_reader_realdata_summary",
            "reader_smoke",
            "outputs/results/hotpot_reader_realdata_summary.csv",
            "Tiny HotpotQA real-data downstream reader comparison for lexical and span heuristic readers; not final QA benchmark evidence.",
            "uv run python scripts/run_reader_comparison.py --dataset hotpot --num-examples 50 --readers lexical,span --output-dir outputs/codex_reader_hotpot_realdata_50",
        ),
        artifact(
            "hotpot_reader_realdata_200_summary",
            "reader_smoke",
            "outputs/results/hotpot_reader_realdata_200_summary.csv",
            "Larger HotpotQA downstream reader comparison for lexical, span, and answer-type heuristic readers; still not final QA benchmark evidence.",
            "uv run python scripts/run_reader_comparison.py --dataset hotpot --num-examples 200 --readers lexical,span,answer_type --output-dir outputs/codex_reader_hotpot_realdata_200",
        ),
        artifact(
            "nq_reader_realdata_summary",
            "reader_smoke",
            "outputs/results/nq_reader_realdata_summary.csv",
            "Natural Questions downstream reader comparison for lexical, span, and answer-type heuristic readers; small real-data diagnostic only.",
            "uv run python scripts/run_reader_comparison.py --dataset nq --num-examples 50 --readers lexical,span,answer_type --output-dir outputs/codex_reader_nq_realdata_50",
        ),
        artifact(
            "hotpot_gemini_pilot_summary",
            "api_pilot",
            "outputs/results/hotpot_gemini_pilot_summary.csv",
            "Bounded HotpotQA Gemini rewrite/decompose baseline pilot with at most 8 new Gemini calls; not final generated-action evidence.",
            "CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_gemini_baseline.py --data-path data/raw/HotpotQA/hotpot_dev_distractor_v1.json --num-examples 10 --seed 42 --cache-path outputs/cache/codex_gemini_rewrites_realdata.jsonl --allow-api --max-new-calls 8 --output-dir outputs/codex_gemini_realdata_pilot",
        ),
        artifact(
            "hotpot_gemini_repeated_pilot_summary",
            "api_pilot",
            "outputs/results/hotpot_gemini_repeated_pilot_summary.csv",
            "Repeated-seed HotpotQA Gemini rewrite/decompose pilot with 24 cache hits and 0 new calls in this run; not final generated-action evidence.",
            "CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_repeated_gemini_baseline.py --data-path data/raw/HotpotQA/hotpot_dev_distractor_v1.json --seeds 41,42,43 --num-examples 10 --cache-path outputs/cache/codex_gemini_repeated_realdata.jsonl --allow-api --max-new-calls 24 --output-dir outputs/codex_gemini_repeated_realdata_pilot",
        ),
        artifact(
            "hotpot_multistep_fqi_summary",
            "rl_extension",
            "outputs/results/multistep_summary.csv",
            "HotpotQA two-step FQI extension summary; useful for RL framing but not the final BEIR benchmark claim.",
            "uv run python scripts/run_multistep_hotpot.py --num-examples 600 --seed 42",
        ),
        artifact(
            "hotpot_multistep_fqi_detailed",
            "rl_extension",
            "outputs/results/multistep_detailed.csv",
            "HotpotQA two-step FQI per-example trace diagnostics with selected trace, reward, and retrieval metrics.",
            "uv run python scripts/run_multistep_hotpot.py --num-examples 600 --seed 42",
        ),
        artifact(
            "hotpot_multistep_fqi_metadata",
            "rl_extension",
            "outputs/results/multistep_metadata.json",
            "HotpotQA two-step FQI metadata recording horizon, seed, cost settings, and checkpoint paths.",
            "uv run python scripts/run_multistep_hotpot.py --num-examples 600 --seed 42",
        ),
        artifact(
            "hotpot_fqi_diagnostics_summary",
            "rl_extension",
            "outputs/results/hotpot_fqi_diagnostics_summary.csv",
            "Post-hoc HotpotQA FQI diagnostics comparing Multi-step FQI against train-best fixed trace and the two-step oracle.",
            "uv run python scripts/run_fqi_diagnostics.py --dataset hotpot --detailed-csv outputs/results/multistep_detailed.csv --summary-csv outputs/results/hotpot_fqi_diagnostics_summary.csv --trace-csv outputs/results/hotpot_fqi_trace_distribution.csv --split test",
        ),
        artifact(
            "hotpot_fqi_trace_distribution",
            "rl_extension",
            "outputs/results/hotpot_fqi_trace_distribution.csv",
            "HotpotQA Multi-step FQI selected action-trace distribution with reward and retrieval-call diagnostics.",
            "uv run python scripts/run_fqi_diagnostics.py --dataset hotpot --detailed-csv outputs/results/multistep_detailed.csv --summary-csv outputs/results/hotpot_fqi_diagnostics_summary.csv --trace-csv outputs/results/hotpot_fqi_trace_distribution.csv --split test",
        ),
        artifact(
            "hotpot_multistep_metrics_figure",
            "paper_asset",
            "outputs/figures/multistep_metrics.png",
            "HotpotQA two-step FQI metric comparison figure.",
            "uv run python scripts/run_multistep_hotpot.py --num-examples 600 --seed 42",
        ),
        artifact(
            "hotpot_multistep_action_traces",
            "paper_asset",
            "outputs/figures/multistep_action_traces.png",
            "HotpotQA two-step FQI action-trace frequency figure.",
            "uv run python scripts/run_multistep_hotpot.py --num-examples 600 --seed 42",
        ),
        artifact(
            "scifact_policy_summary",
            "main_result",
            "outputs/results/scifact_retrieval_policy_summary.csv",
            "BEIR SciFact full-corpus retrieval-action policy summary.",
            "uv run python scripts/run_retrieval_policy_scifact.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --policy-model auto",
        ),
        artifact(
            "nfcorpus_policy_summary",
            "main_result",
            "outputs/results/nfcorpus_retrieval_policy_summary.csv",
            "BEIR NFCorpus full-corpus retrieval-action policy summary.",
            "uv run python scripts/run_retrieval_policy_nfcorpus.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --policy-model auto",
        ),
        artifact(
            "scifact_linucb_baseline_summary",
            "bandit_baseline",
            "outputs/results/scifact_linucb_baseline_summary.csv",
            "SciFact full-corpus LinUCB contextual-bandit baseline under selected-action feedback.",
            "uv run python scripts/run_bandit_baselines.py --dataset scifact --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --alpha 1.0",
        ),
        artifact(
            "scifact_linucb_baseline_history",
            "bandit_diagnostic",
            "outputs/results/scifact_linucb_baseline_history.csv",
            "SciFact LinUCB replay history with chosen action, observed reward, oracle reward, and regret per training query.",
            "uv run python scripts/run_bandit_baselines.py --dataset scifact --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --alpha 1.0",
        ),
        artifact(
            "nfcorpus_linucb_baseline_summary",
            "bandit_baseline",
            "outputs/results/nfcorpus_linucb_baseline_summary.csv",
            "NFCorpus full-corpus LinUCB contextual-bandit baseline under selected-action feedback.",
            "uv run python scripts/run_bandit_baselines.py --dataset nfcorpus --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --alpha 1.0",
        ),
        artifact(
            "nfcorpus_linucb_baseline_history",
            "bandit_diagnostic",
            "outputs/results/nfcorpus_linucb_baseline_history.csv",
            "NFCorpus LinUCB replay history with chosen action, observed reward, oracle reward, and regret per training query.",
            "uv run python scripts/run_bandit_baselines.py --dataset nfcorpus --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --alpha 1.0",
        ),
        artifact(
            "scifact_bandit_replay_summary",
            "bandit_diagnostic",
            "outputs/results/scifact_bandit_replay_summary.csv",
            "Full-corpus SciFact selected-action replay summary comparing full-information direct method, LinUCB, epsilon-greedy, Thompson sampling, train-best, and oracle policies.",
            "uv run python scripts/run_bandit_replay_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-dir outputs --split train --seed 42 --moving-average-window 50",
        ),
        artifact(
            "scifact_bandit_replay_history",
            "bandit_diagnostic",
            "outputs/results/scifact_bandit_replay_history.csv",
            "Full-corpus SciFact per-step selected-action replay history with cumulative regret, oracle match rate, and action entropy.",
            "uv run python scripts/run_bandit_replay_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-dir outputs --split train --seed 42 --moving-average-window 50",
        ),
        artifact(
            "nfcorpus_bandit_replay_summary",
            "bandit_diagnostic",
            "outputs/results/nfcorpus_bandit_replay_summary.csv",
            "Full-corpus NFCorpus selected-action replay summary comparing full-information direct method, LinUCB, epsilon-greedy, Thompson sampling, train-best, and oracle policies.",
            "uv run python scripts/run_bandit_replay_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-dir outputs --split train --seed 42 --moving-average-window 50",
        ),
        artifact(
            "nfcorpus_bandit_replay_history",
            "bandit_diagnostic",
            "outputs/results/nfcorpus_bandit_replay_history.csv",
            "Full-corpus NFCorpus per-step selected-action replay history with cumulative regret, oracle match rate, and action entropy.",
            "uv run python scripts/run_bandit_replay_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-dir outputs --split train --seed 42 --moving-average-window 50",
        ),
        artifact(
            "scifact_bandit_replay_regret_figure",
            "paper_asset",
            "outputs/figures/scifact_bandit_replay_regret.png",
            "SciFact cumulative regret curve for selected-action bandit replay diagnostics.",
            "uv run python scripts/run_bandit_replay_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-dir outputs --split train --seed 42 --moving-average-window 50",
        ),
        artifact(
            "nfcorpus_bandit_replay_regret_figure",
            "paper_asset",
            "outputs/figures/nfcorpus_bandit_replay_regret.png",
            "NFCorpus cumulative regret curve for selected-action bandit replay diagnostics.",
            "uv run python scripts/run_bandit_replay_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-dir outputs --split train --seed 42 --moving-average-window 50",
        ),
        artifact(
            "scifact_budget_curve",
            "budget_curve",
            "outputs/results/scifact_budget_curve.csv",
            "SciFact retrieval-call budget curve selecting the best non-oracle method under each call budget.",
            "uv run python scripts/run_budget_curve.py --dataset scifact_main --summary-csv outputs/results/scifact_retrieval_policy_summary.csv --output-csv outputs/results/scifact_budget_curve.csv --budgets 1.0,1.25,1.5,2.0",
        ),
        artifact(
            "nfcorpus_budget_curve",
            "budget_curve",
            "outputs/results/nfcorpus_budget_curve.csv",
            "NFCorpus retrieval-call budget curve selecting the best non-oracle method under each call budget.",
            "uv run python scripts/run_budget_curve.py --dataset nfcorpus_main --summary-csv outputs/results/nfcorpus_retrieval_policy_summary.csv --output-csv outputs/results/nfcorpus_budget_curve.csv --budgets 1.0,1.25,1.5,2.0",
        ),
        artifact(
            "scifact_ope_diagnostics",
            "off_policy_evaluation",
            "outputs/results/scifact_ope_diagnostics.csv",
            "SciFact offline contextual-bandit OPE diagnostics comparing DM, IPS, SNIPS, and doubly robust estimates against full-information reward.",
            "uv run python scripts/run_ope_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_ope_diagnostics.csv --seed 42",
        ),
        artifact(
            "nfcorpus_ope_diagnostics",
            "off_policy_evaluation",
            "outputs/results/nfcorpus_ope_diagnostics.csv",
            "NFCorpus offline contextual-bandit OPE diagnostics comparing DM, IPS, SNIPS, and doubly robust estimates against full-information reward.",
            "uv run python scripts/run_ope_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_ope_diagnostics.csv --seed 42",
        ),
        artifact(
            "scifact_ope_stability",
            "off_policy_evaluation",
            "outputs/results/scifact_ope_stability.csv",
            "SciFact repeated-seed OPE stability summary with mean estimator error, uncertainty intervals, match rate, and effective sample size.",
            "uv run python scripts/run_ope_stability.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_ope_stability.csv --seeds 1,2,3,4,5,6,7,8,9,10",
        ),
        artifact(
            "nfcorpus_ope_stability",
            "off_policy_evaluation",
            "outputs/results/nfcorpus_ope_stability.csv",
            "NFCorpus repeated-seed OPE stability summary with mean estimator error, uncertainty intervals, match rate, and effective sample size.",
            "uv run python scripts/run_ope_stability.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_ope_stability.csv --seeds 1,2,3,4,5,6,7,8,9,10",
        ),
        artifact(
            "scifact_constrained_policy_sweep",
            "constrained_bandit",
            "outputs/results/scifact_constrained_policy_sweep.csv",
            "SciFact Lagrangian call-penalty sweep for a constrained direct-method retrieval policy.",
            "uv run python scripts/run_constrained_policy_sweep.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_constrained_policy_sweep.csv --call-penalties 0,0.01,0.03,0.06,0.1,0.2",
        ),
        artifact(
            "nfcorpus_constrained_policy_sweep",
            "constrained_bandit",
            "outputs/results/nfcorpus_constrained_policy_sweep.csv",
            "NFCorpus Lagrangian call-penalty sweep for a constrained direct-method retrieval policy.",
            "uv run python scripts/run_constrained_policy_sweep.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_constrained_policy_sweep.csv --call-penalties 0,0.01,0.03,0.06,0.1,0.2",
        ),
        artifact(
            "scifact_constrained_policy_bootstrap",
            "constrained_bandit",
            "outputs/results/scifact_constrained_policy_bootstrap.csv",
            "SciFact paired bootstrap confidence intervals for constrained-policy utility and retrieval-call deltas versus the train-selected fixed action.",
            "uv run python scripts/run_constrained_policy_bootstrap.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_constrained_policy_bootstrap.csv --call-penalties 0,0.01,0.03,0.06,0.1,0.2 --bootstrap-samples 1000 --seed 42",
        ),
        artifact(
            "nfcorpus_constrained_policy_bootstrap",
            "constrained_bandit",
            "outputs/results/nfcorpus_constrained_policy_bootstrap.csv",
            "NFCorpus paired bootstrap confidence intervals for constrained-policy utility and retrieval-call deltas versus the train-selected fixed action.",
            "uv run python scripts/run_constrained_policy_bootstrap.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_constrained_policy_bootstrap.csv --call-penalties 0,0.01,0.03,0.06,0.1,0.2 --bootstrap-samples 1000 --seed 42",
        ),
        artifact(
            "scifact_policy_diagnostics",
            "diagnostic",
            "outputs/results/scifact_policy_diagnostics.csv",
            "Per-query policy regret and action-change diagnostics for SciFact.",
            "uv run python scripts/run_policy_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_policy_diagnostics.csv",
        ),
        artifact(
            "nfcorpus_policy_diagnostics",
            "diagnostic",
            "outputs/results/nfcorpus_policy_diagnostics.csv",
            "Per-query policy regret and action-change diagnostics for NFCorpus.",
            "uv run python scripts/run_policy_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_policy_diagnostics.csv",
        ),
        artifact(
            "scifact_bootstrap",
            "statistical_check",
            "outputs/results/scifact_bootstrap_diagnostics.csv",
            "Paired bootstrap intervals for SciFact policy-vs-baseline reward gaps.",
            "uv run python scripts/run_statistical_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_bootstrap_diagnostics.csv",
        ),
        artifact(
            "nfcorpus_bootstrap",
            "statistical_check",
            "outputs/results/nfcorpus_bootstrap_diagnostics.csv",
            "Paired bootstrap intervals for NFCorpus policy-vs-baseline reward gaps.",
            "uv run python scripts/run_statistical_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_bootstrap_diagnostics.csv",
        ),
        artifact(
            "nfcorpus_learning_curve",
            "diagnostic",
            "outputs/results/nfcorpus_policy_learning_curve.csv",
            "NFCorpus train-size sensitivity for policy reward.",
            "uv run python scripts/run_policy_learning_curve.py --dataset nfcorpus --train-sizes 50,100,200 --num-test-examples 150 --seed 42 --full-corpus --policy-model auto",
        ),
        artifact(
            "scifact_complexity_buckets",
            "diagnostic",
            "outputs/results/scifact_complexity_buckets.csv",
            "SciFact full-corpus query difficulty bucket diagnostics for policy behavior.",
            "uv run python scripts/run_complexity_diagnostics.py --dataset scifact --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_complexity_buckets.csv --action-distribution-csv outputs/results/scifact_complexity_action_distribution.csv",
        ),
        artifact(
            "nfcorpus_complexity_buckets",
            "diagnostic",
            "outputs/results/nfcorpus_complexity_buckets.csv",
            "NFCorpus full-corpus query difficulty bucket diagnostics for policy behavior.",
            "uv run python scripts/run_complexity_diagnostics.py --dataset nfcorpus --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_complexity_buckets.csv --action-distribution-csv outputs/results/nfcorpus_complexity_action_distribution.csv",
        ),
        artifact(
            "cross_dataset_transfer_summary",
            "transfer_check",
            "outputs/results/cross_dataset_transfer_summary.csv",
            "SciFact/NFCorpus cross-dataset transfer summary for source-trained retrieval policies.",
            "uv run python scripts/run_cross_dataset_transfer.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus",
        ),
        artifact(
            "nfcorpus_confidence_gate_smoke",
            "diagnostic",
            "outputs/confidence_gate_smoke/results/nfcorpus_retrieval_policy_summary.csv",
            "No-API NFCorpus smoke test for confidence-gated policy fallback based on predicted action-score margin.",
            "uv run python scripts/run_retrieval_policy_nfcorpus.py --output-dir outputs/confidence_gate_smoke --num-train-examples 12 --num-test-examples 8 --pool-size 20 --embedder fake --policy-model ridge --tuning-folds 2 --knn-k-candidates 1 --confidence-gate-margin 999",
        ),
        artifact(
            "nfcorpus_confidence_gate_sweep",
            "diagnostic",
            "outputs/results/nfcorpus_confidence_gate_smoke_sweep.csv",
            "No-API NFCorpus confidence-gate margin threshold sweep over predicted action-score margins.",
            "uv run python scripts/run_confidence_gate_sweep.py --dataset nfcorpus_confidence_gate_smoke --detailed-csv outputs/confidence_gate_smoke/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_confidence_gate_smoke_sweep.csv --margins 0,0.001,0.005,0.01,0.02,0.05,0.1,999",
        ),
        artifact(
            "nfcorpus_main_confidence_gate_sweep",
            "diagnostic",
            "outputs/results/nfcorpus_confidence_gate_sweep.csv",
            "full-corpus NFCorpus confidence-gate margin threshold sweep over predicted action-score margins.",
            "uv run python scripts/run_confidence_gate_sweep.py --dataset nfcorpus_main --detailed-csv outputs/results/nfcorpus_retrieval_policy_detailed.csv --output-csv outputs/results/nfcorpus_confidence_gate_sweep.csv --margins 0,0.001,0.005,0.01,0.02,0.05,0.1",
        ),
        artifact(
            "scifact_main_confidence_gate_sweep",
            "diagnostic",
            "outputs/results/scifact_confidence_gate_sweep.csv",
            "full-corpus SciFact confidence-gate margin threshold sweep over predicted action-score margins.",
            "uv run python scripts/run_confidence_gate_sweep.py --dataset scifact_main --detailed-csv outputs/results/scifact_retrieval_policy_detailed.csv --output-csv outputs/results/scifact_confidence_gate_sweep.csv --margins 0,0.001,0.005,0.01,0.02,0.05,0.1",
        ),
        artifact(
            "final_checkpoint_manifest",
            "checkpoint_manifest",
            "outputs/results/final_checkpoint_manifest.csv",
            "Inspectable summary of saved policy checkpoints, model classes, action spaces, and feature widths.",
            "uv run python scripts/run_checkpoint_manifest.py --output-csv outputs/results/final_checkpoint_manifest.csv",
        ),
        artifact(
            "final_evidence_consistency",
            "consistency_check",
            "outputs/results/final_evidence_consistency.csv",
            "Machine-readable consistency check tying FINAL_REPORT claims to protocol, deployment, and artifact-index evidence.",
            "uv run python scripts/run_evidence_consistency.py --output-csv outputs/results/final_evidence_consistency.csv",
        ),
        artifact(
            "final_markdown_consistency",
            "consistency_check",
            "outputs/results/final_markdown_consistency.csv",
            "Machine-readable consistency check tying final Markdown defense files to generated result values and artifact paths.",
            "uv run python scripts/run_markdown_consistency.py --output-csv outputs/results/final_markdown_consistency.csv",
        ),
        artifact(
            "final_claims_matrix",
            "defense_artifact",
            "outputs/results/final_claims_matrix.csv",
            "Machine-readable final defense matrix linking each main research claim to metrics, deltas, commands, and evidence artifacts.",
            "uv run python scripts/run_final_claims_matrix.py --output-csv outputs/results/final_claims_matrix.csv",
        ),
        artifact(
            "experiment_dashboard",
            "evidence_dashboard",
            "outputs/results/experiment_dashboard.csv",
            "Machine-readable evidence level dashboard separating final claims, full benchmarks, smoke checks, and API pilots.",
            "uv run python scripts/run_experiment_dashboard.py --output-csv outputs/results/experiment_dashboard.csv --output-md docs/EXPERIMENT_DASHBOARD.md",
        ),
        artifact(
            "experiment_dashboard_markdown",
            "document",
            "docs/EXPERIMENT_DASHBOARD.md",
            "Human-readable evidence level dashboard for avoiding smoke/API-pilot overclaiming.",
            "uv run python scripts/run_experiment_dashboard.py --output-csv outputs/results/experiment_dashboard.csv --output-md docs/EXPERIMENT_DASHBOARD.md",
        ),
        artifact(
            "poster_claim_audit",
            "document",
            "docs/POSTER_CLAIM_AUDIT.md",
            "Poster claim-boundary audit against the final retrieval-stage evidence and known limitations.",
            "manual claim-boundary audit against poster/poster.tex and outputs/results/final_claims_matrix.csv",
        ),
        artifact(
            "final_main_results_table",
            "paper_asset",
            "outputs/results/final_main_results_table.csv",
            "Paper-ready cross-dataset result table combining main policies, LinUCB, constrained policy, and oracle rows.",
            "uv run python scripts/run_final_paper_assets.py --results-dir outputs/results --figures-dir outputs/figures",
        ),
        artifact(
            "final_main_results_latex",
            "paper_asset",
            "outputs/results/final_main_results_table.tex",
            "LaTeX export of the paper-ready final main result table.",
            "uv run python scripts/run_final_paper_assets.py --results-dir outputs/results --figures-dir outputs/figures",
        ),
        artifact(
            "final_reward_delta_ci_figure",
            "paper_asset",
            "outputs/figures/final_reward_delta_ci.png",
            "Presentation figure for selective-policy reward deltas with paired-bootstrap confidence intervals.",
            "uv run python scripts/run_final_paper_assets.py --results-dir outputs/results --figures-dir outputs/figures",
        ),
        artifact(
            "final_cost_reward_frontier_figure",
            "paper_asset",
            "outputs/figures/final_cost_reward_frontier.png",
            "Presentation figure for constrained-policy utility versus retrieval-call frontier.",
            "uv run python scripts/run_final_paper_assets.py --results-dir outputs/results --figures-dir outputs/figures",
        ),
        artifact(
            "final_ope_estimator_error_figure",
            "paper_asset",
            "outputs/figures/final_ope_estimator_error.png",
            "Presentation figure comparing uniform-log OPE estimator absolute errors.",
            "uv run python scripts/run_final_paper_assets.py --results-dir outputs/results --figures-dir outputs/figures",
        ),
        artifact(
            "final_linucb_comparison_figure",
            "paper_asset",
            "outputs/figures/final_linucb_comparison.png",
            "Presentation figure comparing train-best, LinUCB, and direct-method selective policy rewards.",
            "uv run python scripts/run_final_paper_assets.py --results-dir outputs/results --figures-dir outputs/figures",
        ),
        artifact(
            "nfcorpus_vertex_repeated_selection_diagnostics",
            "selection_check",
            "outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_diagnostics.csv",
            "Cross-seed validation-selection guardrail diagnostics for Vertex semantic features.",
            "uv run python scripts/run_repeated_selection.py --dataset nfcorpus --seeds 41,42,43 --policy-models ridge,auto --feature-sets full,no_semantic --num-train-examples 30 --num-test-examples 30 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --semantic-depth 5 --knn-k-candidates 1,3 --tuning-folds 5 --auto-candidate-models knn,ridge,extra_trees,random_forest --output-dir outputs/repeated_selection_runs/nfcorpus_vertex_30x30x3",
        ),
        artifact(
            "nfcorpus_vertex_repeated_10x10_diagnostics",
            "selection_check",
            "outputs/results/nfcorpus_vertex_repeated_10x10_diagnostics.csv",
            "Tiny NFCorpus Vertex semantic-feature repeated-seed diagnostics with 208 new embedding texts; API pilot only.",
            "CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_repeated_selection.py --dataset nfcorpus --seeds 41,42,43 --policy-models ridge --feature-sets full,no_semantic --num-train-examples 10 --num-test-examples 10 --full-corpus --embedder fake --semantic-features vertex --semantic-cache-path outputs/cache/codex_nfcorpus_vertex_repeated_10x10.jsonl --semantic-allow-api --semantic-max-new-texts 90 --semantic-depth 3 --knn-k-candidates 1 --tuning-folds 2 --auto-candidate-models ridge --output-dir outputs/codex_vertex_repeated_10x10",
        ),
        artifact(
            "nfcorpus_vertex_repeated_10x10_stability",
            "selection_check",
            "outputs/results/nfcorpus_vertex_repeated_10x10_stability.csv",
            "Tiny NFCorpus Vertex semantic-feature selection stability; guardrail still falls back to train-best fixed.",
            "CODEX_ALLOW_API_CALLS=1 uv run python scripts/run_repeated_selection.py --dataset nfcorpus --seeds 41,42,43 --policy-models ridge --feature-sets full,no_semantic --num-train-examples 10 --num-test-examples 10 --full-corpus --embedder fake --semantic-features vertex --semantic-cache-path outputs/cache/codex_nfcorpus_vertex_repeated_10x10.jsonl --semantic-allow-api --semantic-max-new-texts 90 --semantic-depth 3 --knn-k-candidates 1 --tuning-folds 2 --auto-candidate-models ridge --output-dir outputs/codex_vertex_repeated_10x10",
        ),
        artifact(
            "nfcorpus_vertex_repeated_semantic_depth_diagnostics",
            "selection_check",
            "outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_selection_diagnostics.csv",
            "Cross-seed semantic-depth validation-vs-heldout selection diagnostics.",
            "uv run python scripts/run_repeated_semantic_depth_sweep.py --dataset nfcorpus --seeds 41,42,43 --semantic-depths 5,8 --policy-models ridge,auto --feature-sets full,no_semantic --num-train-examples 30 --num-test-examples 30 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --knn-k-candidates 1,3 --tuning-folds 5 --auto-candidate-models knn,ridge,extra_trees,random_forest --output-dir outputs/repeated_semantic_depth_runs/nfcorpus_vertex_repeated_depth_30x30x3",
        ),
        artifact(
            "nfcorpus_vertex_repeated_semantic_depth_stability",
            "selection_check",
            "outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_stability.csv",
            "Cross-seed semantic-depth effect stability with paired bootstrap confidence intervals.",
            "uv run python scripts/run_repeated_semantic_depth_sweep.py --dataset nfcorpus --seeds 41,42,43 --semantic-depths 5,8 --policy-models ridge,auto --feature-sets full,no_semantic --num-train-examples 30 --num-test-examples 30 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --knn-k-candidates 1,3 --tuning-folds 5 --auto-candidate-models knn,ridge,extra_trees,random_forest --output-dir outputs/repeated_semantic_depth_runs/nfcorpus_vertex_repeated_depth_30x30x3",
        ),
        artifact(
            "nfcorpus_vertex_repeated_semantic_depth_selection_stability",
            "selection_check",
            "outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_selection_stability.csv",
            "Cross-seed semantic-depth validation-match stability with Wilson confidence intervals.",
            "uv run python scripts/run_repeated_semantic_depth_sweep.py --dataset nfcorpus --seeds 41,42,43 --semantic-depths 5,8 --policy-models ridge,auto --feature-sets full,no_semantic --num-train-examples 30 --num-test-examples 30 --semantic-features vertex --semantic-cache-path outputs/cache/nfcorpus_pilot_vertex_embeddings.jsonl --knn-k-candidates 1,3 --tuning-folds 5 --auto-candidate-models knn,ridge,extra_trees,random_forest --output-dir outputs/repeated_semantic_depth_runs/nfcorpus_vertex_repeated_depth_30x30x3",
        ),
        artifact(
            "nfcorpus_vertex_protocol_summary",
            "decision_record",
            "outputs/results/nfcorpus_vertex_selection_protocol_summary.csv",
            "Machine-readable protocol combining policy and semantic-depth guardrails.",
            "uv run python scripts/run_selection_protocol_summary.py --dataset nfcorpus_vertex_30x30x3 --policy-diagnostics-csv outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_diagnostics.csv --depth-selection-diagnostics-csv outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_selection_diagnostics.csv --depth-stability-csv outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_stability.csv --output-csv outputs/results/nfcorpus_vertex_selection_protocol_summary.csv --deployment-decision-csv outputs/results/nfcorpus_vertex_deployment_decision.csv",
        ),
        artifact(
            "nfcorpus_vertex_deployment_decision",
            "decision_record",
            "outputs/results/nfcorpus_vertex_deployment_decision.csv",
            "Final runtime policy choice distilled from the selection protocol.",
            "uv run python scripts/run_selection_protocol_summary.py --dataset nfcorpus_vertex_30x30x3 --policy-diagnostics-csv outputs/results/nfcorpus_vertex_repeated_selection_30x30x3_diagnostics.csv --depth-selection-diagnostics-csv outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_selection_diagnostics.csv --depth-stability-csv outputs/results/nfcorpus_vertex_repeated_semantic_depth_30x30x3_stability.csv --output-csv outputs/results/nfcorpus_vertex_selection_protocol_summary.csv --deployment-decision-csv outputs/results/nfcorpus_vertex_deployment_decision.csv",
        ),
        artifact(
            "nfcorpus_policy_checkpoint",
            "checkpoint",
            "outputs/checkpoints/nfcorpus_retrieval_policy.pkl",
            "Serialized NFCorpus retrieval-action policy and feature transform.",
            "uv run python scripts/run_retrieval_policy_nfcorpus.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --policy-model auto",
        ),
        artifact(
            "scifact_policy_checkpoint",
            "checkpoint",
            "outputs/checkpoints/scifact_retrieval_policy.pkl",
            "Serialized SciFact retrieval-action policy and feature transform.",
            "uv run python scripts/run_retrieval_policy_scifact.py --num-train-examples 600 --num-test-examples 300 --seed 42 --full-corpus --policy-model auto",
        ),
    ]


def _artifact_row(spec: ArtifactSpec, *, root: Path) -> dict[str, object]:
    path = spec.path
    resolved_path = path if path.is_absolute() else (root / path).resolve()
    exists = resolved_path.exists()
    return {
        "artifact_id": spec.artifact_id,
        "category": spec.category,
        "path": _relative_path(resolved_path, root),
        "exists": exists,
        "size_bytes": resolved_path.stat().st_size if exists else 0,
        "sha256_12": _sha256_12(resolved_path) if exists else "",
        "role": spec.role,
        "producer_command": spec.producer_command,
    }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_12(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]
