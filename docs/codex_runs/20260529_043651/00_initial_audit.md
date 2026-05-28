# Initial Audit

## Current Branch And Commit

- Branch: `codex/autonomous-improvements-20260529-0436`
- Commit: `26924a9`
- Base branch before this run: `main`
- Working tree before branch creation: clean
- Remote: `origin https://github.com/shangjung1012/Reinforcement-Learning-Final-Project.git`

## Bootstrap Commands

The requested bootstrap was run in PowerShell. The Unix `find ... | sed ...`
command was represented with an equivalent `Get-ChildItem` command because the
active shell is PowerShell.

- `pwd`
- `git status --short`
- `git branch --show-current`
- `git log --oneline -5`
- `Get-ChildItem -File -Recurse -Depth 2 ... | Select-Object -First 200`

## Repository Structure Summary

Tracked project-sized areas, excluding `.git`, `.venv`, and `.pytest_cache`:

| Area | Files | Bytes | Role |
| --- | ---: | ---: | --- |
| root | 9 | 776725 | README, final report, final slides, package metadata, lockfile |
| `data` | 2 | 1106 | Raw-data ignore marker and download script |
| `docs` | 5 | 63476 | Existing design specs and implementation plans |
| `outputs` | 215 | 41302029 | Checked-in final result CSVs, figures, and checkpoints |
| `scripts` | 43 | 102889 | CLI entry points for experiments and diagnostics |
| `src` | 49 | 412409 | Main `selective_rag_rl` package |
| `tests` | 34 | 204508 | Pytest suite |

## Main Scripts Discovered

The script inventory includes:

- Main retrieval policies: `run_retrieval_policy_hotpot.py`,
  `run_retrieval_policy_nq.py`, `run_retrieval_policy_scifact.py`,
  `run_retrieval_policy_nfcorpus.py`
- Toy and baseline experiments: `run_hotpot_toy.py`, `run_nq_toy.py`,
  `run_dense_hotpot.py`, `run_bandit_baselines.py`
- Diagnostics: `run_ope_diagnostics.py`, `run_ope_stability.py`,
  `run_complexity_diagnostics.py`, `run_policy_diagnostics.py`,
  `run_statistical_diagnostics.py`
- Selection and semantic pilots: `run_policy_model_sweep.py`,
  `run_repeated_selection.py`, `run_semantic_depth_sweep.py`,
  `run_repeated_semantic_depth_sweep.py`, `run_embedding_preflight.py`
- Final artifact builders: `run_final_paper_assets.py`,
  `run_final_claims_matrix.py`, `run_artifact_index.py`,
  `run_checkpoint_manifest.py`, `run_markdown_consistency.py`

## Existing Test Suite Summary

There are 34 test files with 142 discovered test functions. The largest coverage
area is `tests/test_retrieval_policy_experiment.py` with 44 tests. Existing
tests cover loaders, BM25/dense retrieval, rewrite actions, bandit policies,
OPE, constrained policies, confidence gates, feature diagnostics, artifact
indexing, final-claims generation, and semantic-depth selection diagnostics.

## Existing Result Artifacts Summary

Key checked-in final artifacts already exist:

- `outputs/results/final_main_results_table.csv`
- `outputs/results/final_claims_matrix.csv`
- `outputs/results/final_artifact_index.csv`
- `outputs/results/final_checkpoint_manifest.csv`
- `outputs/results/final_markdown_consistency.csv`
- `outputs/figures/final_reward_delta_ci.png`
- `outputs/figures/final_cost_reward_frontier.png`
- `outputs/figures/final_ope_estimator_error.png`
- `outputs/figures/final_linucb_comparison.png`
- `outputs/checkpoints/scifact_retrieval_policy.pkl`
- `outputs/checkpoints/nfcorpus_retrieval_policy.pkl`

The current final-claim posture is conservative and retrieval-stage focused:
offline contextual-bandit retrieval routing improves cost-aware evidence
retrieval on SciFact and NFCorpus, supported by bootstrap, OPE, selected-action
baselines, constrained utility, and repeated-seed diagnostics.

## Risks, Missing Data, And Dependency Concerns

- `data/raw/` currently contains only `download.sh`; HotpotQA, Natural
  Questions, SciFact, and NFCorpus raw files are absent locally.
- Full-data experiment scripts should not be expected to run until raw data is
  downloaded into the documented layout.
- Vertex/Gemini paths exist but must not be called by default because they may
  require credentials and spend quota.
- Default tests and smoke reproduction should avoid model downloads and external
  APIs.
- `.venv` and `.pytest_cache` are not tracked, but `.gitignore` currently does
  not explicitly list `.venv/`, `.pytest_cache/`, `outputs/cache/`, or broad
  `data/raw/` protection.

## Proposed Priority Backlog

1. P0: Add a raw-data-free smoke reproduction runner and final reproduction
   quickstart.
2. P0: Harden `.gitignore` for local environments, raw data, and caches without
   removing existing checked-in final evidence.
3. P1: Add tested SQuAD-style EM/F1 answer metrics and a deterministic lexical
   reader evaluation path.
4. P3: Add RL framing documentation and targeted OPE/bandit sanity tests if
   coverage gaps remain after inspection.
5. P2: Add validation protocol documentation and guardrail tests around
   train-best fallback decisions.
6. P5/P6: Add cost-model documentation and final run summary after source
   changes are verified.
