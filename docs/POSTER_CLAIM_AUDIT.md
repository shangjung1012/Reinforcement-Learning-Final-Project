# Poster Claim Audit

The current poster is aligned with the conservative project claim. It describes
the work as retrieval-stage contextual-bandit routing for RAG, and explicitly
states that the evidence is not generated-answer EM/F1 and not LLM fine-tuning.

## Claim Check

- Main claim: cost-aware evidence retrieval on full-corpus SciFact and NFCorpus.
- Supported by: `outputs/results/final_main_results_table.csv` and
  `outputs/results/final_claims_matrix.csv`.
- Gemini actions: described as expensive extensions, not the final deployed
  improvement.
- Semantic features: described as analysis-only.
- Smoke tests: described as code-path checks only.

No research-claim edit is required for the current code-quality, warning, and
evidence-dashboard fixes.

## Manual Review Needed

The author line in `poster/poster.tex` appears to contain mojibake for Chinese
names. That is a presentation-quality issue, not a claim-validity issue. Fix it
before final submission if the correct author names are available.
