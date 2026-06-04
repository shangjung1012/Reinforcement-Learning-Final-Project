"""Selective query rewriting and retrieval-routing experiments for RAG.

The implementation is organized by responsibility:

- ``core``: shared data structures, retrieval primitives, metrics, rewrites.
- ``policies``: contextual-bandit, fitted-Q, heuristic, confidence, and OPE code.
- ``experiments``: runnable experiment orchestration and policy sweeps.
- ``diagnostics``: analysis exporters and post-hoc result diagnostics.
- ``preflight``: local data/API/cache availability checks.
- ``integrations``: external service adapters.
- ``reports``: final-report artifact and evidence exporters.

Only a very small set of root-level compatibility aliases is kept for saved
checkpoint pickle paths. New code should import from the grouped packages.
"""

__all__ = [
    "core",
    "diagnostics",
    "experiments",
    "integrations",
    "policies",
    "preflight",
    "reports",
]
