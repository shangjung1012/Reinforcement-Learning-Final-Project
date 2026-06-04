from __future__ import annotations

from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class ConfidenceGatedDecision:
    action: str
    policy_action: str
    fallback_action: str
    fallback_used: bool
    predicted_margin: float
    policy_score: float
    runner_up_score: float


def prediction_margin(scores: dict[str, float], actions: list[str]) -> tuple[str, float]:
    """Return the top action and top-vs-second predicted score margin."""
    if not actions:
        raise ValueError("actions must not be empty")
    missing = [action for action in actions if action not in scores]
    if missing:
        raise KeyError(f"Missing scores for actions: {missing}")
    ordered = sorted(
        actions,
        key=lambda action: (scores[action], -actions.index(action)),
        reverse=True,
    )
    if len(ordered) == 1:
        return ordered[0], inf
    return ordered[0], float(scores[ordered[0]] - scores[ordered[1]])


def confidence_gated_action(
    scores: dict[str, float],
    actions: list[str],
    fallback_action: str,
    min_margin: float,
) -> ConfidenceGatedDecision:
    if fallback_action not in actions:
        raise ValueError(f"fallback_action must be in actions: {fallback_action}")
    policy_action, margin = prediction_margin(scores, actions)
    ordered = sorted(
        actions,
        key=lambda action: (scores[action], -actions.index(action)),
        reverse=True,
    )
    runner_up = ordered[1] if len(ordered) > 1 else ordered[0]
    fallback_used = margin < min_margin
    return ConfidenceGatedDecision(
        action=fallback_action if fallback_used else policy_action,
        policy_action=policy_action,
        fallback_action=fallback_action,
        fallback_used=fallback_used,
        predicted_margin=margin,
        policy_score=float(scores[policy_action]),
        runner_up_score=float(scores[runner_up]),
    )
