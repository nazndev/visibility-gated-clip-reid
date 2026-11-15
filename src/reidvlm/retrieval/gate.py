"""Visibility and consistency gate.

The corrected score is a retrieval score, not a probability.

    d = (d_AB + d_BA) / 2
    s = v + alpha * r * d          proposed, visibility-only, consistency-only
    s = v + alpha * d              ungated

A cue is stable only when both images show it in both orders, the relation is
match or mismatch, and the relation is unchanged by swapping the images.
"""

from __future__ import annotations

from dataclasses import dataclass

# Unchanged scoring rule. Prompt edits do not revise this definition.
GATE_VERSION = "visibility-consistency-v1"

CUES = (
    "upper_clothing",
    "lower_clothing",
    "footwear",
    "carried_object",
    "distinctive_mark",
)
VERDICT_TO_SCORE = {"same": 1.0, "different": -1.0, "uncertain": 0.0}
VARIANTS = ("visual", "ungated", "proposed", "visibility", "consistency")


@dataclass(frozen=True)
class CueView:
    visible_in_a: bool
    visible_in_b: bool
    relation: str


@dataclass(frozen=True)
class JudgmentView:
    valid: bool
    verdict: str | None
    cues: dict[str, CueView]


@dataclass(frozen=True)
class GateDecision:
    delta: float
    r: float
    d: float
    stable_cues: int
    accepted: bool
    reason: str


def stable_cue_count(first: JudgmentView, second: JudgmentView) -> int:
    if not first.valid or not second.valid:
        return 0
    count = 0
    for name in CUES:
        left = first.cues[name]
        right = second.cues[name]
        both_orders_visible = left.visible_in_a and left.visible_in_b and right.visible_in_a and right.visible_in_b
        agreed = left.relation == right.relation and left.relation in {"match", "mismatch"}
        if both_orders_visible and agreed:
            count += 1
    return count


def decision_for(
    first: JudgmentView,
    second: JudgmentView,
    alpha: float,
    min_shared_cues: int,
    variant: str,
) -> GateDecision:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if variant == "visual":
        return GateDecision(0.0, 0.0, 0.0, 0, False, "visual retrieval has no LVLM correction")
    if not first.valid or not second.valid or first.verdict is None or second.verdict is None:
        return GateDecision(0.0, 0.0, 0.0, 0, False, "invalid response uses zero correction")
    d_value = (VERDICT_TO_SCORE[first.verdict] + VERDICT_TO_SCORE[second.verdict]) / 2.0
    stable = stable_cue_count(first, second)
    agree = first.verdict == second.verdict
    definite = agree and first.verdict != "uncertain"
    if variant == "ungated":
        return GateDecision(alpha * d_value, 0.0, d_value, stable, False, "ungated correction has no visibility gate")
    if variant == "proposed":
        passed = definite and stable >= min_shared_cues
        ratio = (stable / 5.0) if passed else 0.0
        reason = "proposed gate accepted" if passed else "proposed gate abstained"
    elif variant == "visibility":
        passed = stable >= min_shared_cues
        ratio = (stable / 5.0) if passed else 0.0
        reason = "visibility-only gate"
    elif variant == "consistency":
        passed = definite
        ratio = 1.0 if passed else 0.0
        reason = "consistency-only gate"
    else:
        raise AssertionError(variant)
    return GateDecision(alpha * ratio * d_value, ratio, d_value, stable, ratio > 0.0, reason)
