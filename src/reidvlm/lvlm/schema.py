"""Structured LVLM pair judgment. Visibility flags are model-reported, not annotated."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "pair-judgment-v1"
CUES = (
    "upper_clothing",
    "lower_clothing",
    "footwear",
    "carried_object",
    "distinctive_mark",
)
Relation = Literal["match", "mismatch", "unknown"]
Verdict = Literal["same", "different", "uncertain"]


class CueJudgment(BaseModel):
    visible_in_a: bool
    visible_in_b: bool
    relation: Relation
    observation_a: str = Field(max_length=240)
    observation_b: str = Field(max_length=240)


class PairJudgment(BaseModel):
    upper_clothing: CueJudgment
    lower_clothing: CueJudgment
    footwear: CueJudgment
    carried_object: CueJudgment
    distinctive_mark: CueJudgment
    overall_verdict: Verdict

    def cue_map(self) -> dict[str, CueJudgment]:
        return {name: getattr(self, name) for name in CUES}
