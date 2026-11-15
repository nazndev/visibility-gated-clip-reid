"""Bounded JSON parsing. Invalid output becomes a recorded fallback, not invented evidence."""

from __future__ import annotations

import json

from pydantic import ValidationError

from reidvlm.lvlm.schema import PairJudgment


def extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object in the model output.")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON value is not an object.")
    return _normalise(payload)


def _normalise(payload: dict) -> dict:
    verdict = payload.get("overall_verdict")
    if isinstance(verdict, str):
        payload["overall_verdict"] = verdict.strip().lower()
    for key, value in list(payload.items()):
        if not isinstance(value, dict):
            continue
        relation = value.get("relation")
        if isinstance(relation, str):
            value["relation"] = relation.strip().lower()
        for flag in ("visible_in_a", "visible_in_b"):
            if isinstance(value.get(flag), str):
                lowered = value[flag].strip().lower()
                if lowered in {"true", "yes"}:
                    value[flag] = True
                elif lowered in {"false", "no", "unknown", "unclear", "n/a", "none", "not visible"}:
                    # The prompt reserves "unknown" for relation. A model that writes it
                    # in a visibility field is reporting that the cue was not observed.
                    value[flag] = False
    return payload


def parse_judgment(text: str) -> PairJudgment:
    try:
        return PairJudgment.model_validate(extract_json_object(text))
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(str(exc)) from exc
