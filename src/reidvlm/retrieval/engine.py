"""Shared search engine used by the CLI and the Streamlit application."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from reidvlm.data.images import image_sha256, load_rgb
from reidvlm.lvlm.qwen import JudgmentRecord, LVLMUnavailable
from reidvlm.lvlm.schema import PairJudgment
from reidvlm.retrieval.cosine import rank_indices
from reidvlm.retrieval.gate import VARIANTS, CueView, GateDecision, JudgmentView, decision_for
from reidvlm.retrieval.rerank import corrected_scores, rerank_shortlist


@dataclass
class GalleryItem:
    anonymous_id: str
    image: Image.Image | None
    source_id: str
    bbox: tuple[int, int, int, int]
    feature: np.ndarray | None = None
    path: str | None = None

    def load(self) -> Image.Image:
        if self.image is None:
            if not self.path:
                raise ValueError("Gallery item has no image.")
            self.image = load_rgb(self.path)
        return self.image


@dataclass
class Hit:
    anonymous_id: str
    source_id: str
    bbox: tuple[int, int, int, int]
    image: Image.Image
    original_rank: int
    revised_rank: int
    visual_similarity: float
    corrected_score: float
    evidence: str
    gate_accepted: bool | None
    gate_reason: str


@dataclass
class SearchResult:
    hits: list[Hit]
    mode: str
    k: int
    seconds: float
    cached_judgments: int
    fresh_judgments: int
    lvlm_calls: int
    invalid_judgments: int
    gate_accepted_pairs: int
    judged_pairs: int
    lvlm_used: bool
    note: str = ""


def judgment_view(record: JudgmentRecord) -> JudgmentView:
    if not record.valid or record.parsed is None:
        return JudgmentView(False, None, {})
    cues = {
        name: CueView(cue.visible_in_a, cue.visible_in_b, cue.relation)
        for name, cue in record.parsed.cue_map().items()
    }
    return JudgmentView(True, record.parsed.overall_verdict, cues)


def evidence_text(record: JudgmentRecord) -> str:
    if not record.valid or record.parsed is None:
        return "No valid LVLM observation. The pair kept its visual score."
    parts = []
    for name, cue in record.parsed.cue_map().items():
        if cue.relation == "unknown":
            continue
        parts.append(f"{name}: {cue.relation} ({cue.observation_a} / {cue.observation_b})")
    verdict = record.parsed.overall_verdict
    summary = "; ".join(parts[:3]) if parts else "No stable cue observation."
    return f"Verdict {verdict}. {summary}"


def record_to_dict(record: JudgmentRecord) -> dict:
    return {
        "valid": record.valid,
        "parsed": None if record.parsed is None else record.parsed.model_dump(),
        "attempts": [attempt.__dict__ for attempt in record.attempts],
    }


def record_from_dict(payload: dict) -> JudgmentRecord:
    from reidvlm.lvlm.qwen import Attempt

    parsed = None if payload.get("parsed") is None else PairJudgment.model_validate(payload["parsed"])
    attempts = [Attempt(**item) for item in payload.get("attempts", [])]
    return JudgmentRecord(bool(payload.get("valid")), parsed, attempts, cached=True)


class SearchEngine:
    """Image and feature search. It does not accept identity labels or filenames."""

    def __init__(
        self,
        items: list[GalleryItem],
        encoder=None,
        lvlm=None,
        cache=None,
        k: int = 5,
        alpha: float = 0.05,
        min_shared_cues: int = 2,
    ):
        if not items:
            raise ValueError("Gallery is empty.")
        self.items = list(items)
        self.encoder = encoder
        self.lvlm = lvlm
        self.cache = cache
        self.k = k
        self.alpha = alpha
        self.min_shared_cues = min_shared_cues
        if any(item.feature is None for item in self.items):
            if encoder is None:
                raise ValueError("Gallery features are missing and no encoder was provided.")
            features = encoder.embed_images([item.load() for item in self.items])
            for item, feature in zip(self.items, features):
                item.feature = feature
        self.features = np.stack([item.feature for item in self.items]).astype(np.float32)

    def _judge(self, image_a: Image.Image, image_b: Image.Image, order: str) -> tuple[JudgmentRecord, bool]:
        if self.lvlm is None:
            raise LVLMUnavailable("LVLM re-ranking is unavailable. Visual retrieval can still be used.")
        hash_a = image_sha256(image_a)
        hash_b = image_sha256(image_b)
        if self.cache is not None:
            cached = self.cache.get(hash_a, hash_b, order)
            if cached is not None:
                return record_from_dict(cached), True
        record = self.lvlm.judge(image_a, image_b)
        if self.cache is not None:
            self.cache.put(hash_a, hash_b, order, record_to_dict(record))
        return record, False

    def search(self, query_image: Image.Image, mode: str = "visual") -> SearchResult:
        if mode not in VARIANTS:
            raise ValueError(f"Unknown search mode: {mode}")
        if self.encoder is None:
            raise RuntimeError("No encoder is loaded.")
        started = time.perf_counter()
        query_feature = self.encoder.embed_images([query_image])[0]
        return self.search_from_feature(query_feature, query_image, mode, started)

    def search_from_feature(
        self,
        query_feature: np.ndarray,
        query_image: Image.Image,
        mode: str,
        started: float | None = None,
    ) -> SearchResult:
        started = time.perf_counter() if started is None else started
        similarities = self.features @ np.asarray(query_feature, dtype=np.float32)
        original = rank_indices(similarities)[0]
        deltas = np.zeros(len(self.items), dtype=np.float64)
        decisions: dict[int, GateDecision] = {}
        records: dict[int, JudgmentRecord] = {}
        cached = fresh = calls = invalid = accepted = 0
        lvlm_used = mode != "visual"
        if lvlm_used:
            for gallery_index in original[: self.k]:
                candidate = self.items[int(gallery_index)].load()
                forward, was_cached = self._judge(query_image, candidate, "ab")
                backward, back_cached = self._judge(candidate, query_image, "ba")
                cached += int(was_cached) + int(back_cached)
                fresh += int(not was_cached) + int(not back_cached)
                calls += forward.attempt_count + backward.attempt_count
                if not forward.valid or not backward.valid:
                    invalid += 1
                decision = decision_for(
                    judgment_view(forward),
                    judgment_view(backward),
                    self.alpha,
                    self.min_shared_cues,
                    mode,
                )
                deltas[int(gallery_index)] = decision.delta
                decisions[int(gallery_index)] = decision
                records[int(gallery_index)] = forward
                accepted += int(decision.accepted)
        revised = rerank_shortlist(similarities, deltas, self.k if lvlm_used else 0)
        original_rank = {int(index): rank + 1 for rank, index in enumerate(original)}
        scores = corrected_scores(similarities, deltas)
        hits = []
        for rank, index in enumerate(revised[:10], start=1):
            item = self.items[int(index)]
            decision = decisions.get(int(index))
            record = records.get(int(index))
            hits.append(
                Hit(
                    item.anonymous_id,
                    item.source_id,
                    item.bbox,
                    item.load(),
                    original_rank[int(index)],
                    rank,
                    float(similarities[int(index)]),
                    float(scores[int(index)]),
                    evidence_text(record) if record is not None else "Visual similarity only.",
                    None if decision is None else decision.accepted,
                    "not judged" if decision is None else decision.reason,
                )
            )
        return SearchResult(
            hits=hits,
            mode=mode,
            k=self.k,
            seconds=time.perf_counter() - started,
            cached_judgments=cached,
            fresh_judgments=fresh,
            lvlm_calls=calls,
            invalid_judgments=invalid,
            gate_accepted_pairs=accepted,
            judged_pairs=min(self.k, len(self.items)) if lvlm_used else 0,
            lvlm_used=lvlm_used,
            note="" if lvlm_used else "Visual retrieval only. No LVLM re-ranking was applied.",
        )
