"""Shortlist re-ranking. The tail outside the original top K is not moved."""

from __future__ import annotations

import numpy as np

from reidvlm.retrieval.cosine import rank_indices


def rerank_shortlist(similarities: np.ndarray, deltas: np.ndarray, k: int) -> np.ndarray:
    original = rank_indices(similarities)[0]
    if k <= 0:
        return original
    k = min(k, len(original))
    if np.all(np.asarray(deltas) == 0):
        return original
    original_rank = {int(index): rank for rank, index in enumerate(original)}
    head = list(original[:k])
    tail = list(original[k:])
    head.sort(key=lambda index: (-(float(similarities[index]) + float(deltas[index])), original_rank[index]))
    return np.asarray(head + tail, dtype=np.int64)


def corrected_scores(similarities: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    return np.asarray(similarities, dtype=np.float64) + np.asarray(deltas, dtype=np.float64)
