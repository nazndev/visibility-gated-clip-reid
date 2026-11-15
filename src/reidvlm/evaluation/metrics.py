"""Market-1501 CMC and mAP.

Junk images (pid < 0) are removed from the ranked list. Same-identity,
same-camera gallery images are removed. Other people from the query camera
remain and count as negatives. Queries with no remaining positive are excluded
from the averages. This follows the protocol implemented by Zheng/torchreid and
by Syliz517/CLIP-ReID ``utils/metrics.py`` after junk has already been dropped
at load time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QueryMetrics:
    excluded: bool
    ap: float | None
    cmc_hits: list[int]
    top1_correct: bool | None
    valid_positive_count: int
    shortlist_hit: bool | None
    shortlist_positive_count: int


@dataclass
class EvalResult:
    map: float
    rank1: float
    rank5: float
    rank10: float
    valid_queries: int
    excluded_queries: int
    per_query: list[QueryMetrics]
    shortlist_hit_rate: float
    positive_coverage_at_k: float


def kept_indices(order: np.ndarray, q_pid: int, q_cam: int, g_pids: np.ndarray, g_cams: np.ndarray) -> np.ndarray:
    keep = []
    for index in order:
        pid = int(g_pids[index])
        cam = int(g_cams[index])
        if pid < 0:
            continue
        if pid == q_pid and cam == q_cam:
            continue
        keep.append(int(index))
    return np.asarray(keep, dtype=np.int64)


def average_precision(matches: np.ndarray) -> float:
    hits = np.flatnonzero(matches)
    if len(hits) == 0:
        return 0.0
    precisions = [(rank + 1) / (position + 1) for rank, position in enumerate(hits)]
    return float(np.mean(precisions))


def evaluate_rankings(
    orders: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    k: int = 5,
    max_rank: int = 10,
) -> EvalResult:
    """``orders`` is query-major, best gallery index first, before protocol filtering."""
    per_query: list[QueryMetrics] = []
    aps = []
    cmc_rows = []
    covered = 0
    total_positives = 0
    hits = 0
    for q_index in range(orders.shape[0]):
        q_pid = int(q_pids[q_index])
        q_cam = int(q_cams[q_index])
        kept = kept_indices(orders[q_index], q_pid, q_cam, g_pids, g_cams)
        if len(kept) == 0:
            per_query.append(QueryMetrics(True, None, [], None, 0, None, 0))
            continue
        matches = (g_pids[kept] == q_pid).astype(np.int32)
        positive_count = int(matches.sum())
        if positive_count == 0:
            per_query.append(QueryMetrics(True, None, [], None, 0, None, 0))
            continue
        ap = average_precision(matches)
        cmc = np.minimum(np.cumsum(matches), 1)
        if len(cmc) < max_rank:
            cmc = np.pad(cmc, (0, max_rank - len(cmc)), constant_values=int(cmc[-1]))
        row = cmc[:max_rank].astype(np.int32)
        # Raw top-K, not the protocol-filtered prefix. A same-camera image can occupy a slot.
        raw_top = orders[q_index][:k]
        raw_positive = 0
        for index in raw_top:
            pid = int(g_pids[index])
            cam = int(g_cams[index])
            if pid == q_pid and cam != q_cam and pid >= 0:
                raw_positive += 1
        shortlist_hit = raw_positive > 0
        aps.append(ap)
        cmc_rows.append(row)
        covered += raw_positive
        total_positives += positive_count
        hits += int(shortlist_hit)
        per_query.append(
            QueryMetrics(
                False,
                ap,
                row.tolist(),
                bool(row[0]),
                positive_count,
                shortlist_hit,
                raw_positive,
            )
        )
    if not aps:
        raise ValueError("No query has a valid gallery positive.")
    cmc_mean = np.mean(np.stack(cmc_rows), axis=0)
    return EvalResult(
        map=float(np.mean(aps)),
        rank1=float(cmc_mean[0]),
        rank5=float(cmc_mean[min(4, max_rank - 1)]),
        rank10=float(cmc_mean[min(9, max_rank - 1)]),
        valid_queries=len(aps),
        excluded_queries=len(per_query) - len(aps),
        per_query=per_query,
        shortlist_hit_rate=hits / len(aps),
        positive_coverage_at_k=covered / total_positives if total_positives else 0.0,
    )


def reference_eval_func(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 10,
) -> tuple[np.ndarray, float]:
    """Port of Syliz517/CLIP-ReID ``utils/metrics.py:eval_func`` for cross-checking.

    Smaller distance is better. Junk must already have been removed, because the
    original function treats every non-match as a negative.
    """
    indices = np.argsort(distmat, axis=1, kind="mergesort")
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_ap = []
    for q_index in range(distmat.shape[0]):
        order = indices[q_index]
        remove = (g_pids[order] == q_pids[q_index]) & (g_cams[order] == q_cams[q_index])
        orig = matches[q_index][np.invert(remove)]
        if not np.any(orig):
            continue
        cmc = np.cumsum(orig)
        cmc[cmc > 1] = 1
        if len(cmc) < max_rank:
            cmc = np.pad(cmc, (0, max_rank - len(cmc)), constant_values=cmc[-1])
        all_cmc.append(cmc[:max_rank].astype(np.float64))
        num_rel = orig.sum()
        tmp = np.cumsum(orig) / np.arange(1, len(orig) + 1)
        all_ap.append(float(np.sum(tmp * orig) / num_rel))
    if not all_cmc:
        raise ValueError("No valid query.")
    return np.mean(all_cmc, axis=0), float(np.mean(all_ap))


def top1_changes(baseline: EvalResult, revised: EvalResult) -> dict[str, int]:
    repairs = damage = 0
    for before, after in zip(baseline.per_query, revised.per_query):
        if before.excluded or after.excluded:
            continue
        if before.top1_correct is False and after.top1_correct is True:
            repairs += 1
        elif before.top1_correct is True and after.top1_correct is False:
            damage += 1
    return {"top1_repairs": repairs, "top1_damage": damage}
