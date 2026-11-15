"""k-reciprocal re-ranking.

Adapted from Zhong et al., CVPR 2017, and the Python implementation in
Syliz517/CLIP-ReID ``utils/reranking.py`` (commit eb1898b). The distance uses
query-query and gallery-gallery pairs. It is not a top-K-only reorder.

Paper defaults are k1=20, k2=6, lambda=0.3. The CLIP-ReID repository instead
calls k1=50, k2=15; that difference is exposed in the configuration and is not
applied silently.
"""

from __future__ import annotations

import numpy as np

from reidvlm.retrieval.cosine import l2_normalize


def _squared_euclidean(features: np.ndarray) -> np.ndarray:
    square = np.sum(features * features, axis=1, keepdims=True)
    distance = square + square.T - 2.0 * (features @ features.T)
    return np.maximum(distance, 0.0).astype(np.float32)


def k_reciprocal_rerank(
    query: np.ndarray,
    gallery: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    query = l2_normalize(np.asarray(query, dtype=np.float32))
    gallery = l2_normalize(np.asarray(gallery, dtype=np.float32))
    query_num = query.shape[0]
    features = np.concatenate([query, gallery], axis=0)
    all_num = features.shape[0]
    original = _squared_euclidean(features)
    column_max = np.max(original, axis=0)
    column_max[column_max == 0] = 1.0
    original = np.transpose(original / column_max)
    k1 = max(1, min(k1, all_num - 1))
    k2 = max(1, min(k2, all_num))
    initial_rank = np.argsort(original, axis=1).astype(np.int32)
    V = np.zeros_like(original)
    for i in range(all_num):
        forward = initial_rank[i, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal = forward[np.where(backward == i)[0]]
        expanded = reciprocal
        half = int(np.around(k1 / 2.0))
        for candidate in reciprocal:
            cand_forward = initial_rank[candidate, : half + 1]
            cand_backward = initial_rank[cand_forward, : half + 1]
            cand_reciprocal = cand_forward[np.where(cand_backward == candidate)[0]]
            overlap = np.intersect1d(cand_reciprocal, reciprocal)
            if len(cand_reciprocal) > 0 and len(overlap) > (2.0 / 3.0) * len(cand_reciprocal):
                expanded = np.append(expanded, cand_reciprocal)
        expanded = np.unique(expanded)
        weight = np.exp(-original[i, expanded])
        V[i, expanded] = weight / np.sum(weight)
    if k2 != 1:
        expanded_v = np.zeros_like(V)
        for i in range(all_num):
            expanded_v[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = expanded_v
    inverse = [np.where(V[:, column] != 0)[0] for column in range(all_num)]
    jaccard = np.zeros((query_num, all_num), dtype=np.float32)
    for i in range(query_num):
        nonzero = np.where(V[i] != 0)[0]
        minimum = np.zeros(all_num, dtype=np.float32)
        for index in nonzero:
            images = inverse[index]
            minimum[images] += np.minimum(V[i, index], V[images, index])
        jaccard[i] = 1.0 - minimum / (2.0 - minimum)
    fused = jaccard * (1.0 - lambda_value) + original[:query_num] * lambda_value
    return fused[:, query_num:]
