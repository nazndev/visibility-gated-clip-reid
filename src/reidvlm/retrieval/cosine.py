"""Exact cosine retrieval on L2-normalized features."""

from __future__ import annotations

import numpy as np


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def cosine_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return l2_normalize(query) @ l2_normalize(gallery).T


def rank_indices(similarities: np.ndarray) -> np.ndarray:
    """Best first. Ties break toward the smaller gallery index."""
    if similarities.ndim == 1:
        similarities = similarities[np.newaxis, :]
    gallery_index = np.broadcast_to(np.arange(similarities.shape[1]), similarities.shape)
    # lexsort uses the last key as the primary key.
    return np.lexsort((gallery_index, -similarities), axis=1).astype(np.int64)
