"""Iteration-2 plan, training-derived manifest, and resumable embedding.

The released CLIP-ReID encoder was trained on Market-1501 training identities.
A query/gallery split of those images is not an identity-unseen validation set.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from reidvlm.data.market1501 import Sample
from reidvlm.evaluation.metrics import evaluate_rankings
from reidvlm.persistence import RunStore
from reidvlm.pruning import (
    GRID_BLOCKS,
    GRID_KEEP,
    PATCHES,
    RANDOM_SEEDS,
    TIE_LATENCY_MS,
    TIE_MAP,
    TIE_RANK1,
    VALIDATION_SEED,
)
from reidvlm.pruning.selection import SelectionConfig
from reidvlm.retrieval.cosine import rank_indices


def experiment_plan() -> dict:
    return {
        "iteration": 2,
        "question": "How do inference-time patch selection, pruning depth, and retained-token count affect retrieval accuracy and measured inference cost for a frozen CLIP-ReID encoder?",
        "historical_anchor": {
            "label": "previously reported full Market-1501 results; not a claim that new code reproduces them on another machine",
            "full_cosine_mAP": 89.41,
            "full_cosine_rank1": 94.98,
            "full_kreciprocal_mAP": 94.48,
            "full_kreciprocal_rank1": 95.37,
            "attention_block6_keep64_cosine_mAP": 74.86,
            "attention_block6_keep64_cosine_rank1": 88.93,
            "attention_block6_keep64_kreciprocal_mAP": 88.40,
            "attention_block6_keep64_kreciprocal_rank1": 90.62,
            "files": [
                "results/market1501_test_20261115T182047Z.json",
                "results/market1501_test_20261115T183006Z.json",
            ],
        },
        "configurations": [
            {"strategy": "none", "prune_block": None, "keep_patch_count": PATCHES, "role": "reference"},
            *[
                {"strategy": "cls_attention", "prune_block": block, "keep_patch_count": keep, "role": "grid"}
                for block in GRID_BLOCKS
                for keep in GRID_KEEP
            ],
        ],
        "random_seeds": list(RANDOM_SEEDS),
        "attention_backends": ["reference", "cls_row"],
        "default_attention_backend": "reference",
        "validation": {
            "source": "Market-1501 bounding_box_train only",
            "seed": VALIDATION_SEED,
            "rule": "one query image per identity that has at least two cameras; remaining training images are the gallery",
            "encoder_saw_these_identities": True,
            "not_an_unseen_identity_split": True,
        },
        "selection_rule": {
            "candidates": "the four cls_attention grid cells, not the random controls",
            "order": "highest raw validation mAP, then Rank-1, then lower measured median encoder latency",
            "tie_mAP": TIE_MAP,
            "tie_rank1": TIE_RANK1,
            "tie_latency_ms": TIE_LATENCY_MS,
            "ceiling_fallback": "if every candidate has validation mAP at the ceiling, use block 9 and 96 patches and label the choice predefined",
            "official_test_not_used": True,
        },
        "timing": {
            "batch_sizes": [1, 32],
            "warmup": 10,
            "iterations": 50,
            "trials": 3,
            "precision": "float32",
            "scopes": ["encoder_inference", "load_preprocess_encode_search"],
        },
        "comparisons": [
            "cls_attention versus each matched random seed",
            "cls_attention versus the mean of the five random seeds",
            "every compressed setting versus the full encoder",
            "raw cosine is primary; k-reciprocal is secondary and compared only with k-reciprocal",
        ],
        "official_test_priority": [
            "full encoder",
            "block 6, 64 patches, cls_attention",
            "block 6, 64 patches, five random seeds",
            "the additional attention configuration chosen on validation, unless it is the same cell",
            "that configuration's five random seeds",
        ],
    }


def write_plan(path: Path) -> dict:
    plan = experiment_plan()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2))
    return plan


def build_validation_manifest(train: list[Sample], seed: int = VALIDATION_SEED) -> dict:
    """Training images only. Query and gallery paths do not overlap."""
    groups: dict[int, list[Sample]] = defaultdict(list)
    for sample in train:
        if sample.pid > 0 and not sample.junk:
            groups[sample.pid].append(sample)
    generator = np.random.Generator(np.random.PCG64(seed))
    queries: list[Sample] = []
    for pid in sorted(groups):
        cameras = {sample.camid for sample in groups[pid]}
        if len(cameras) < 2:
            continue
        pick = int(generator.integers(0, len(groups[pid])))
        queries.append(groups[pid][pick])
    query_paths = {sample.path for sample in queries}
    gallery = [sample for sample in train if sample.path not in query_paths and not sample.junk]
    if query_paths & {sample.path for sample in gallery}:
        raise RuntimeError("Validation query and gallery overlap.")
    valid = 0
    for query in queries:
        positives = [
            sample
            for sample in gallery
            if sample.pid == query.pid and sample.camid != query.camid and sample.pid > 0
        ]
        valid += int(bool(positives))
    return {
        "seed": seed,
        "encoder_saw_these_identities": True,
        "not_an_unseen_identity_split": True,
        "source": "bounding_box_train",
        "eligible_identities": len(queries),
        "query_images": len(queries),
        "gallery_images": len(gallery),
        "train_images_used": len(queries) + len(gallery),
        "query_cameras": sorted({sample.camid for sample in queries}),
        "gallery_cameras": sorted({sample.camid for sample in gallery}),
        "valid_queries_with_cross_camera_positive": valid,
        "queries": [{"path": sample.path, "pid": sample.pid, "camid": sample.camid} for sample in queries],
        "gallery": [{"path": sample.path, "pid": sample.pid, "camid": sample.camid} for sample in gallery],
    }


def manifest_counts(manifest: dict) -> dict:
    hidden = {"queries", "gallery"}
    return {key: value for key, value in manifest.items() if key not in hidden}


def choose_configuration(rows: list[dict], latency_ms: dict[str, float] | None = None) -> dict:
    """rows need name, mAP, rank1. latency_ms maps name to median milliseconds."""
    if not rows:
        raise ValueError("No validation rows to choose from.")
    if all(row["mAP"] >= 1.0 - TIE_MAP for row in rows):
        return {
            "choice": "block9_keep96",
            "reason": "validation mAP is at the ceiling, so the additional test configuration is the predefined block-9/96-patch setting",
            "predefined": True,
        }
    latency_ms = latency_ms or {}

    def sort_key(row: dict) -> tuple:
        late = latency_ms.get(row["name"], 0.0)
        return (-round(row["mAP"] / TIE_MAP), -round(row["rank1"] / TIE_RANK1), round(late / TIE_LATENCY_MS))

    ordered = sorted(rows, key=sort_key)
    winner = ordered[0]
    return {"choice": winner["name"], "reason": "highest validation mAP, then Rank-1, then lower median encoder latency", "predefined": False, "row": winner}


def paired_query_changes(before_ap: list[float], after_ap: list[float], tolerance: float = 1e-8) -> dict:
    improved = worsened = unchanged = 0
    for left, right in zip(before_ap, after_ap):
        if right > left + tolerance:
            improved += 1
        elif left > right + tolerance:
            worsened += 1
        else:
            unchanged += 1
    return {"improved": improved, "worsened": worsened, "unchanged": unchanged, "queries": len(before_ap)}


def selection_for(strategy: str, block: int, keep: int, seed: int | None = None, backend: str = "reference") -> SelectionConfig:
    config = SelectionConfig(
        strategy=strategy,
        prune_block=block,
        keep_patch_count=keep if strategy != "none" else PATCHES,
        seed=seed,
        attention_backend=backend,
        patch_count=PATCHES,
    )
    config.validate()
    return config


def score_split(query_features: np.ndarray, gallery_features: np.ndarray, queries: list[dict], gallery: list[dict]) -> dict:
    similarities = query_features @ gallery_features.T
    orders = rank_indices(similarities)
    q_pids = np.array([item["pid"] for item in queries])
    g_pids = np.array([item["pid"] for item in gallery])
    q_cams = np.array([item["camid"] for item in queries])
    g_cams = np.array([item["camid"] for item in gallery])
    result = evaluate_rankings(orders, q_pids, g_pids, q_cams, g_cams, k=5)
    return {
        "mAP": result.map,
        "rank1": result.rank1,
        "rank5": result.rank5,
        "rank10": result.rank10,
        "valid_queries": result.valid_queries,
        "excluded_queries": result.excluded_queries,
        "raw_top5_hit_rate": result.shortlist_hit_rate,
        "raw_top5_positive_coverage": result.positive_coverage_at_k,
        "raw_top5_note": "Computed on the unfiltered top 5. Same-person same-camera images can occupy a slot. This is not CMC Rank-5.",
        "per_query_ap": [item.ap for item in result.per_query if item.ap is not None],
        "per_query_first_positive_rank": [
            None if item.excluded else item.first_positive_rank for item in result.per_query
        ],
    }


def resume_embed(store: RunStore, identity: dict, ids: list[str], embed_missing, chunk_size: int = 32) -> np.ndarray:
    """embed_missing(list[str]) -> np.ndarray aligned with that list.

    Completed chunks are saved before the next chunk starts. A later call with the
    same identity skips identifiers that already verify.
    """
    table = store.load_feature_table(identity) or {}
    unknown = [item for item in table if item not in ids]
    if unknown:
        raise ValueError("Feature cache contains identifiers outside this manifest.")
    missing = [item for item in ids if item not in table]
    for start in range(0, len(missing), chunk_size):
        batch = missing[start : start + chunk_size]
        fresh = np.asarray(embed_missing(batch), dtype=np.float32)
        if len(fresh) != len(batch):
            raise RuntimeError("Embedding resumed the wrong number of rows.")
        for item, row in zip(batch, fresh):
            table[item] = row
        store.save_feature_table(identity, table)
    reloaded = store.load_feature_table(identity)
    if reloaded is None:
        raise RuntimeError("Saved feature table failed verification.")
    return np.stack([reloaded[item] for item in ids]).astype(np.float32)


def resume_embed_chunks(store: RunStore, identity: dict, ids: list[str], embed_missing, chunk_size: int = 32) -> np.ndarray:
    """Save each new batch as its own verified chunk. Earlier chunks are not rewritten."""
    table = store.load_feature_chunks(identity)
    unknown = [item for item in table if item not in ids]
    if unknown:
        raise ValueError("Feature cache contains identifiers outside this manifest.")
    missing = [item for item in ids if item not in table]
    for start in range(0, len(missing), chunk_size):
        batch = missing[start : start + chunk_size]
        fresh = np.asarray(embed_missing(batch), dtype=np.float32)
        if len(fresh) != len(batch):
            raise RuntimeError("Embedding resumed the wrong number of rows.")
        store.save_feature_chunk(identity, batch, fresh)
    reloaded = store.load_feature_chunks(identity)
    if any(item not in reloaded for item in ids):
        raise RuntimeError("A resumed chunk did not verify.")
    return np.stack([reloaded[item] for item in ids]).astype(np.float32)
