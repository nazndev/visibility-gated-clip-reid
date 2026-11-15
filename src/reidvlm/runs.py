"""Resumable feature and judgment runs. They call the existing experiment code."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reidvlm.config import ExperimentConfig
from reidvlm.data.images import image_sha256, load_rgb
from reidvlm.experiments import (
    ALPHAS,
    MIN_CUES,
    _apply_variant,
    _encoder,
    _ids,
    _metric_dict,
    judge_shortlist,
    load_split,
)
from reidvlm.evaluation.metrics import evaluate_rankings
from reidvlm.lvlm.prompts import PROMPT_VERSION
from reidvlm.lvlm.qwen import QwenVLBackend
from reidvlm.lvlm.schema import SCHEMA_VERSION
from reidvlm.persistence import RunStore
from reidvlm.retrieval.cosine import rank_indices
from reidvlm.retrieval.gate import GATE_VERSION
from reidvlm.runtime import package_tree_hash, resolve_path, set_seed, sha256_json, source_revision


def observation_identity(cfg: ExperimentConfig, checkpoint_sha256: str) -> dict:
    return {
        "dataset": cfg.dataset.name,
        "dataset_version": cfg.dataset.version,
        "split": cfg.split,
        "seed": cfg.seed,
        "checkpoint_sha256": checkpoint_sha256,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "gate_version": GATE_VERSION,
        "k": cfg.retrieval.k,
        "lvlm_model_id": cfg.lvlm.model_id,
        "lvlm_dtype": cfg.lvlm.dtype,
        "max_new_tokens": cfg.lvlm.max_new_tokens,
        "do_sample": cfg.lvlm.do_sample,
        "source_revision": source_revision(),
        "package_hash": package_tree_hash(),
    }


def _store_features(store: RunStore, samples, features: np.ndarray, role: str, checkpoint_sha256: str) -> int:
    features = np.asarray(features, dtype=np.float32)
    if len(samples) != len(features):
        raise RuntimeError(f"{role} feature rows do not match the image list")
    wrote = 0
    chunk = 512
    for start in range(0, len(samples), chunk):
        batch = samples[start : start + chunk]
        block = features[start : start + chunk]
        hashes = [image_sha256(load_rgb(sample.path)) for sample in batch]
        digest = sha256_json({"role": role, "start": start, "hashes": hashes, "encoder": checkpoint_sha256})
        npy_rel = f"features/{role}_{digest[:16]}.npy"
        meta_rel = f"features/{role}_{digest[:16]}.json"
        meta = {
            "role": role,
            "start": start,
            "count": int(len(batch)),
            "shape": [int(block.shape[0]), int(block.shape[1])],
            "dtype": "float32",
            "image_sha256": hashes,
            "encoder_sha256": checkpoint_sha256,
        }
        if not (store.is_verified(npy_rel) and store.is_verified(meta_rel)):
            store.publish_bytes(npy_rel, block.tobytes())
            store.publish_json(meta_rel, meta)
            wrote += 1
    return wrote


def load_feature_rows(store: RunStore, role: str, hashes: list[str], checkpoint_sha256: str) -> np.ndarray | None:
    found: dict[str, np.ndarray] = {}
    feature_dir = store.run_dir / "features"
    if not feature_dir.is_dir():
        return None
    for path in feature_dir.glob(f"{role}_*.json"):
        relative = path.relative_to(store.run_dir).as_posix()
        meta = store.read_verified_json(relative)
        npy_rel = relative[:-5] + ".npy"
        if meta is None or not store.is_verified(npy_rel):
            continue
        if meta.get("encoder_sha256") != checkpoint_sha256 or meta.get("dtype") != "float32":
            continue
        raw = (store.run_dir / npy_rel).read_bytes()
        array = np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"])
        for digest, row in zip(meta["image_sha256"], array):
            found[digest] = np.array(row, copy=True)
    if any(digest not in found for digest in hashes):
        return None
    return np.stack([found[digest] for digest in hashes])


def import_existing_features(cfg: ExperimentConfig, store: RunStore) -> dict:
    """Verify saved arrays against fresh embeddings of three rows, then copy them in chunks."""
    from reidvlm.experiments import load_split

    set_seed(cfg.seed)
    queries, gallery, _report = load_split(cfg)
    encoder = _encoder(cfg)
    sha = encoder.info.checkpoint_sha256
    feature_dir = resolve_path(cfg.output_dir) / "features"
    imported = {}
    for role, samples in (("query", queries), ("gallery", gallery)):
        if role == "gallery":
            pattern = f"{cfg.dataset.name}_{cfg.split}_{sha[:12]}_*_{len(gallery)}_gallery.npy"
            direct = feature_dir / f"{cfg.dataset.name}_{cfg.split}_{sha[:12]}_{len(gallery)}_gallery.npy"
        else:
            pattern = f"{cfg.dataset.name}_{cfg.split}_{sha[:12]}_{len(queries)}_{len(gallery)}_query.npy"
            direct = feature_dir / pattern
        path = direct if direct.is_file() else next(iter(sorted(feature_dir.glob(pattern))), None)
        if path is None or not Path(path).is_file():
            imported[role] = "no saved array"
            continue
        features = np.load(path)
        if features.shape[0] != len(samples):
            imported[role] = f"row count {features.shape[0]} does not match {len(samples)} images"
            continue
        indexes = [0, len(samples) // 2, len(samples) - 1]
        fresh = encoder.embed_images([load_rgb(samples[index].path) for index in indexes])
        mismatches = []
        for row, index in enumerate(indexes):
            cosine = float(np.dot(fresh[row], features[index]))
            if cosine < 0.999:
                mismatches.append({"index": index, "cosine": cosine})
        if mismatches:
            imported[role] = {"refused": True, "mismatches": mismatches}
            continue
        chunks = _store_features(store, samples, features, role, sha)
        imported[role] = {"rows": int(features.shape[0]), "new_chunks": chunks, "verified_rows": indexes}
    identity = observation_identity(cfg, sha)
    store.publish_json(f"meta/source_{cfg.split}_{identity['package_hash'][:16]}.json", identity)
    return {"identity": identity, "features": imported}


def measure_fresh_judgments(cfg: ExperimentConfig, run_dir: Path, count: int) -> dict:
    """Time distinct uncached judgments. This is the Colab GPU measurement, not a benchmark score."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to report a non-GPU batch as Colab throughput.")
    torch.cuda.reset_peak_memory_stats()
    store = RunStore(run_dir)
    store.prepare()
    store.acquire()
    try:
        set_seed(cfg.seed)
        queries, gallery, _report = load_split(cfg)
        encoder = _encoder(cfg)
        query_features = encoder.embed_images([load_rgb(sample.path) for sample in queries[: min(8, len(queries))]])
        gallery_features = encoder.embed_images([load_rgb(sample.path) for sample in gallery[: min(64, len(gallery))]])
        orders = rank_indices(query_features @ gallery_features.T)
        backend = QwenVLBackend(
            cfg.lvlm.model_id,
            cfg.lvlm.revision,
            cfg.lvlm.dtype,
            cfg.lvlm.max_new_tokens,
            cfg.lvlm.max_attempts,
            cfg.lvlm.min_pixels,
            cfg.lvlm.max_pixels,
        )
        started_load = __import__("time").perf_counter()
        backend.load()
        load_seconds = __import__("time").perf_counter() - started_load
        from reidvlm.lvlm.cache import JudgmentCache

        cache = JudgmentCache(resolve_path(cfg.cache_dir), backend.identity())
        timed = []
        invalid = 0
        for query_index in range(orders.shape[0]):
            if len(timed) >= count:
                break
            image_a = load_rgb(queries[query_index].path)
            for gallery_index in orders[query_index, : cfg.retrieval.k]:
                image_b = load_rgb(gallery[int(gallery_index)].path)
                for order, first, second in (("ab", image_a, image_b), ("ba", image_b, image_a)):
                    if len(timed) >= count:
                        break
                    hash_a = image_sha256(first)
                    hash_b = image_sha256(second)
                    if cache.get(hash_a, hash_b, order) or store.read_judgment(hash_a, hash_b, order):
                        continue
                    one = __import__("time").perf_counter()
                    record = backend.judge(first, second)
                    seconds = __import__("time").perf_counter() - one
                    payload = {
                        "identity": backend.identity(),
                        "hash_a": hash_a,
                        "hash_b": hash_b,
                        "order": order,
                        "valid": record.valid,
                        "parsed": None if record.parsed is None else record.parsed.model_dump(),
                        "attempts": [attempt.__dict__ for attempt in record.attempts],
                    }
                    cache.put(hash_a, hash_b, order, {"valid": payload["valid"], "parsed": payload["parsed"], "attempts": payload["attempts"]})
                    store.remember_judgment(payload)
                    tokens = [attempt.new_tokens for attempt in record.attempts if attempt.new_tokens is not None]
                    if not record.valid:
                        invalid += 1
                    timed.append({"seconds": seconds, "new_tokens": tokens, "valid": record.valid})
        store.flush()
        mean = None if not timed else sum(item["seconds"] for item in timed) / len(timed)
        token_values = [token for item in timed for token in item["new_tokens"]]
        summary = {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "dtype": cfg.lvlm.dtype,
            "do_sample": cfg.lvlm.do_sample,
            "max_new_tokens": cfg.lvlm.max_new_tokens,
            "load_seconds": load_seconds,
            "judgments": len(timed),
            "invalid": invalid,
            "invalid_rate": None if not timed else invalid / len(timed),
            "seconds_per_judgment": mean,
            "new_tokens_mean": None if not token_values else sum(token_values) / len(token_values),
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "prompt_version": PROMPT_VERSION,
            "model_revision": backend.identity().get("model_revision"),
        }
        if mean is not None:
            summary["projected_validation_seconds_160"] = mean * 160
            summary["projected_full_market_k5_seconds"] = mean * 33680
        store.publish_json("meta/throughput_" + sha256_json(summary)[:16] + ".json", summary)
        return summary
    finally:
        store.flush()
        store.release()


def execute_run(cfg: ExperimentConfig, run_dir: Path, allow_full_lvlm: bool, judgments: bool = True) -> dict:
    """Resume verified features and judgments. A test-split LVLM run stays closed until calibration exists."""
    store = RunStore(run_dir)
    store.prepare()
    store.acquire()
    try:
        set_seed(cfg.seed)
        queries, gallery, report = load_split(cfg)
        encoder = _encoder(cfg)
        sha = encoder.info.checkpoint_sha256
        identity = observation_identity(cfg, sha)
        store.publish_json(f"meta/source_{cfg.split}_{identity['package_hash'][:16]}.json", identity)
        if cfg.split == "test" and allow_full_lvlm:
            frozen = Path(resolve_path("configs/frozen") / f"{cfg.dataset.name}.yaml")
            if not frozen.is_file():
                return {
                    "started_lvlm": False,
                    "blocker": "full test LVLM is closed until a frozen validation config exists",
                    "identity": identity,
                    "status": store.status(),
                }
        query_hashes = [image_sha256(load_rgb(sample.path)) for sample in queries]
        gallery_hashes = [image_sha256(load_rgb(sample.path)) for sample in gallery]
        query_features = load_feature_rows(store, "query", query_hashes, sha)
        gallery_features = load_feature_rows(store, "gallery", gallery_hashes, sha)
        if query_features is None:
            from reidvlm.experiments import _embed

            query_features = _embed(encoder, queries, cfg.encoder.batch_size)
            _store_features(store, queries, query_features, "query", sha)
        if gallery_features is None:
            from reidvlm.experiments import _embed

            gallery_features = _embed(encoder, gallery, cfg.encoder.batch_size)
            _store_features(store, gallery, gallery_features, "gallery", sha)
        similarities = query_features @ gallery_features.T
        orders = rank_indices(similarities)
        q_pids, q_cams = _ids(queries)
        g_pids, g_cams = _ids(gallery)
        visual = evaluate_rankings(orders, q_pids, g_pids, q_cams, g_cams, cfg.retrieval.k)
        coverage = _metric_dict(visual, cfg.retrieval.k)
        coverage_payload = {
            "split": cfg.split,
            "query_count": len(queries),
            "gallery_count": len(gallery),
            "identities": len({int(sample.pid) for sample in queries}),
            "query_cameras": sorted({int(sample.camid) for sample in queries}),
            "gallery_cameras": sorted({int(sample.camid) for sample in gallery}),
            "metrics": coverage,
        }
        store.publish_json("meta/coverage_" + sha256_json(coverage_payload)[:16] + ".json", coverage_payload)
        validation = report.get("validation") or {}
        result = {
            "identity": identity,
            "coverage": coverage_payload,
            "query_count": len(queries),
            "gallery_count": len(gallery),
            "eligible_identities": validation.get("eligible_identities"),
            "selected_identity_count": validation.get("selected_identity_count"),
            "started_lvlm": False,
        }
        run_lvlm = judgments and (cfg.split == "validation" or allow_full_lvlm)
        if not run_lvlm:
            result["blocker"] = "LVLM judgments were not started. Measure a GPU batch and finish validation before a full test run."
            result["status"] = store.status()
            return result
        expected = []
        for query_index in range(orders.shape[0]):
            for gallery_index in orders[query_index, : cfg.retrieval.k]:
                gallery_index = int(gallery_index)
                query_hash = query_hashes[query_index]
                gallery_hash = gallery_hashes[int(gallery_index)]
                expected.append({"hash_a": query_hash, "hash_b": gallery_hash, "order": "ab", "query_index": query_index})
                expected.append({"hash_a": gallery_hash, "hash_b": query_hash, "order": "ba", "query_index": query_index})
        store.publish_json("meta/expected_" + sha256_json({"ids": expected})[:16] + ".json", {"ids": expected, "identity": identity})
        backend = QwenVLBackend(
            cfg.lvlm.model_id,
            cfg.lvlm.revision,
            cfg.lvlm.dtype,
            cfg.lvlm.max_new_tokens,
            cfg.lvlm.max_attempts,
            cfg.lvlm.min_pixels,
            cfg.lvlm.max_pixels,
        )
        records, stats = judge_shortlist(cfg, queries, gallery, orders, backend, store=store)
        result["started_lvlm"] = True
        result["judgment"] = stats
        if not stats.get("complete"):
            result["frozen"] = None
            result["selection"] = "not frozen because the validation judgments are incomplete"
            result["status"] = store.status()
            return result
        if cfg.split != "validation":
            result["status"] = store.status()
            return result
        grid = []
        best = None
        for alpha in ALPHAS:
            for min_cues in MIN_CUES:
                revised, _gate = _apply_variant(
                    similarities, orders, records, alpha, min_cues, "proposed", cfg.retrieval.k
                )
                metrics = evaluate_rankings(revised, q_pids, g_pids, q_cams, g_cams, cfg.retrieval.k)
                row = {"alpha": alpha, "min_shared_cues": min_cues, "mAP": metrics.map, "rank1": metrics.rank1}
                grid.append(row)
                rank_key = (metrics.map, metrics.rank1, -alpha, min_cues)
                if best is None or rank_key > best[0]:
                    best = (rank_key, row)
        chosen = best[1]
        frozen = {
            "selected_on": "training-derived validation",
            "encoder_saw_these_identities": True,
            "criterion": "highest validation mAP, then Rank-1, then smaller alpha, then larger min_shared_cues",
            "alpha": chosen["alpha"],
            "min_shared_cues": chosen["min_shared_cues"],
            "k": cfg.retrieval.k,
            "grid": grid,
            "identity_control_included": True,
            "coverage": coverage,
            "note": "If alpha is 0, the gate did not improve validation mAP over leaving the ranking unchanged.",
        }
        import yaml

        path = resolve_path("configs/frozen") / f"{cfg.dataset.name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            current = yaml.safe_load(path.read_text())
            if current.get("alpha") != frozen["alpha"] or current.get("min_shared_cues") != frozen["min_shared_cues"]:
                raise RuntimeError("a different frozen config already exists; it was not overwritten")
        else:
            path.write_text(yaml.safe_dump(frozen, sort_keys=False))
        store.publish_json("meta/frozen_" + sha256_json(frozen)[:16] + ".json", frozen)
        result["frozen"] = frozen
        result["frozen_path"] = str(path.relative_to(resolve_path(".")))
        result["status"] = store.status()
        return result
    finally:
        store.flush()
        store.release()
