"""Experiment commands shared with the CLI."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from reidvlm.config import ExperimentConfig, load_config
from reidvlm.data.images import load_rgb
from reidvlm.data.market1501 import Market1501, Sample
from reidvlm.data.msmt17 import MSMT17
from reidvlm.data.validation import build_validation, load_manifest, save_manifest
from reidvlm.encoders.clip_reid import CLIPReIDEncoder
from reidvlm.evaluation.metrics import evaluate_rankings, top1_changes
from reidvlm.lvlm.cache import JudgmentCache
from reidvlm.lvlm.prompts import PROMPT_VERSION
from reidvlm.lvlm.qwen import LVLMUnavailable, QwenVLBackend
from reidvlm.lvlm.schema import SCHEMA_VERSION
from reidvlm.retrieval.cosine import rank_indices
from reidvlm.retrieval.engine import judgment_view, record_from_dict, record_to_dict
from reidvlm.retrieval.gate import decision_for
from reidvlm.retrieval.kreciprocal import k_reciprocal_rerank
from reidvlm.retrieval.rerank import rerank_shortlist
from reidvlm.runtime import hardware_report, package_versions, peak_gpu_bytes, peak_rss_bytes, project_root, resolve_path, set_seed, source_revision

ALPHAS = (0.0, 0.025, 0.05, 0.10)
MIN_CUES = (1, 2, 3)
GATED = ("ungated", "proposed", "visibility", "consistency")


def _samples_from_manifest(items: list[dict]) -> list[Sample]:
    return [Sample(item["path"], int(item["pid"]), int(item["camid"]), False) for item in items]


def load_split(cfg: ExperimentConfig):
    root = resolve_path(cfg.dataset.root)
    if cfg.dataset.name == "market1501":
        data = Market1501.load(root)
        report = data.report()
        if cfg.split == "test":
            return data.query, data.gallery, report
        if cfg.split != "validation":
            raise ValueError("split must be test or validation")
        manifest_path = resolve_path(cfg.output_dir) / "validation_market1501.json"
        if manifest_path.is_file():
            manifest = load_manifest(manifest_path)
        else:
            manifest = build_validation(data.train, cfg.seed, cfg.limits.validation_queries)
            save_manifest(manifest, manifest_path)
        return _samples_from_manifest(manifest["queries"]), _samples_from_manifest(manifest["gallery"]), {**report, "validation": manifest}
    if cfg.dataset.name == "msmt17":
        if not cfg.dataset.version:
            raise ValueError("MSMT17 requires dataset.version to be V1 or V2.")
        data = MSMT17.load(root, cfg.dataset.version)
        report = data.report()
        if cfg.split == "validation":
            return data.validation, data.train, report
        return data.query, data.gallery, report
    raise ValueError(f"Unsupported dataset {cfg.dataset.name}")


def _encoder(cfg: ExperimentConfig) -> CLIPReIDEncoder:
    enc = cfg.encoder
    return CLIPReIDEncoder(
        resolve_path(enc.checkpoint),
        enc.image_height,
        enc.image_width,
        enc.stride,
        enc.pixel_mean,
        enc.pixel_std,
        enc.sie_camera,
        enc.sie_coefficient,
    )


def _embed(encoder: CLIPReIDEncoder, samples: list[Sample], batch_size: int) -> np.ndarray:
    rows = []
    for start in tqdm(range(0, len(samples), batch_size), desc="embed"):
        batch = samples[start : start + batch_size]
        images = [load_rgb(sample.path) for sample in batch]
        rows.append(encoder.embed_images(images, batch_size=batch_size))
    return np.concatenate(rows, axis=0)


def _metric_dict(result, k: int) -> dict:
    return {
        "mAP": result.map,
        "rank1": result.rank1,
        "rank5": result.rank5,
        "rank10": result.rank10,
        "valid_queries": result.valid_queries,
        "excluded_queries": result.excluded_queries,
        "shortlist_hit_rate": result.shortlist_hit_rate,
        "positive_coverage_at_k": result.positive_coverage_at_k,
        "k": k,
    }


def _ids(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([sample.pid for sample in samples]), np.asarray([sample.camid for sample in samples])


def _apply_variant(similarities: np.ndarray, orders: np.ndarray, pair_records: dict, alpha: float, min_cues: int, variant: str, k: int):
    revised = []
    accepted = 0
    judged = 0
    invalid = 0
    for query_index, row in enumerate(similarities):
        deltas = np.zeros(row.shape[0], dtype=np.float64)
        if variant != "visual":
            for gallery_index in orders[query_index, :k]:
                gallery_index = int(gallery_index)
                forward = pair_records.get((query_index, gallery_index, "ab"))
                backward = pair_records.get((query_index, gallery_index, "ba"))
                if forward is None or backward is None:
                    continue
                judged += 1
                if not forward.valid or not backward.valid:
                    invalid += 1
                decision = decision_for(judgment_view(forward), judgment_view(backward), alpha, min_cues, variant)
                deltas[gallery_index] = decision.delta
                accepted += int(decision.accepted)
        revised.append(rerank_shortlist(row, deltas, k if variant != "visual" else 0))
    return np.stack(revised), {"judged_pairs": judged, "invalid_pairs": invalid, "gate_accepted_pairs": accepted}


def judge_shortlist(cfg: ExperimentConfig, queries, gallery, orders, backend: QwenVLBackend, store=None) -> tuple[dict, dict]:
    from reidvlm.data.images import image_sha256

    backend.load()
    cache = JudgmentCache(resolve_path(cfg.cache_dir), backend.identity())
    records = {}
    calls = 0
    cached_hits = 0
    fresh = 0
    fresh_seconds = []
    limit_note = None
    expected = 0
    started = time.perf_counter()
    images = {}

    def image_for(sample: Sample):
        if sample.path not in images:
            images[sample.path] = load_rgb(sample.path)
        return images[sample.path]

    for query_index in range(orders.shape[0]):
        if limit_note:
            break
        for gallery_index in orders[query_index, : cfg.retrieval.k]:
            gallery_index = int(gallery_index)
            for order, first, second in (
                ("ab", queries[query_index], gallery[gallery_index]),
                ("ba", gallery[gallery_index], queries[query_index]),
            ):
                expected += 1
                if cfg.limits.max_calls is not None and calls >= cfg.limits.max_calls:
                    limit_note = "stopped at max_calls"
                    break
                if cfg.limits.max_seconds is not None and (time.perf_counter() - started) >= cfg.limits.max_seconds:
                    limit_note = "stopped at max_seconds"
                    break
                image_a = image_for(first)
                image_b = image_for(second)
                hash_a = image_sha256(image_a)
                hash_b = image_sha256(image_b)
                cached = cache.get(hash_a, hash_b, order)
                stored = None if store is None else store.read_judgment(hash_a, hash_b, order)
                if cached is None and stored is not None and cache._matches(stored.get("identity") or {}):
                    cached = stored
                one = time.perf_counter()
                if cached is not None:
                    record = record_from_dict(cached)
                    cached_hits += 1
                    cache.put(hash_a, hash_b, order, {"valid": cached.get("valid"), "parsed": cached.get("parsed"), "attempts": cached.get("attempts", [])})
                else:
                    record = backend.judge(image_a, image_b)
                    payload = record_to_dict(record)
                    cache.put(hash_a, hash_b, order, payload)
                    if store is not None:
                        store.remember_judgment(
                            {"identity": backend.identity(), "hash_a": hash_a, "hash_b": hash_b, "order": order, **payload}
                        )
                    fresh += 1
                    fresh_seconds.append(time.perf_counter() - one)
                calls += record.attempt_count
                records[(query_index, gallery_index, order)] = record
            if limit_note:
                break
    if store is not None:
        store.flush()
    remaining_pairs = max(0, (expected - len(records)) / 2)
    stats = {
        "calls_including_retries": calls,
        "cached_judgments": cached_hits,
        "fresh_judgments": fresh,
        "seconds": time.perf_counter() - started,
        "throughput": None
        if not fresh_seconds
        else {
            "seconds_per_uncached_judgment": sum(fresh_seconds) / len(fresh_seconds),
            "uncached_judgments_timed": len(fresh_seconds),
            "projected_remaining_seconds": (sum(fresh_seconds) / len(fresh_seconds)) * remaining_pairs * 2,
        },
        "limit": limit_note,
        "complete": limit_note is None and len(records) == expected and expected > 0,
        "model_identity": backend.identity(),
        "peak_gpu_bytes": peak_gpu_bytes(),
    }
    return records, stats


def run_evaluation(cfg: ExperimentConfig, use_lvlm: bool) -> dict:
    set_seed(cfg.seed)
    queries, gallery, data_report = load_split(cfg)
    subset = False
    if cfg.limits.max_queries is not None and len(queries) > cfg.limits.max_queries:
        queries = queries[: cfg.limits.max_queries]
        subset = True
    encoder = _encoder(cfg)
    sha = encoder.info.checkpoint_sha256[:12]
    feature_dir = resolve_path(cfg.output_dir) / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    query_cache = feature_dir / f"{cfg.dataset.name}_{cfg.split}_{sha}_{len(queries)}_{len(gallery)}_query.npy"
    gallery_cache = feature_dir / f"{cfg.dataset.name}_{cfg.split}_{sha}_{len(gallery)}_gallery.npy"
    started = time.perf_counter()
    reused = list(feature_dir.glob(f"{cfg.dataset.name}_{cfg.split}_{sha}_*_{len(gallery)}_gallery.npy"))
    if gallery_cache.is_file():
        gallery_features = np.load(gallery_cache)
    elif reused:
        gallery_features = np.load(reused[0])
        np.save(gallery_cache, gallery_features)
    else:
        gallery_features = _embed(encoder, gallery, cfg.encoder.batch_size)
        np.save(gallery_cache, gallery_features)
    if query_cache.is_file():
        query_features = np.load(query_cache)
    else:
        query_features = _embed(encoder, queries, cfg.encoder.batch_size)
        np.save(query_cache, query_features)
    embed_seconds = time.perf_counter() - started
    similarities = query_features @ gallery_features.T
    orders = rank_indices(similarities)
    q_pids, q_cams = _ids(queries)
    g_pids, g_cams = _ids(gallery)
    methods = {}
    visual = evaluate_rankings(orders, q_pids, g_pids, q_cams, g_cams, cfg.retrieval.k)
    methods["visual"] = _metric_dict(visual, cfg.retrieval.k)
    krecip_note = "not run"
    if cfg.retrieval.run_kreciprocal:
        try:
            distances = k_reciprocal_rerank(
                query_features,
                gallery_features,
                cfg.retrieval.k1,
                cfg.retrieval.k2,
                cfg.retrieval.kreciprocal_lambda,
            )
            k_orders = np.argsort(distances, axis=1, kind="mergesort")
            k_result = evaluate_rankings(k_orders, q_pids, g_pids, q_cams, g_cams, cfg.retrieval.k)
            methods["kreciprocal"] = _metric_dict(k_result, cfg.retrieval.k)
            methods["kreciprocal"]["parameters"] = {
                "k1": cfg.retrieval.k1,
                "k2": cfg.retrieval.k2,
                "lambda": cfg.retrieval.kreciprocal_lambda,
                "uses_query_query_and_gallery_gallery": True,
            }
            krecip_note = "completed"
        except MemoryError as exc:
            krecip_note = f"blocked: {exc}"
    judgment_stats = None
    records = {}
    if use_lvlm:
        backend = QwenVLBackend(
            cfg.lvlm.model_id,
            cfg.lvlm.revision,
            cfg.lvlm.dtype,
            cfg.lvlm.max_new_tokens,
            cfg.lvlm.max_attempts,
            cfg.lvlm.min_pixels,
            cfg.lvlm.max_pixels,
        )
        try:
            records, judgment_stats = judge_shortlist(cfg, queries, gallery, orders, backend)
        except LVLMUnavailable as exc:
            judgment_stats = {"complete": False, "error": str(exc)}
            records = {}
        if records:
            for variant in GATED:
                revised, gate_stats = _apply_variant(
                    similarities,
                    orders,
                    records,
                    cfg.retrieval.alpha,
                    cfg.retrieval.min_shared_cues,
                    variant,
                    cfg.retrieval.k,
                )
                result = evaluate_rankings(revised, q_pids, g_pids, q_cams, g_cams, cfg.retrieval.k)
                methods[variant] = {**_metric_dict(result, cfg.retrieval.k), **gate_stats, **top1_changes(visual, result)}
    label = "preliminary query-subset evaluation" if subset else "full split evaluation"
    summary = {
        "label": label,
        "dataset": cfg.dataset.name,
        "split": cfg.split,
        "query_count": len(queries),
        "gallery_count": len(gallery),
        "query_names": [Path(sample.path).name for sample in queries],
        "gallery_was_subset": False,
        "seed": cfg.seed,
        "alpha": cfg.retrieval.alpha,
        "min_shared_cues": cfg.retrieval.min_shared_cues,
        "k": cfg.retrieval.k,
        "checkpoint_sha256": encoder.info.checkpoint_sha256,
        "positional_tokens": encoder.info.positional_tokens,
        "omitted_training_only_keys": encoder.info.omitted_training_only_keys,
        "encoder_device": encoder.info.device,
        "encoder_dtype": encoder.info.dtype,
        "embed_seconds": embed_seconds,
        "methods": methods,
        "judgment": judgment_stats,
        "kreciprocal": krecip_note,
        "data_report": {key: value for key, value in data_report.items() if key != "validation"},
        "hardware": hardware_report(),
        "packages": package_versions(),
        "source_revision": source_revision(),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "peak_rss_bytes": peak_rss_bytes(),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    out = resolve_path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{cfg.dataset.name}_{cfg.split}_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2))
    public = public_summary(summary)
    results = project_root() / "results"
    results.mkdir(parents=True, exist_ok=True)
    public_path = results / f"{cfg.dataset.name}_{cfg.split}_{stamp}.json"
    public_path.write_text(json.dumps(public, indent=2))
    summary["output_path"] = str(path.relative_to(project_root()))
    summary["public_path"] = str(public_path.relative_to(project_root()))
    return summary


def public_summary(summary: dict) -> dict:
    """Keep a tracked summary free of filenames and host names."""
    keep = dict(summary)
    keep.pop("query_names", None)
    keep.pop("output_path", None)
    hardware = dict(keep.get("hardware") or {})
    hardware.pop("platform", None)
    keep["hardware"] = hardware
    return keep


def calibrate(cfg: ExperimentConfig) -> dict:
    """Select alpha and min_shared_cues on training-derived validation only."""
    if cfg.split != "validation":
        raise ValueError("Calibration must use the validation split.")
    set_seed(cfg.seed)
    queries, gallery, _ = load_split(cfg)
    encoder = _encoder(cfg)
    query_features = _embed(encoder, queries, cfg.encoder.batch_size)
    gallery_features = _embed(encoder, gallery, cfg.encoder.batch_size)
    similarities = query_features @ gallery_features.T
    orders = rank_indices(similarities)
    backend = QwenVLBackend(
        cfg.lvlm.model_id,
        cfg.lvlm.revision,
        cfg.lvlm.dtype,
        cfg.lvlm.max_new_tokens,
        cfg.lvlm.max_attempts,
        cfg.lvlm.min_pixels,
        cfg.lvlm.max_pixels,
    )
    records, stats = judge_shortlist(cfg, queries, gallery, orders, backend)
    if not stats["complete"]:
        return {"frozen": None, "selection": "not frozen because the validation judgments are incomplete", "judgment": stats}
    q_pids, q_cams = _ids(queries)
    g_pids, g_cams = _ids(gallery)
    grid = []
    best = None
    for alpha in ALPHAS:
        for min_cues in MIN_CUES:
            revised, _ = _apply_variant(similarities, orders, records, alpha, min_cues, "proposed", cfg.retrieval.k)
            result = evaluate_rankings(revised, q_pids, g_pids, q_cams, g_cams, cfg.retrieval.k)
            row = {"alpha": alpha, "min_shared_cues": min_cues, "mAP": result.map, "rank1": result.rank1}
            grid.append(row)
            rank_key = (result.map, result.rank1, -alpha, min_cues)
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
        "note": "If alpha is 0, the gate did not improve validation mAP over leaving the ranking unchanged.",
    }
    path = project_root() / "configs" / "frozen" / f"{cfg.dataset.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(frozen, sort_keys=False))
    return {"frozen_path": str(path.relative_to(project_root())), "frozen": frozen}


def apply_frozen(cfg: ExperimentConfig) -> ExperimentConfig:
    path = project_root() / "configs" / "frozen" / f"{cfg.dataset.name}.yaml"
    if not path.is_file():
        return cfg
    frozen = yaml.safe_load(path.read_text())
    cfg.retrieval.alpha = float(frozen["alpha"])
    cfg.retrieval.min_shared_cues = int(frozen["min_shared_cues"])
    cfg.retrieval.k = int(frozen["k"])
    return cfg


def write_executed_table() -> Path:
    rows = []
    for path in sorted((project_root() / "results").glob("*.json")):
        payload = json.loads(path.read_text())
        if "methods" not in payload:
            continue
        rows.append(payload)
    lines = [
        "# Executed results",
        "",
        "These rows are produced from files in `results/`. Literature numbers are not included.",
        "",
    ]
    if not rows:
        lines.append("No executed result files are present yet.")
    for payload in rows:
        lines.append(
            f"- {payload.get('dataset')} {payload.get('split')} ({payload.get('label')}): "
            f"queries={payload.get('query_count')} gallery={payload.get('gallery_count')} "
            f"checkpoint={str(payload.get('checkpoint_sha256', ''))[:12]}"
        )
        for name, metrics in payload.get("methods", {}).items():
            lines.append(
                f"  - {name}: mAP={metrics.get('mAP'):.4f} Rank-1={metrics.get('rank1'):.4f}"
                if isinstance(metrics.get("mAP"), float)
                else f"  - {name}: {metrics}"
            )
    destination = project_root() / "results" / "executed_table.md"
    destination.write_text("\n".join(lines) + "\n")
    return destination


def load_experiment(path: str) -> ExperimentConfig:
    return load_config(path)
