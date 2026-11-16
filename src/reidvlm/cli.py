"""Command line entry point. Search uses the same engine as the application."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from reidvlm.data.images import ImageReadError, load_rgb
from reidvlm.data.market1501 import Market1501
from reidvlm.data.msmt17 import MSMT17
from reidvlm.detection.detector import PersonDetector, select_detection
from reidvlm.detection.gallery import DemoGallery
from reidvlm.experiments import apply_frozen, calibrate, load_experiment, run_evaluation, write_executed_table
from reidvlm.retrieval.engine import SearchEngine
from reidvlm.runtime import auth_report, hardware_report, load_project_env, package_versions, project_root, resolve_path, source_revision


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_diagnose(_args) -> int:
    env = load_project_env()
    _print(
        {
            "hardware": hardware_report(),
            "packages": package_versions(),
            "revision": source_revision(),
            "HF_TOKEN_set": env["HF_TOKEN_set"],
            "token_source": env["token_source"],
        }
    )
    return 0


def cmd_auth(_args) -> int:
    report = auth_report()
    _print(report)
    if report.get("blocker"):
        return 2
    if report.get("access") in {"authentication_failed", "permission_failed"}:
        return 2
    return 0


def cmd_status(args) -> int:
    from reidvlm.persistence import RunStore

    store = RunStore(resolve_path(args.run_dir))
    if not store.run_dir.is_dir():
        print("Run directory does not exist.", file=sys.stderr)
        return 2
    _print(store.status())
    return 0


def cmd_import_cache(args) -> int:
    from reidvlm.persistence import RunStore

    store = RunStore(resolve_path(args.run_dir))
    store.prepare()
    _print(store.import_judgments(resolve_path(args.cache_dir)))
    return 0


def cmd_import_features(args) -> int:
    from reidvlm.runs import import_existing_features

    cfg = load_experiment(args.config)
    from reidvlm.persistence import RunStore

    store = RunStore(resolve_path(args.run_dir))
    store.prepare()
    _print(import_existing_features(cfg, store))
    return 0


def cmd_run(args) -> int:
    if not args.resume:
        print("Pass --resume. Completed judgments are reused; this command does not restart from zero.", file=sys.stderr)
        return 2
    from reidvlm.runs import execute_run

    cfg = load_experiment(args.config)
    if args.validation:
        cfg.split = "validation"
    summary = execute_run(
        cfg,
        resolve_path(args.run_dir),
        allow_full_lvlm=args.allow_full_lvlm,
        judgments=not args.features_only,
    )
    _print(summary)
    return 0 if summary.get("started_lvlm") or summary.get("coverage") else 2


def cmd_measure(args) -> int:
    import torch

    if not torch.cuda.is_available():
        print(
            "No CUDA device is visible. This measurement has to run on a Colab GPU. "
            "A local result would not be a Colab throughput number.",
            file=sys.stderr,
        )
        return 2
    from reidvlm.runs import measure_fresh_judgments

    cfg = load_experiment(args.config)
    _print(measure_fresh_judgments(cfg, resolve_path(args.run_dir), args.judgments))
    return 0


def cmd_verify_data(args) -> int:
    cfg = load_experiment(args.config)
    root = resolve_path(cfg.dataset.root)
    if cfg.dataset.name == "market1501":
        report = Market1501.load(root).report()
    elif cfg.dataset.name == "msmt17":
        report = MSMT17.load(root, cfg.dataset.version or "").report()
    else:
        print(f"Unknown dataset {cfg.dataset.name}", file=sys.stderr)
        return 2
    _print(report)
    mismatch = report.get("count_mismatch", {})
    return 1 if any(mismatch.values()) else 0


def cmd_verify_checkpoint(args) -> int:
    from reidvlm.encoders.clip_reid import CLIPReIDEncoder

    cfg = load_experiment(args.config)
    encoder = CLIPReIDEncoder(
        resolve_path(cfg.encoder.checkpoint),
        cfg.encoder.image_height,
        cfg.encoder.image_width,
        cfg.encoder.stride,
        cfg.encoder.pixel_mean,
        cfg.encoder.pixel_std,
        cfg.encoder.sie_camera,
        cfg.encoder.sie_coefficient,
    )
    _print(encoder.info.__dict__)
    return 0


def cmd_prepare_validation(args) -> int:
    cfg = load_experiment(args.config)
    cfg.split = "validation"
    from reidvlm.experiments import load_split

    _queries, _gallery, report = load_split(cfg)
    validation = report.get("validation", {})
    _print({key: validation[key] for key in validation if key not in {"queries", "gallery"}})
    return 0


def cmd_evaluate(args) -> int:
    cfg = load_experiment(args.config)
    if args.keep_ratio is not None:
        cfg.encoder.keep_ratio = args.keep_ratio
    if args.prune_layer is not None:
        cfg.encoder.prune_layer = args.prune_layer
    if args.frozen:
        cfg = apply_frozen(cfg)
    summary = run_evaluation(cfg, use_lvlm=not args.visual_only)
    _print({"public_path": summary["public_path"], "label": summary["label"], "methods": summary["methods"], "judgment": summary["judgment"]})
    return 0


def cmd_calibrate(args) -> int:
    cfg = load_experiment(args.config)
    cfg.split = "validation"
    if args.run_dir:
        from reidvlm.runs import execute_run

        _print(execute_run(cfg, resolve_path(args.run_dir), allow_full_lvlm=False))
        return 0
    _print(calibrate(cfg))
    return 0


def cmd_tables(_args) -> int:
    path = write_executed_table()
    print(path.relative_to(project_root()))
    return 0


def cmd_search(args) -> int:
    from reidvlm.encoders.clip_reid import CLIPReIDEncoder

    cfg = load_experiment(args.config) if args.config else None
    checkpoint = resolve_path(args.checkpoint or (cfg.encoder.checkpoint if cfg else "checkpoints/Market1501_clipreid_ViT-B-16_60.pth"))
    try:
        encoder = CLIPReIDEncoder(checkpoint)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    paths = []
    for entry in args.gallery:
        if entry.is_dir():
            paths.extend(sorted([*entry.glob("*.jpg"), *entry.glob("*.jpeg"), *entry.glob("*.png"), *entry.glob("*.JPG")]))
        else:
            paths.append(entry)
    gallery = DemoGallery()
    detector = None
    for path in paths:
        try:
            image = load_rgb(path)
        except ImageReadError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            return 2
        if args.full_scene:
            detector = detector or PersonDetector()
            gallery.add_full_scene(image, source_id=path.name, detector=detector)
        else:
            gallery.add_single_person_image(image, source_id=path.name)
    if len(gallery) == 0:
        print("Gallery is empty.", file=sys.stderr)
        return 2
    try:
        query = load_rgb(args.query)
    except ImageReadError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    selected_box = None
    if args.full_scene:
        detector = detector or PersonDetector()
        detections = detector.detect(query)
        if not detections:
            print("No person was detected in the query. The full photograph was not used as a person crop.", file=sys.stderr)
            return 2
        choice = args.select
        if choice is None and len(detections) == 1:
            choice = 1
        if choice is None:
            print("Several people were detected. Pass --select N. The largest box is not chosen.", file=sys.stderr)
            for number, detection in enumerate(detections, start=1):
                print(f"{number}: score {detection.score:.3f} box {list(detection.box)}", file=sys.stderr)
            return 2
        try:
            selected_box = list(detections[choice - 1].box)
            query = select_detection(query, detections, choice)
        except (ValueError, IndexError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    before = len(gallery)
    engine = SearchEngine(
        gallery.as_gallery_items(),
        encoder=encoder,
        k=args.k,
        alpha=args.alpha,
        min_shared_cues=args.min_shared_cues,
    )
    if args.mode != "visual":
        from reidvlm.lvlm.cache import JudgmentCache
        from reidvlm.lvlm.qwen import QwenVLBackend

        backend = QwenVLBackend()
        engine.lvlm = backend
        cache_dir = resolve_path(cfg.cache_dir if cfg else "caches/judgments")
        engine.cache = JudgmentCache(cache_dir, backend.identity())
    result = engine.search(query, mode=args.mode)
    if len(gallery) != before:
        print("Query was inserted into the gallery.", file=sys.stderr)
        return 1
    _print(
        {
            "mode": result.mode,
            "seconds": result.seconds,
            "gallery_size": before,
            "selected_query_box": selected_box,
            "cached_judgments": result.cached_judgments,
            "note": result.note,
            "hits": [
                {
                    "anonymous_id": hit.anonymous_id,
                    "source_id": hit.source_id,
                    "bbox": hit.bbox,
                    "original_rank": hit.original_rank,
                    "revised_rank": hit.revised_rank,
                    "visual_similarity": hit.visual_similarity,
                    "corrected_score": hit.corrected_score,
                    "gate_accepted": hit.gate_accepted,
                    "evidence": hit.evidence,
                }
                for hit in result.hits
            ],
        }
    )
    return 0


def _encoder_from_config(config_path: str):
    from reidvlm.encoders.clip_reid import CLIPReIDEncoder

    config = load_experiment(config_path)
    enc = config.encoder
    return config, CLIPReIDEncoder(
        resolve_path(enc.checkpoint),
        enc.image_height,
        enc.image_width,
        enc.stride,
        enc.pixel_mean,
        enc.pixel_std,
        enc.sie_camera,
        enc.sie_coefficient,
        device=None,
        keep_ratio=1.0,
        prune_layer=6,
    )


def cmd_prune_plan(args) -> int:
    from reidvlm.pruning.study import write_plan

    path = resolve_path(args.output)
    plan = write_plan(path)
    _print({"wrote": str(path), "configurations": len(plan["configurations"]), "seeds": plan["random_seeds"]})
    return 0


def cmd_prune_manifest(args) -> int:
    from reidvlm.pruning.study import build_validation_manifest, manifest_counts

    config = load_experiment(args.config)
    market = Market1501.load(resolve_path(config.dataset.root))
    manifest = build_validation_manifest(market.train, seed=553)
    destination = resolve_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest))
    counts = manifest_counts(manifest)
    counts["note"] = "The encoder was trained on these identities. This is not an identity-unseen split."
    _print(counts)
    return 0


def cmd_prune_benchmark(args) -> int:
    from reidvlm.data.images import load_rgb
    from reidvlm.encoders.clip_reid import preprocess
    from reidvlm.pruning.benchmark import time_methods
    from reidvlm.pruning.selection import SelectionConfig, content_id_from_tensor

    config, encoder = _encoder_from_config(args.config)
    market = Market1501.load(resolve_path(config.dataset.root))
    images = [load_rgb(Path(sample.path)) for sample in market.query[: args.batch_size]]
    tensor = torch.stack(
        [
            preprocess(image, encoder.image_height, encoder.image_width, encoder.pixel_mean, encoder.pixel_std)
            for image in images
        ]
    )
    content_ids = [content_id_from_tensor(row) for row in tensor]
    patches = encoder.model.image_encoder.grid_h * encoder.model.image_encoder.grid_w
    methods = [
        {"name": "full", "selection": None},
        {"name": "cls_attention_reference", "selection": SelectionConfig("cls_attention", args.prune_block, args.keep, None, "reference", patches)},
        {"name": "cls_attention_cls_row", "selection": SelectionConfig("cls_attention", args.prune_block, args.keep, None, "cls_row", patches)},
        {"name": "random", "selection": SelectionConfig("random", args.prune_block, args.keep, args.seed, "reference", patches)},
    ]
    measured = time_methods(encoder, tensor, content_ids, methods, args.warmup, args.iterations, args.trials, order_seed=553)
    full_ms = float(np.median([row["median_batch_wall_ms"] for row in measured["methods"]["full"]]))
    rows = []
    for name, trials in measured["methods"].items():
        for row in trials:
            row["speedup_vs_full_wall"] = None if not full_ms else full_ms / row["median_batch_wall_ms"]
            rows.append(row)
    payload = {
        "label": "benchmark",
        "protocol_requested": {"warmup": 10, "iterations": 50, "trials": 3, "batch_sizes": [1, 32], "precision": "float32", "autocast": False},
        "protocol_executed": {"warmup": args.warmup, "iterations": args.iterations, "trials": args.trials, "batch_size": args.batch_size},
        "planned_protocol": args.warmup == 10 and args.iterations == 50 and args.trials == 3 and args.batch_size in (1, 32),
        "order_seed": measured["order_seed"],
        "trial_orders": measured["trial_orders"],
        "note": "Device-resident encoder only. CUDA-event time is stored separately from synchronized wall-clock time. Image loading is excluded.",
        "rows": rows,
    }
    if args.output:
        path = resolve_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    _print(payload)
    return 0


def cmd_prune_smoke(args) -> int:
    from reidvlm.data.images import load_rgb
    from reidvlm.pruning.selection import SelectionConfig

    config, encoder = _encoder_from_config(args.config)
    market = Market1501.load(resolve_path(config.dataset.root))
    chosen = market.query[: args.images]
    images = [load_rgb(Path(sample.path)) for sample in chosen]
    patches = encoder.model.image_encoder.grid_h * encoder.model.image_encoder.grid_w
    encoder.set_selection(None)
    full = encoder.embed_images(images)
    attention = SelectionConfig("cls_attention", 6, 64, None, "reference", patches)
    encoder.set_selection(attention)
    pruned = encoder.embed_images(images)
    attention_tokens = list(encoder.model.image_encoder.last_tokens_per_block)
    reference_indices = encoder.model.image_encoder.last_kept_patch_index.detach().cpu()
    encoder.set_selection(SelectionConfig("cls_attention", 6, 64, None, "cls_row", patches))
    alternative = encoder.embed_images(images)
    cls_row_indices = encoder.model.image_encoder.last_kept_patch_index.detach().cpu()
    encoder.set_selection(SelectionConfig("random", 6, 64, 11, "reference", patches))
    random_a = encoder.embed_images(images)
    random_b = encoder.embed_images(list(reversed(images)))
    payload = {
        "label": "smoke",
        "not_a_market_score": True,
        "images": args.images,
        "checkpoint_sha256": encoder.info.checkpoint_sha256,
        "tokens_per_block_attention": attention_tokens,
        "descriptor_dim": int(full.shape[1]),
        "max_abs_full_vs_attention": float(np.max(np.abs(full - pruned))),
        "max_abs_reference_vs_cls_row": float(np.max(np.abs(pruned - alternative))),
        "reference_vs_cls_row_indices_equal": bool(torch.equal(reference_indices, cls_row_indices)),
        "reference_vs_cls_row_index_mismatches": int((reference_indices != cls_row_indices).sum()),
        "random_order_max_abs": float(np.max(np.abs(random_a - random_b[::-1]))),
        "names": [Path(sample.path).name for sample in chosen],
    }
    if args.output:
        path = resolve_path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    _print(payload)
    return 0


def cmd_prune_embed(args) -> int:
    """Embed one configuration. A limit marks the file as a smoke test and cannot select a configuration."""
    from reidvlm.data.images import load_rgb
    from reidvlm.encoders.clip_reid import preprocess
    from reidvlm.persistence import RunStore
    from reidvlm.pruning.identity import execution_identity, file_sha256
    from reidvlm.pruning.selection import SelectionConfig, content_id_from_tensor, same_selection
    from reidvlm.pruning.study import build_validation_manifest, resume_embed_chunks, score_split
    from reidvlm.runtime import sha256_file, source_revision

    config, encoder = _encoder_from_config(args.config)
    market = Market1501.load(resolve_path(config.dataset.root))
    if args.split == "validation":
        manifest = build_validation_manifest(market.train, seed=553)
        queries = manifest["queries"]
        gallery = manifest["gallery"]
    elif args.split == "test":
        queries = [{"path": sample.path, "pid": sample.pid, "camid": sample.camid} for sample in market.query]
        gallery = [{"path": sample.path, "pid": sample.pid, "camid": sample.camid} for sample in market.gallery]
        manifest = {"split": "official_test", "query_images": len(queries), "gallery_images": len(gallery)}
    else:
        raise ValueError(args.split)
    limited = args.limit is not None
    if limited:
        queries = queries[: args.limit]
        gallery = gallery[: args.limit]
    patches = encoder.model.image_encoder.grid_h * encoder.model.image_encoder.grid_w
    selection = SelectionConfig(args.strategy, args.prune_block, args.keep if args.strategy != "none" else patches, args.seed, args.backend, patches)
    same_selection(selection, selection)
    encoder.set_token_pruning(1.0, 6)
    encoder.set_selection(selection if args.strategy != "none" else None)
    ordered = queries + gallery
    manifest_ids = [item["path"] for item in ordered]
    manifest_rows = []
    for index, item in enumerate(ordered):
        manifest_rows.append(
            {
                "role": "query" if index < len(queries) else "gallery",
                "name": Path(item["path"]).name,
                "content_sha256": file_sha256(Path(item["path"])),
                "pid": item["pid"],
                "camid": item["camid"],
            }
        )
    identity = execution_identity(
        {
            "selection": selection.identity(),
            "checkpoint_sha256": sha256_file(resolve_path(config.encoder.checkpoint)),
            "preprocess": {
                "height": encoder.image_height,
                "width": encoder.image_width,
                "mean": encoder.pixel_mean,
                "std": encoder.pixel_std,
                "descriptor": "bn768_cls_cat_bn512_proj_l2",
                "implementation": "reidvlm.encoders.clip_reid.preprocess",
            },
            "manifest": manifest_rows,
            "precision": "float32",
            "autocast": False,
            "split": args.split,
            "limited": limited,
            "counts": {"queries": len(queries), "gallery": len(gallery)},
        }
    )
    by_path = {item["path"]: item for item in ordered}
    store = RunStore(resolve_path(args.run_dir))
    store.prepare()

    def embed_missing(paths: list[str]) -> np.ndarray:
        images = [load_rgb(Path(path)) for path in paths]
        rows = []
        batch = args.batch_size
        for start in range(0, len(images), batch):
            chunk = images[start : start + batch]
            tensors = torch.stack(
                [preprocess(image, encoder.image_height, encoder.image_width, encoder.pixel_mean, encoder.pixel_std) for image in chunk]
            )
            ids = [content_id_from_tensor(row) for row in tensors]
            with torch.inference_mode():
                rows.append(encoder.model(tensors.to(encoder.device), None, ids).detach().cpu().numpy())
        return np.concatenate(rows, axis=0)

    features = resume_embed_chunks(store, identity, manifest_ids, embed_missing, chunk_size=args.batch_size)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    query_features = features[: len(queries)]
    gallery_features = features[len(queries) :]
    scored = score_split(query_features, gallery_features, queries, gallery)
    evaluation = (
        {
            "protocol": "training-derived validation",
            "identity_seen": True,
            "note": "The encoder was trained on these identities and source training images. This is not an official Market-1501 test result.",
        }
        if args.split == "validation"
        else {
            "protocol": "official Market-1501 test",
            "identity_seen": False,
            "note": "Official query and gallery lists after junk removal. Same-person same-camera gallery images are excluded at scoring time.",
        }
    )
    payload = {
        "label": "smoke" if limited else "full",
        "not_a_market_score": bool(limited or args.split != "test"),
        "evaluation": evaluation,
        "identity": identity,
        "counts": {"queries": len(queries), "gallery": len(gallery)},
        "scores": {key: value for key, value in scored.items() if key not in {"per_query_ap", "per_query_first_positive_rank"}},
        "per_query_ap": scored["per_query_ap"],
        "per_query_first_positive_rank": scored["per_query_first_positive_rank"],
        "kreciprocal": "not_run",
        "git": source_revision(),
        "legacy_cache_note": "Runs saved before chunks-v1 are not reused. Their files stay in place. Image-content hashes were not stored for those runs, so equivalence with this code is not assumed.",
    }
    destination = resolve_path(args.output) if args.output else resolve_path(args.run_dir) / "score.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload))
    printable = {key: value for key, value in payload.items() if key not in {"per_query_ap", "per_query_first_positive_rank", "identity"}}
    printable["execution_sha256"] = identity.get("execution_sha256")
    printable["execution_source_sha256"] = identity.get("execution_source_sha256")
    printable["checkpoint_sha256"] = identity.get("checkpoint_sha256")
    printable["selection"] = identity.get("selection")
    printable["git"] = payload["git"]
    _print(printable)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reidvlm")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("diagnose")
    sub.add_parser("auth")
    for name in ("verify-data", "verify-checkpoint", "prepare-validation"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--run-dir", default=None)
    status = sub.add_parser("status")
    status.add_argument("--run-dir", required=True)
    importer = sub.add_parser("import-cache")
    importer.add_argument("--run-dir", required=True)
    importer.add_argument("--cache-dir", required=True)
    features = sub.add_parser("import-features")
    features.add_argument("--config", required=True)
    features.add_argument("--run-dir", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--validation", action="store_true")
    run.add_argument("--features-only", action="store_true")
    run.add_argument("--allow-full-lvlm", action="store_true")
    measure = sub.add_parser("measure")
    measure.add_argument("--config", required=True)
    measure.add_argument("--run-dir", required=True)
    measure.add_argument("--judgments", type=int, default=20)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--visual-only", action="store_true")
    evaluate.add_argument("--frozen", action="store_true")
    evaluate.add_argument("--keep-ratio", type=float, default=None)
    evaluate.add_argument("--prune-layer", type=int, default=None)
    sub.add_parser("tables")
    search = sub.add_parser("search")
    search.add_argument("--query", type=resolve_path, required=True)
    search.add_argument("--gallery", type=resolve_path, nargs="+", required=True)
    search.add_argument("--config", default=None)
    search.add_argument("--checkpoint", default=None)
    search.add_argument("--mode", default="visual", choices=["visual", "proposed", "ungated", "visibility", "consistency"])
    search.add_argument("--k", type=int, default=5)
    search.add_argument("--alpha", type=float, default=0.05)
    search.add_argument("--min-shared-cues", type=int, default=2)
    search.add_argument("--full-scene", action="store_true")
    search.add_argument("--select", type=int, default=None)
    plan = sub.add_parser("prune-plan")
    plan.add_argument("--output", default="configs/iteration2_plan.json")
    manifest = sub.add_parser("prune-manifest")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--output", default="outputs/iteration2/validation_manifest.json")
    smoke = sub.add_parser("prune-smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--images", type=int, default=2)
    smoke.add_argument("--output", default="outputs/iteration2/smoke.json")
    bench = sub.add_parser("prune-benchmark")
    bench.add_argument("--config", required=True)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--iterations", type=int, default=50)
    bench.add_argument("--trials", type=int, default=3)
    bench.add_argument("--prune-block", type=int, default=6)
    bench.add_argument("--keep", type=int, default=64)
    bench.add_argument("--seed", type=int, default=11)
    bench.add_argument("--output", default="outputs/iteration2/benchmark.json")
    embed = sub.add_parser("prune-embed")
    embed.add_argument("--config", required=True)
    embed.add_argument("--split", choices=["validation", "test"], required=True)
    embed.add_argument("--strategy", choices=["none", "cls_attention", "random"], required=True)
    embed.add_argument("--prune-block", type=int, default=6)
    embed.add_argument("--keep", type=int, default=64)
    embed.add_argument("--seed", type=int, default=None)
    embed.add_argument("--backend", choices=["reference", "cls_row"], default="reference")
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--limit", type=int, default=None)
    embed.add_argument("--run-dir", default="outputs/iteration2/embed")
    embed.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "diagnose": cmd_diagnose,
        "auth": cmd_auth,
        "verify-data": cmd_verify_data,
        "verify-checkpoint": cmd_verify_checkpoint,
        "prepare-validation": cmd_prepare_validation,
        "evaluate": cmd_evaluate,
        "calibrate": cmd_calibrate,
        "tables": cmd_tables,
        "search": cmd_search,
        "status": cmd_status,
        "import-cache": cmd_import_cache,
        "import-features": cmd_import_features,
        "run": cmd_run,
        "measure": cmd_measure,
        "prune-plan": cmd_prune_plan,
        "prune-manifest": cmd_prune_manifest,
        "prune-smoke": cmd_prune_smoke,
        "prune-benchmark": cmd_prune_benchmark,
        "prune-embed": cmd_prune_embed,
    }
    load_project_env()
    try:
        return commands[args.command](args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
