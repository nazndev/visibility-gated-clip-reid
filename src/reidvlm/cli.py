"""Command line entry point. Search uses the same engine as the application."""

from __future__ import annotations

import argparse
import json
import sys

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
    }
    load_project_env()
    try:
        return commands[args.command](args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
