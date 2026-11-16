"""Device-resident encoder timing.

Image loading and preprocessing are outside this scope. Selection, including
CPU index generation for random pruning, is inside it.

The short MPS timings recorded before this correction ran without
torch.inference_mode(). Those numbers are preliminary and are not repeated
here. Their original values stay in RESEARCH_ITERATION_2_STATUS.md.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from reidvlm.runtime import hardware_report, package_versions


def freeze_encoder(encoder) -> None:
    encoder.model.eval()
    for parameter in encoder.model.parameters():
        parameter.requires_grad_(False)


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _attention_settings() -> dict:
    settings = {}
    cuda = getattr(torch.backends, "cuda", None)
    if cuda is None:
        return settings
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        function = getattr(cuda, name, None)
        if not callable(function):
            continue
        try:
            settings[name] = bool(function())
        except Exception:
            settings[name] = None
    return settings


def _device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return device


def _forward(encoder, batch: torch.Tensor, content_ids: list[str]) -> torch.Tensor:
    features = encoder.model(batch, None, content_ids)
    return torch.nn.functional.normalize(features, dim=1)


def _one_batch(encoder, batch: torch.Tensor, content_ids: list[str], device: str) -> dict:
    """One synchronized batch. CUDA events are an extra measurement, not the whole cost."""
    _sync(device)
    started = time.perf_counter()
    cuda_ms = None
    if device.startswith("cuda") and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        descriptor = _forward(encoder, batch, content_ids)
        end.record()
        torch.cuda.synchronize()
        cuda_ms = float(start.elapsed_time(end))
    else:
        descriptor = _forward(encoder, batch, content_ids)
        _sync(device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "wall_ms": float(wall_ms),
        "cuda_event_ms": cuda_ms,
        "requires_grad": bool(descriptor.requires_grad),
        "grad_enabled": bool(torch.is_grad_enabled()),
    }


def time_encoder(
    encoder,
    batch: torch.Tensor,
    content_ids: list[str],
    warmup: int,
    iterations: int,
    trials: int,
) -> dict:
    """Time one method. Parameters are frozen and every forward uses inference_mode."""
    freeze_encoder(encoder)
    device = encoder.device
    batch = batch.to(device)
    before = {key: value.detach().clone() for key, value in encoder.model.state_dict().items()}
    batch_walls: list[float] = []
    batch_cuda: list[float] = []
    grad_flags: list[bool] = []
    with torch.inference_mode():
        for _ in range(warmup):
            _one_batch(encoder, batch, content_ids, device)
        for _trial in range(trials):
            for _ in range(iterations):
                sample = _one_batch(encoder, batch, content_ids, device)
                batch_walls.append(sample["wall_ms"])
                grad_flags.append(sample["grad_enabled"] or sample["requires_grad"])
                if sample["cuda_event_ms"] is not None:
                    batch_cuda.append(sample["cuda_event_ms"])
    after = encoder.model.state_dict()
    for key, value in before.items():
        if not torch.equal(value, after[key]):
            raise RuntimeError(f"Benchmark changed parameter or buffer {key}.")
    if any(grad_flags):
        raise RuntimeError("A timed forward tracked gradients.")
    walls = np.array(batch_walls, dtype=np.float64)
    batch_size = int(batch.shape[0])
    per_trial = walls.reshape(trials, iterations).mean(axis=1)
    median = float(np.median(walls))
    peak = None
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            _one_batch(encoder, batch, content_ids, device)
        peak = int(torch.cuda.max_memory_allocated())
    mps_allocated = None
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        with torch.inference_mode():
            _one_batch(encoder, batch, content_ids, device)
        mps_allocated = int(torch.mps.current_allocated_memory())
    return {
        "scope": "encoder_inference_device_resident",
        "scope_note": "Preprocessed batch already on device, through the L2-normalised descriptor. Includes selection. Excludes image loading and preprocessing.",
        "correction": "Timed under torch.inference_mode with frozen parameters. Earlier MPS timings were not.",
        "hardware": hardware_report(),
        "device_name": _device_name(device),
        "packages": package_versions(),
        "precision": "float32",
        "autocast": False,
        "attention_backend_settings": _attention_settings(),
        "batch_size": batch_size,
        "input_shape": list(batch.shape),
        "tokens_per_block": list(encoder.model.image_encoder.last_tokens_per_block),
        "warmup": warmup,
        "iterations": iterations,
        "trials": trials,
        "timing_wall": "perf_counter around a synchronized forward",
        "timing_cuda_event": "cuda_event" if batch_cuda else None,
        "cuda_event_note": "CUDA-event time excludes host work that finishes before the event range is closed only if that work is outside the recorded events. Wall-clock time is the complete measurement for this scope.",
        "median_batch_wall_ms": median,
        "p90_batch_wall_ms": float(np.percentile(walls, 90)),
        "trial_mean_wall_ms": [float(item) for item in per_trial],
        "median_trial_mean_wall_ms": float(np.median(per_trial)),
        "batch_wall_ms": [float(item) for item in walls],
        "median_batch_cuda_event_ms": float(np.median(batch_cuda)) if batch_cuda else None,
        "p90_batch_cuda_event_ms": float(np.percentile(batch_cuda, 90)) if batch_cuda else None,
        "batch_cuda_event_ms": batch_cuda,
        "per_image_wall_ms": median / batch_size,
        "images_per_second": 1000.0 * batch_size / median if median else None,
        "throughput_scope": "device-resident encoder only",
        "peak_cuda_bytes_allocated": peak,
        "mps_current_allocated_bytes": mps_allocated,
        "mps_memory_note": "MPS current allocated bytes are not a peak-memory measurement.",
        "selection": None if encoder.model.selection is None else encoder.model.selection.identity(),
        "grad_enabled_during_timing": False,
    }


def time_methods(encoder, batch: torch.Tensor, content_ids: list[str], methods: list, warmup: int, iterations: int, trials: int, order_seed: int = 553) -> dict:
    """Run each method once per trial. Method order changes across trials and is recorded."""
    generator = np.random.Generator(np.random.PCG64(order_seed))
    order_log = []
    collected: dict[str, list] = {method["name"]: [] for method in methods}
    by_name = {method["name"]: method for method in methods}
    for trial in range(trials):
        names = [method["name"] for method in methods]
        order = np.arange(len(names))
        generator.shuffle(order)
        names = [names[int(index)] for index in order]
        order_log.append(names)
        for name in names:
            method = by_name[name]
            encoder.set_token_pruning(1.0, 6)
            encoder.set_selection(method["selection"])
            row = time_encoder(encoder, batch, content_ids, warmup, iterations, trials=1)
            row["trial"] = trial
            row["name"] = name
            collected[name].append(row)
    return {"order_seed": order_seed, "trial_orders": order_log, "methods": collected}
