"""Encoder timing. Cached features are not inference.

CUDA events are used when CUDA is available. Wall-clock time is always recorded
because random index generation and attention-score work can sit on the CPU.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from reidvlm.runtime import hardware_report, package_versions


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return device


def time_encoder(encoder, batch: torch.Tensor, content_ids: list[str], warmup: int, iterations: int, trials: int) -> dict:
    """Time device-resident inputs through the final descriptor. Includes selection work."""
    device = encoder.device
    batch = batch.to(device)
    for _ in range(warmup):
        _sync()
        torch.nn.functional.normalize(encoder.model(batch, None, content_ids), dim=1)
        _sync()
    trial_ms = []
    for _trial in range(trials):
        _sync()
        if device.startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                torch.nn.functional.normalize(encoder.model(batch, None, content_ids), dim=1)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / iterations
        else:
            _sync()
            started = time.perf_counter()
            for _ in range(iterations):
                torch.nn.functional.normalize(encoder.model(batch, None, content_ids), dim=1)
            _sync()
            elapsed = (time.perf_counter() - started) * 1000.0 / iterations
        trial_ms.append(float(elapsed))
    values = np.array(trial_ms, dtype=np.float64)
    batch_size = int(batch.shape[0])
    median = float(np.median(values))
    peak = None
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        _sync()
        torch.nn.functional.normalize(encoder.model(batch, None, content_ids), dim=1)
        _sync()
        peak = int(torch.cuda.max_memory_allocated())
    mps_allocated = None
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        _sync()
        torch.nn.functional.normalize(encoder.model(batch, None, content_ids), dim=1)
        _sync()
        mps_allocated = int(torch.mps.current_allocated_memory())
    return {
        "scope": "encoder_inference",
        "hardware": hardware_report(),
        "device_name": _device_name(device),
        "packages": package_versions(),
        "precision": "float32",
        "autocast": False,
        "batch_size": batch_size,
        "input_shape": list(batch.shape),
        "tokens_per_block": list(encoder.model.image_encoder.last_tokens_per_block),
        "warmup": warmup,
        "iterations": iterations,
        "trials": trials,
        "timing": "cuda_event" if device.startswith("cuda") else "wall_clock_after_device_sync",
        "median_batch_ms": median,
        "p90_batch_ms": float(np.percentile(values, 90)),
        "per_image_ms": median / batch_size,
        "images_per_second": 1000.0 * batch_size / median if median else None,
        "trial_ms": trial_ms,
        "peak_gpu_bytes": peak,
        "mps_current_allocated_bytes": mps_allocated,
        "selection": None if encoder.model.selection is None else encoder.model.selection.identity(),
    }
