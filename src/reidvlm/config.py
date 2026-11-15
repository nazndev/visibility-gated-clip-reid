"""YAML experiment configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from reidvlm.runtime import resolve_path


class DatasetConfig(BaseModel):
    name: str
    root: str
    version: str | None = None


class EncoderConfig(BaseModel):
    checkpoint: str = "checkpoints/Market1501_clipreid_ViT-B-16_60.pth"
    image_height: int = 256
    image_width: int = 128
    stride: int = 16
    pixel_mean: list[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    pixel_std: list[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    neck_feat: str = "after"
    sie_camera: bool = False
    sie_coefficient: float = 1.0
    batch_size: int = 32
    # Official Market-1501 ViT-CLIP-ReID weights from the Syliz517 release.
    gdrive_id: str | None = "1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7"
    code_revision: str = "eb1898b72c882875f478bebfc6d41644eece0a5d"


class LVLMConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    revision: str | None = None
    dtype: str = "float16"
    max_new_tokens: int = 384
    do_sample: bool = False
    max_attempts: int = 2
    # Recorded image-processor limits. None means the checkpoint default.
    min_pixels: int | None = None
    max_pixels: int | None = None


class RetrievalConfig(BaseModel):
    k: int = 5
    alpha: float = 0.05
    min_shared_cues: int = 2
    k1: int = 20
    k2: int = 6
    kreciprocal_lambda: float = 0.3
    run_kreciprocal: bool = True


class LimitConfig(BaseModel):
    max_queries: int | None = None
    max_calls: int | None = None
    max_seconds: float | None = None
    validation_queries: int = 32


class ExperimentConfig(BaseModel):
    seed: int = 553
    split: str = "test"
    dataset: DatasetConfig
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    lvlm: LVLMConfig = Field(default_factory=LVLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    limits: LimitConfig = Field(default_factory=LimitConfig)
    output_dir: str = "outputs"
    cache_dir: str = "caches/judgments"


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(resolve_path(path).read_text())
    return ExperimentConfig.model_validate(raw)
