"""Inference-only ViT-B/16 CLIP-ReID encoder.

The visual tower follows OpenAI CLIP (MIT License) as modified by
Syliz517/CLIP-ReID ``model/clip/model.py`` and ``model/make_model_clipreid.py``
at commit eb1898b72c882875f478bebfc6d41644eece0a5d. Training-only text prompts
and identity classifiers are not constructed. Their checkpoint keys are ignored
only after the image encoder and both BN necks load with ``strict=True``.

Test features match the official evaluation setting ``NECK_FEAT=after``:
concatenate the 768-d and 512-d batch-normalised CLS tokens. The official test
code then L2-normalises. With that normalisation, its Euclidean distance is
rank-equivalent to the cosine similarity used here.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from reidvlm.data.images import load_rgb
from reidvlm.retrieval.cosine import l2_normalize
from reidvlm.runtime import pick_device, sha256_file

OMITTED_PREFIXES = ("classifier", "classifier_proj", "prompt_learner", "text_encoder")


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(width * 4, width)),
                ]
            )
        )
        self.ln_2 = LayerNorm(width)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn = self.attn(
                self.ln_1(x),
                self.ln_1(x),
                self.ln_1(x),
                need_weights=True,
                average_attn_weights=True,
            )
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


def prune_patch_tokens(tokens: torch.Tensor, attn: torch.Tensor, keep_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the class token and the patches it attends to most.

    tokens is [sequence, batch, dim] and index 0 is the class token.
    attn is [batch, sequence, sequence], already averaged over heads.
    Kept patches stay in their original left-to-right, top-to-bottom order.
    """
    sequence, _batch, dim = tokens.shape
    patches = sequence - 1
    keep = max(1, int(round(patches * keep_ratio)))
    scores = attn[:, 0, 1:]
    chosen = torch.sort(scores.topk(keep, dim=1).indices, dim=1).values
    gathered = torch.gather(tokens[1:].permute(1, 0, 2), 1, chosen.unsqueeze(-1).expand(-1, -1, dim))
    kept = torch.cat([tokens[0].unsqueeze(1), gathered], dim=1).permute(1, 0, 2)
    return kept, chosen


class VisionTransformer(nn.Module):
    def __init__(self, height: int, width: int, patch: int, stride: int, dim: int = 768, layers: int = 12, heads: int = 12, output_dim: int = 512):
        super().__init__()
        grid_h = (height - patch) // stride + 1
        grid_w = (width - patch) // stride + 1
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.patch = patch
        self.conv1 = nn.Conv2d(3, dim, kernel_size=patch, stride=stride, bias=False)
        scale = dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn(grid_h * grid_w + 1, dim))
        self.ln_pre = LayerNorm(dim)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(*[ResidualAttentionBlock(dim, heads) for _ in range(layers)])
        self.ln_post = LayerNorm(dim)
        self.proj = nn.Parameter(scale * torch.randn(dim, output_dim))
        self.last_kept_patch_index = None
        self.last_tokens_in = grid_h * grid_w + 1
        self.last_tokens_out = grid_h * grid_w + 1

    def forward(self, x: torch.Tensor, camera_embed: torch.Tensor | None = None, keep_ratio: float = 1.0, prune_layer: int = 6):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        if camera_embed is not None:
            x[:, 0] = x[:, 0] + camera_embed
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).permute(1, 0, 2)
        patches = x.shape[0] - 1
        self.last_tokens_in = int(x.shape[0])
        if keep_ratio >= 1.0:
            self.last_kept_patch_index = torch.arange(patches, device=x.device).expand(x.shape[1], patches)
            self.last_tokens_out = int(x.shape[0])
            x11 = self.transformer.resblocks[:11](x)
            x12 = self.transformer.resblocks[11](x11)
            x12 = self.ln_post(x12.permute(1, 0, 2))
            return x11.permute(1, 0, 2), x12, x12 @ self.proj
        if not 0 < keep_ratio < 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")
        if not 1 <= prune_layer <= 11:
            raise ValueError("prune_layer must be from 1 to 11 so the remaining layers run on the shorter sequence.")
        x11 = None
        for layer, block in enumerate(self.transformer.resblocks, start=1):
            if layer == prune_layer:
                x, attn = block(x, return_attn=True)
                x, self.last_kept_patch_index = prune_patch_tokens(x, attn, keep_ratio)
            else:
                x = block(x)
            if layer == 11:
                x11 = x
        self.last_tokens_out = int(x.shape[0])
        x12 = self.ln_post(x.permute(1, 0, 2))
        return x11.permute(1, 0, 2), x12, x12 @ self.proj


class CLIPReIDViT(nn.Module):
    def __init__(self, image_height: int, image_width: int, stride: int, sie_camera: bool, camera_count: int, sie_coefficient: float):
        super().__init__()
        self.image_encoder = VisionTransformer(image_height, image_width, 16, stride)
        self.bottleneck = nn.BatchNorm1d(768)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck_proj = nn.BatchNorm1d(512)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.sie_camera = sie_camera
        self.sie_coefficient = sie_coefficient
        if sie_camera:
            self.cv_embed = nn.Parameter(torch.zeros(camera_count, 768))
        self.keep_ratio = 1.0
        self.prune_layer = 6

    def forward(self, images: torch.Tensor, camera_ids: torch.Tensor | None = None) -> torch.Tensor:
        camera_embed = None
        if self.sie_camera:
            if camera_ids is None:
                camera_embed = torch.zeros(images.shape[0], 768, device=images.device, dtype=images.dtype)
            else:
                camera_embed = self.sie_coefficient * self.cv_embed[camera_ids]
        _, tokens, projected = self.image_encoder(images, camera_embed, self.keep_ratio, self.prune_layer)
        feat = self.bottleneck(tokens[:, 0].float())
        feat_proj = self.bottleneck_proj(projected[:, 0].float())
        return torch.cat([feat, feat_proj], dim=1)


def _strip_prefix(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def load_checkpoint_weights(model: CLIPReIDViT, checkpoint: Path) -> dict:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint does not contain a state dict.")
    state = {_strip_prefix(key): value for key, value in payload.items() if torch.is_tensor(value)}
    omitted = {}
    used = {}
    for key, value in state.items():
        if key.startswith(OMITTED_PREFIXES):
            head = key.split(".", 1)[0]
            omitted[head] = omitted.get(head, 0) + 1
        else:
            used[key] = value
    if not any(key.startswith("image_encoder.") for key in used):
        raise RuntimeError("Checkpoint has no image_encoder weights. Refusing to continue with a randomly initialised backbone.")
    model_keys = set(model.state_dict())
    missing = sorted(key for key in model_keys if key not in used)
    unexpected = sorted(key for key in used if key not in model_keys and key.startswith(("image_encoder.", "bottleneck", "bottleneck_proj", "cv_embed")))
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match the image encoder. Missing: {missing[:12]}. Unexpected: {unexpected[:12]}.")
    unrelated = sorted(key for key in used if key not in model_keys)
    filtered = {key: value for key, value in used.items() if key in model_keys}
    incompatible = model.load_state_dict(filtered, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict load failed: {incompatible}")
    return {
        "checkpoint": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "omitted_training_only_keys": omitted,
        "ignored_unrelated_keys": unrelated,
        "positional_tokens": int(model.image_encoder.positional_embedding.shape[0]),
        "missing": [],
        "unexpected": [],
    }


def preprocess(image: Image.Image, height: int, width: int, mean: list[float], std: list[float]) -> torch.Tensor:
    # PIL bilinear matches the torchvision 0.9 validation resize used by CLIP-ReID,
    # which did not apply the later antialias default.
    resized = image.resize((width, height), Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    for channel, (mu, sigma) in enumerate(zip(mean, std)):
        array[:, :, channel] = (array[:, :, channel] - mu) / sigma
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


@dataclass
class EncoderInfo:
    checkpoint_sha256: str
    positional_tokens: int
    omitted_training_only_keys: dict
    device: str
    dtype: str
    sie_camera: bool
    keep_ratio: float = 1.0
    prune_layer: int = 6
    patch_tokens: int = 128
    kept_patch_tokens: int = 128


class CLIPReIDEncoder:
    def __init__(
        self,
        checkpoint: Path,
        image_height: int = 256,
        image_width: int = 128,
        stride: int = 16,
        pixel_mean: list[float] | None = None,
        pixel_std: list[float] | None = None,
        sie_camera: bool = False,
        sie_coefficient: float = 1.0,
        camera_count: int = 6,
        device: str | None = None,
        keep_ratio: float = 1.0,
        prune_layer: int = 6,
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"CLIP-ReID checkpoint not found at {checkpoint}. "
                "The Market-1501 ViT weights are the official Google Drive file "
                "1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7 from Syliz517/CLIP-ReID."
            )
        self.image_height = image_height
        self.image_width = image_width
        self.pixel_mean = pixel_mean or [0.5, 0.5, 0.5]
        self.pixel_std = pixel_std or [0.5, 0.5, 0.5]
        self.device = device or pick_device()
        self.model = CLIPReIDViT(image_height, image_width, stride, sie_camera, camera_count, sie_coefficient)
        report = load_checkpoint_weights(self.model, checkpoint)
        self.set_token_pruning(keep_ratio, prune_layer)
        self.model.to(self.device)
        self.model.eval()
        patches = report["positional_tokens"] - 1
        kept = patches if keep_ratio >= 1.0 else max(1, int(round(patches * keep_ratio)))
        self.info = EncoderInfo(
            report["sha256"],
            report["positional_tokens"],
            report["omitted_training_only_keys"],
            self.device,
            "float32",
            sie_camera,
            keep_ratio,
            prune_layer,
            patches,
            kept,
        )

    def set_token_pruning(self, keep_ratio: float, prune_layer: int = 6) -> None:
        if not 0 < keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}.")
        if keep_ratio < 1 and not 1 <= prune_layer <= 11:
            raise ValueError("prune_layer must be from 1 to 11.")
        self.model.keep_ratio = float(keep_ratio)
        self.model.prune_layer = int(prune_layer)
        if getattr(self, "info", None) is not None:
            patches = self.info.patch_tokens
            kept = patches if keep_ratio >= 1.0 else max(1, int(round(patches * keep_ratio)))
            self.info = replace(self.info, keep_ratio=float(keep_ratio), prune_layer=int(prune_layer), kept_patch_tokens=kept)

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image], camera_ids: list[int] | None = None, batch_size: int = 32) -> np.ndarray:
        features = []
        cameras = camera_ids or [None] * len(images)
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = torch.stack([preprocess(image, self.image_height, self.image_width, self.pixel_mean, self.pixel_std) for image in batch])
            tensor = tensor.to(self.device)
            cam = cameras[start : start + batch_size]
            cam_tensor = None
            if self.model.sie_camera and all(item is not None for item in cam):
                cam_tensor = torch.tensor(cam, dtype=torch.long, device=self.device)
            features.append(self.model(tensor, cam_tensor).cpu().numpy())
        stacked = np.concatenate(features, axis=0).astype(np.float32)
        return l2_normalize(stacked)

    def embed_paths(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        return self.embed_images([load_rgb(path) for path in paths], batch_size=batch_size)
