"""Patch selection for a frozen CLIP-ReID vision transformer.

Selection happens after a complete transformer block. The class token is kept.
Retained patches are gathered into a shorter sequence and restored to spatial order.
Positional embeddings are applied by the encoder once, before the transformer.
This is an inference-time adaptation of the class-attention score used by EViT.
It does not fuse discarded tokens and it does not train a pruner.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reidvlm.pruning import ATTENTION_BACKENDS, SELECTION_VERSION, STRATEGIES


@dataclass(frozen=True)
class SelectionConfig:
    strategy: str = "none"
    prune_block: int = 6
    keep_patch_count: int = 128
    seed: int | None = None
    attention_backend: str = "reference"
    patch_count: int = 128

    def validate(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")
        if self.attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")
        if self.patch_count < 1:
            raise ValueError("patch_count must be positive.")
        if not 1 <= self.keep_patch_count <= self.patch_count:
            raise ValueError(
                f"keep_patch_count must be from 1 to {self.patch_count}, got {self.keep_patch_count}."
            )
        if self.strategy == "none" and self.keep_patch_count != self.patch_count:
            raise ValueError("strategy none keeps every patch.")
        if self.strategy != "none" and not 1 <= self.prune_block <= 11:
            raise ValueError("prune_block must be from 1 to 11 so later blocks still run.")
        if self.strategy == "random" and self.seed is None:
            raise ValueError("random selection requires an integer seed.")

    def identity(self) -> dict:
        self.validate()
        return {
            "selection_version": SELECTION_VERSION,
            "strategy": self.strategy,
            "prune_block": None if self.strategy == "none" else self.prune_block,
            "keep_patch_count": self.keep_patch_count,
            "patch_count": self.patch_count,
            "seed": self.seed if self.strategy == "random" else None,
            "attention_backend": self.attention_backend if self.strategy == "cls_attention" else None,
        }


def same_selection(left: SelectionConfig, right: SelectionConfig) -> None:
    if left.identity() != right.identity():
        raise ValueError("Query and gallery selection configurations differ.")


def content_id_from_tensor(image: torch.Tensor) -> str:
    """SHA-256 of one preprocessed CHW tensor. Independent of batch position."""
    array = image.detach().to(dtype=torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def random_patch_indices(seed: int, content_id: str, patch_count: int, keep: int) -> list[int]:
    """Uniform sample without replacement, then spatial order.

    The generator is seeded from SHA-256(selection version, seed, content id).
    Python's hash() is not used. The same image and seed always yield the same subset.
    """
    material = f"{SELECTION_VERSION}|{seed}|{content_id}".encode()
    digest = hashlib.sha256(material).digest()
    sequence = np.random.SeedSequence(int.from_bytes(digest[:16], "little"))
    generator = np.random.Generator(np.random.PCG64(sequence))
    chosen = generator.choice(patch_count, size=keep, replace=False)
    return sorted(int(index) for index in chosen.tolist())


def select_from_scores(scores: torch.Tensor, keep: int) -> torch.Tensor:
    """Highest scores first. Ties keep the smaller spatial index, then sort spatially.

    scores is [batch, patches] in spatial order. Stable argsort preserves that order
    for equal scores, so the lower index wins a tie.
    """
    if keep > scores.shape[1]:
        raise ValueError("keep exceeds the number of patch scores.")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    chosen = order[:, :keep]
    return torch.sort(chosen, dim=1).values


def gather_kept(tokens: torch.Tensor, patch_index: torch.Tensor) -> torch.Tensor:
    """tokens is [sequence, batch, dim], index 0 is CLS. patch_index is [batch, keep], spatial order."""
    _sequence, _batch, dim = tokens.shape
    gathered = torch.gather(tokens[1:].permute(1, 0, 2), 1, patch_index.unsqueeze(-1).expand(-1, -1, dim))
    return torch.cat([tokens[0].unsqueeze(1), gathered], dim=1).permute(1, 0, 2)


def cls_patch_scores(attention: nn.MultiheadAttention, normed: torch.Tensor) -> torch.Tensor:
    """CLS-query attention over every key, then drop the CLS column.

    normed is [sequence, batch, width], the block's LayerNorm output.
    Softmax runs over all keys, including CLS. Head probabilities are averaged
    after softmax. Logits are not averaged first, and the distribution is not
    renormalised over patches.
    """
    if attention.in_proj_weight is None:
        raise RuntimeError("This block does not use a combined QKV projection.")
    width = attention.embed_dim
    heads = attention.num_heads
    if width % heads != 0:
        raise RuntimeError("Attention width is not divisible by the head count.")
    head_dim = width // heads
    qkv = F.linear(normed, attention.in_proj_weight, attention.in_proj_bias)
    query, key, _value = qkv.chunk(3, dim=-1)
    length, batch, _ = query.shape
    query = query.view(length, batch, heads, head_dim).permute(1, 2, 0, 3)
    key = key.view(length, batch, heads, head_dim).permute(1, 2, 0, 3)
    logits = torch.matmul(query[:, :, :1, :], key.transpose(-2, -1)) * (head_dim ** -0.5)
    probabilities = torch.softmax(logits, dim=-1)
    averaged = probabilities.mean(dim=1).squeeze(1)
    return averaged[:, 1:]


def reference_patch_scores(weights: torch.Tensor) -> torch.Tensor:
    """weights is [batch, sequence, sequence] from MultiheadAttention, heads already averaged."""
    return weights[:, 0, 1:]


def config_dict(config: SelectionConfig) -> dict:
    payload = asdict(config)
    payload["selection_version"] = SELECTION_VERSION
    return payload
