"""Inference-time patch selection for a frozen CLIP-ReID encoder."""

SELECTION_VERSION = "selection-v1"
STRATEGIES = ("none", "cls_attention", "random")
ATTENTION_BACKENDS = ("reference", "cls_row")
RANDOM_SEEDS = (11, 23, 37, 53, 71)
GRID_BLOCKS = (6, 9)
GRID_KEEP = (96, 64)
PATCHES = 128
VALIDATION_SEED = 553
TIE_MAP = 1e-6
TIE_RANK1 = 1e-6
TIE_LATENCY_MS = 0.05
