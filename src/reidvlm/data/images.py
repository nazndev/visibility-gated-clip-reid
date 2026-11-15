"""RGB loading and content hashes. Paths never enter model prompts."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from reidvlm.runtime import sha256_bytes


class ImageReadError(ValueError):
    pass


def load_rgb(source: str | Path | bytes | Image.Image) -> Image.Image:
    """Apply EXIF orientation and return an RGB image."""
    try:
        if isinstance(source, Image.Image):
            image = source
        elif isinstance(source, bytes):
            image = Image.open(io.BytesIO(source))
        else:
            image = Image.open(source)
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageReadError("Could not read a valid image.") from exc


def image_sha256(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return sha256_bytes(buffer.getvalue())


def clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image | None:
    clipped = clip_box(box, image.width, image.height)
    if clipped is None:
        return None
    return image.crop(clipped)
