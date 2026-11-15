"""Person detector used only as an application input stage.

Weights: torchvision FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1.
The transform converts RGB to a float tensor in [0, 1]. The model resizes
internally and returns boxes in the original image coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from reidvlm.data.images import clip_box, crop_box, image_sha256
from reidvlm.runtime import pick_device

WEIGHTS_URL = "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Detection:
    score: float
    box: tuple[int, int, int, int]
    label: str


class PersonDetector:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, device: str | None = None, cache_dir: Path | None = None):
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        self.threshold = threshold
        self.weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.categories = list(self.weights.meta["categories"])
        if "person" not in self.categories:
            raise RuntimeError("The installed detection weights do not list a person category.")
        self.person_label = self.categories.index("person")
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.weights, box_score_thresh=threshold)
        self.model.eval()
        self.device = device or pick_device()
        self.model.to(self.device)
        self.preprocess = self.weights.transforms()
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        return {
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "weights": "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1",
            "weights_url": WEIGHTS_URL,
            "person_label": self.person_label,
            "threshold": self.threshold,
            "preprocess": "torchvision ObjectDetection: RGB float tensor in [0, 1], no extra resize",
            "torchvision_file_size_mb": self.weights.meta.get("_file_size"),
        }

    def _cache_path(self, image: Image.Image) -> Path | None:
        if self.cache_dir is None:
            return None
        identity = image_sha256(image) + f"-{self.weights.name}-{self.threshold}"
        return self.cache_dir / f"{identity}.json"

    def detect(self, image: Image.Image) -> list[Detection]:
        cached = self._cache_path(image)
        if cached is not None and cached.is_file():
            payload = json.loads(cached.read_text())
            return [Detection(float(item["score"]), tuple(item["box"]), item["label"]) for item in payload]
        tensor = self.preprocess(image).to(self.device)
        with torch.inference_mode():
            prediction = self.model([tensor])[0]
        detections = []
        boxes = prediction["boxes"].detach().cpu().tolist()
        labels = prediction["labels"].detach().cpu().tolist()
        scores = prediction["scores"].detach().cpu().tolist()
        for box, label, score in zip(boxes, labels, scores):
            if int(label) != self.person_label or float(score) < self.threshold:
                continue
            clipped = clip_box(tuple(box), image.width, image.height)
            if clipped is None:
                continue
            detections.append(Detection(float(score), clipped, "person"))
        detections.sort(key=lambda item: item.score, reverse=True)
        if cached is not None:
            cached.write_text(
                json.dumps(
                    [
                        {"score": item.score, "box": list(item.box), "label": item.label}
                        for item in detections
                    ]
                )
            )
        return detections


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for number, detection in enumerate(detections, start=1):
        draw.rectangle(detection.box, outline=(220, 40, 40), width=3)
        draw.text((detection.box[0] + 2, detection.box[1] + 2), str(number), fill=(220, 40, 40))
    return canvas


def select_detection(image: Image.Image, detections: list[Detection], number: int) -> Image.Image:
    if number < 1 or number > len(detections):
        raise ValueError("Select one of the numbered people. No person was chosen automatically.")
    crop = crop_box(image, detections[number - 1].box)
    if crop is None:
        raise ValueError("The selected box is empty after clipping to the image.")
    return crop
