"""Demo-gallery indexing. Benchmark evaluation does not use this detector."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from reidvlm.data.images import crop_box
from reidvlm.detection.detector import PersonDetector
from reidvlm.retrieval.engine import GalleryItem


@dataclass
class IndexedPerson:
    anonymous_id: str
    source_id: str
    bbox: tuple[int, int, int, int]
    crop: Image.Image
    display_name: str | None = None


class DemoGallery:
    def __init__(self):
        self.people: list[IndexedPerson] = []

    def __len__(self) -> int:
        return len(self.people)

    def add_crop(self, image: Image.Image, source_id: str, bbox: tuple[int, int, int, int], display_name: str | None = None) -> IndexedPerson:
        person = IndexedPerson(f"demo-{len(self.people):06d}", source_id, bbox, image, display_name)
        self.people.append(person)
        return person

    def add_full_scene(self, image: Image.Image, source_id: str, detector: PersonDetector, display_name: str | None = None) -> list[IndexedPerson]:
        added = []
        for detection in detector.detect(image):
            crop = crop_box(image, detection.box)
            if crop is None:
                continue
            added.append(self.add_crop(crop, source_id, detection.box, display_name))
        return added

    def add_single_person_image(self, image: Image.Image, source_id: str, display_name: str | None = None) -> IndexedPerson:
        """Benchmark crops and already-cropped references bypass the detector."""
        return self.add_crop(image, source_id, (0, 0, image.width, image.height), display_name)

    def as_gallery_items(self) -> list[GalleryItem]:
        return [
            GalleryItem(person.anonymous_id, person.crop, person.source_id, person.bbox)
            for person in self.people
        ]


def index_benchmark_images(images: list[Image.Image]) -> list[GalleryItem]:
    """Single-person benchmark images are indexed directly, without detection."""
    items = []
    for index, image in enumerate(images):
        items.append(
            GalleryItem(
                anonymous_id=f"bench-{index:06d}",
                image=image,
                source_id=f"bench-source-{index:06d}",
                bbox=(0, 0, image.width, image.height),
            )
        )
    return items
