"""Market-1501 filename parser and protocol counts.

Filenames follow ``{pid}_c{cam}s{seq}_{frame}_{box}.jpg``. Camera ids in the
filename are 1-based and are stored 0-based. ``pid == -1`` marks junk.
``pid == 0`` marks a distractor, which stays in the gallery as a negative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

FILENAME = re.compile(r"([-\d]+)_c(\d)")

EXPECTED_TRAIN_IMAGES = 12936
EXPECTED_QUERY_IMAGES = 3368
EXPECTED_GALLERY_FOLDER_IMAGES = 19732
EXPECTED_GALLERY_AFTER_JUNK = 15913


@dataclass(frozen=True)
class Sample:
    path: str
    pid: int
    camid: int
    junk: bool


def parse_market_name(path: Path) -> tuple[int, int]:
    match = FILENAME.search(path.name)
    if match is None:
        raise ValueError(f"Unrecognised Market-1501 filename: {path.name}")
    pid, cam = int(match.group(1)), int(match.group(2))
    if not 1 <= cam <= 6:
        raise ValueError(f"Camera id out of range in {path.name}")
    return pid, cam - 1


def read_split(directory: Path) -> list[Sample]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Market-1501 split directory is missing: {directory}")
    samples = []
    for path in sorted(directory.glob("*.jpg")):
        pid, camid = parse_market_name(path)
        samples.append(Sample(str(path), pid, camid, junk=pid < 0))
    return samples


def protocol_samples(samples: list[Sample]) -> list[Sample]:
    """Drop junk. Same-identity same-camera removal is done per query by the evaluator."""
    return [sample for sample in samples if not sample.junk]


def identity_sets(samples: list[Sample]) -> set[int]:
    return {sample.pid for sample in samples if sample.pid > 0}


@dataclass
class Market1501:
    root: Path
    train: list[Sample]
    query: list[Sample]
    gallery_folder: list[Sample]
    gallery: list[Sample]

    @classmethod
    def load(cls, root: Path) -> "Market1501":
        root = root / "Market-1501-v15.09.15" if (root / "Market-1501-v15.09.15").is_dir() else root
        train = read_split(root / "bounding_box_train")
        query = read_split(root / "query")
        gallery_folder = read_split(root / "bounding_box_test")
        return cls(root, train, query, gallery_folder, protocol_samples(gallery_folder))

    def report(self) -> dict:
        junk = sum(sample.junk for sample in self.gallery_folder)
        train_ids = identity_sets(self.train)
        query_ids = identity_sets(self.query)
        gallery_ids = identity_sets(self.gallery)
        return {
            "dataset": "Market-1501",
            "protocol": "single-query, no 500K distractors",
            "root_name": self.root.name,
            "train_images": len(self.train),
            "query_images": len(self.query),
            "gallery_folder_images": len(self.gallery_folder),
            "junk_images": junk,
            "gallery_after_junk_removal": len(self.gallery),
            "train_identities": len(train_ids),
            "query_identities": len(query_ids),
            "gallery_identities_excluding_distractor": len(gallery_ids),
            "train_test_identity_overlap": sorted(train_ids & (query_ids | gallery_ids)),
            "expected": {
                "train_images": EXPECTED_TRAIN_IMAGES,
                "query_images": EXPECTED_QUERY_IMAGES,
                "gallery_folder_images": EXPECTED_GALLERY_FOLDER_IMAGES,
                "gallery_after_junk_removal": EXPECTED_GALLERY_AFTER_JUNK,
            },
            "count_mismatch": {
                "train": len(self.train) != EXPECTED_TRAIN_IMAGES,
                "query": len(self.query) != EXPECTED_QUERY_IMAGES,
                "gallery_folder": len(self.gallery_folder) != EXPECTED_GALLERY_FOLDER_IMAGES,
                "gallery_after_junk": len(self.gallery) != EXPECTED_GALLERY_AFTER_JUNK,
            },
        }
