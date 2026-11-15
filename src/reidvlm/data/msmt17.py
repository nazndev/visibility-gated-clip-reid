"""MSMT17 list parser.

The CLIP-ReID training script (Syliz517/CLIP-ReID, ``datasets/msmt17.py``)
concatenates ``list_train.txt`` and ``list_val.txt`` before training. This
loader keeps those lists separate and reports that merge as the checkpoint's
training protocol rather than applying it silently.

Camera ids are read from the third underscore-separated filename field and
stored 0-based, matching that script. V1 and V2 are never combined.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    path: str
    pid: int
    camid: int
    relative_path: str
    exists: bool


def parse_list_line(line: str, image_root: Path) -> Sample:
    text = line.strip()
    if not text:
        raise ValueError("Empty MSMT17 list line")
    relative, pid_text = text.split()
    parts = Path(relative).name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse camera from MSMT17 path: {relative}")
    try:
        cam_raw = int(parts[2])
        pid = int(pid_text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse MSMT17 ids from: {relative}") from exc
    path = image_root / relative
    return Sample(str(path), pid, cam_raw - 1, relative, path.is_file())


def read_list(list_path: Path, image_root: Path) -> list[Sample]:
    if not list_path.is_file():
        raise FileNotFoundError(f"MSMT17 list is missing: {list_path.name}")
    samples = []
    for line in list_path.read_text().splitlines():
        if line.strip():
            samples.append(parse_list_line(line, image_root))
    return samples


@dataclass
class MSMT17:
    root: Path
    declared_version: str
    train: list[Sample]
    validation: list[Sample]
    query: list[Sample]
    gallery: list[Sample]

    @classmethod
    def load(cls, root: Path, version: str) -> "MSMT17":
        version = version.upper()
        if version not in {"V1", "V2"}:
            raise ValueError("MSMT17 version must be declared as V1 or V2.")
        folder_name = root.name.upper()
        if "V1" in folder_name and version != "V1":
            raise ValueError(f"Folder name {root.name} does not match declared version {version}.")
        if "V2" in folder_name and version != "V2":
            raise ValueError(f"Folder name {root.name} does not match declared version {version}.")
        train = read_list(root / "list_train.txt", root / "train")
        validation = read_list(root / "list_val.txt", root / "train")
        query = read_list(root / "list_query.txt", root / "test")
        gallery = read_list(root / "list_gallery.txt", root / "test")
        return cls(root, version, train, validation, query, gallery)

    def report(self) -> dict:
        def ids(samples: list[Sample]) -> set[int]:
            return {sample.pid for sample in samples}

        def missing(samples: list[Sample]) -> int:
            return sum(not sample.exists for sample in samples)

        train_ids, val_ids = ids(self.train), ids(self.validation)
        query_ids, gallery_ids = ids(self.query), ids(self.gallery)
        return {
            "dataset": "MSMT17",
            "declared_version": self.declared_version,
            "list_train_images": len(self.train),
            "list_val_images": len(self.validation),
            "list_query_images": len(self.query),
            "list_gallery_images": len(self.gallery),
            "missing_files": {
                "train": missing(self.train),
                "val": missing(self.validation),
                "query": missing(self.query),
                "gallery": missing(self.gallery),
            },
            "identities": {
                "train": len(train_ids),
                "val": len(val_ids),
                "query": len(query_ids),
                "gallery": len(gallery_ids),
            },
            "train_val_overlap": sorted(train_ids & val_ids),
            "train_query_overlap": sorted((train_ids | val_ids) & query_ids),
            "checkpoint_training_protocol": (
                "Syliz517/CLIP-ReID datasets/msmt17.py merges list_train and list_val "
                "into the training set. This project does not merge them for evaluation."
            ),
            "published_reference_counts": {
                "source": "Wei et al., CVPR 2018, and the CLIP-ReID dataset docstring",
                "identities": 4101,
                "bounding_boxes": 126441,
                "cameras": 15,
                "train_plus_val_images_commonly_reported": 32621,
                "query_images": 11659,
                "gallery_images": 82161,
            },
        }
