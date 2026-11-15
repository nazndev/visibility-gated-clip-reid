"""Training-derived validation queries.

The official encoder has already seen these identities. The split is only for
calibrating frozen post-processing. Test identities are not used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reidvlm.data.market1501 import Sample


def build_validation(
    train_samples: list[Sample],
    seed: int,
    max_queries: int,
) -> dict:
    grouped: dict[int, list[Sample]] = {}
    for sample in train_samples:
        if sample.junk or sample.pid <= 0:
            continue
        grouped.setdefault(sample.pid, []).append(sample)
    eligible = []
    for pid, samples in grouped.items():
        cameras = {sample.camid for sample in samples}
        if len(cameras) >= 2 and len(samples) >= 2:
            eligible.append(pid)
    eligible.sort()
    rng = np.random.RandomState(seed)
    order = list(eligible)
    rng.shuffle(order)
    chosen = sorted(order[:max_queries])
    queries = []
    gallery = []
    chosen_set = set(chosen)
    for pid in chosen:
        samples = sorted(grouped[pid], key=lambda sample: (sample.camid, Path(sample.path).name))
        query = samples[0]
        queries.append(
            {
                "path": query.path,
                "pid": query.pid,
                "camid": query.camid,
                "name": Path(query.path).name,
            }
        )
    # Gallery contains every image of the chosen identities except the query
    # image itself. Same-camera matches remain here and are removed only by
    # the evaluator.
    query_paths = {item["path"] for item in queries}
    for pid in chosen:
        for sample in sorted(grouped[pid], key=lambda item: item.path):
            if sample.path in query_paths:
                continue
            gallery.append(
                {
                    "path": sample.path,
                    "pid": sample.pid,
                    "camid": sample.camid,
                    "name": Path(sample.path).name,
                }
            )
    manifest = {
        "source": "official training images only",
        "purpose": "calibrate frozen post-processing; not an unseen-identity encoder evaluation",
        "seed": seed,
        "eligible_identities": len(eligible),
        "selected_identities": chosen,
        "query_count": len(queries),
        "gallery_count": len(gallery),
        "encoder_saw_these_identities": True,
        "queries": queries,
        "gallery": gallery,
        "selected_identity_count": len(chosen_set),
    }
    return manifest


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())
