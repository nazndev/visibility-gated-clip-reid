"""Cache identity for new runs.

The execution hash covers code and configuration that change a descriptor.
Documentation files are not in that hash. Git revision and dirty state are
recorded beside it and do not by themselves select a cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from reidvlm.runtime import package_versions, project_root, sha256_file

EXECUTION_FILES = (
    "src/reidvlm/pruning/__init__.py",
    "src/reidvlm/pruning/selection.py",
    "src/reidvlm/pruning/study.py",
    "src/reidvlm/pruning/benchmark.py",
    "src/reidvlm/pruning/identity.py",
    "src/reidvlm/encoders/clip_reid.py",
    "src/reidvlm/evaluation/metrics.py",
    "src/reidvlm/data/market1501.py",
    "src/reidvlm/data/images.py",
    "src/reidvlm/retrieval/cosine.py",
    "src/reidvlm/persistence.py",
    "src/reidvlm/cli.py",
    "configs/market1501.yaml",
    "pyproject.toml",
)

DOCUMENTATION_EXAMPLES = (
    "README.md",
    "paper/main.tex",
    "RESEARCH_ITERATION_2_STATUS.md",
)


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def source_snapshot(root: Path | None = None) -> dict:
    root = root or project_root()
    files = {}
    for relative in EXECUTION_FILES:
        path = root / relative
        files[relative] = file_sha256(path) if path.is_file() else None
    digest = hashlib.sha256()
    for relative in EXECUTION_FILES:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((files[relative] or "missing").encode())
        digest.update(b"\0")
    return {"files": files, "execution_source_sha256": digest.hexdigest()}


def execution_identity(payload: dict, root: Path | None = None) -> dict:
    """Identity that selects a feature cache. Git revision is stored beside this dict, not inside it."""
    snapshot = source_snapshot(root)
    body = {
        "storage": "chunks-v1",
        "execution_source_sha256": snapshot["execution_source_sha256"],
        "packages": package_versions(),
        **payload,
    }
    encoded = _canonical(body)
    body["execution_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def _canonical(payload: dict) -> bytes:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
