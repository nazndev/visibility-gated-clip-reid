"""Resumable judgment cache. Alpha and the compute device are not part of the key."""

from __future__ import annotations

import json
from pathlib import Path

from reidvlm.runtime import sha256_json

# Device is recorded on the judgment but does not decide whether it can be reused.
COMPATIBLE_FIELDS = (
    "backend",
    "model_id",
    "model_revision",
    "processor_revision",
    "dtype",
    "prompt_version",
    "schema_version",
    "max_new_tokens",
    "do_sample",
    "min_pixels",
    "max_pixels",
)


def compatible_identity(identity: dict) -> dict:
    return {key: identity.get(key) for key in COMPATIBLE_FIELDS}


class JudgmentCache:
    def __init__(self, directory: Path, identity: dict):
        self.directory = directory
        self.identity = identity
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index: dict[tuple[str, str, str], Path] | None = None

    def key(self, hash_a: str, hash_b: str, order: str) -> str:
        return sha256_json({**compatible_identity(self.identity), "hash_a": hash_a, "hash_b": hash_b, "order": order})

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def _matches(self, identity: dict) -> bool:
        return compatible_identity(identity) == compatible_identity(self.identity)

    def _read(self, path: Path, hash_a: str, hash_b: str, order: str) -> dict | None:
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        if not self._matches(record.get("identity") or {}):
            return None
        if (record.get("hash_a"), record.get("hash_b"), record.get("order")) != (hash_a, hash_b, order):
            return None
        return record

    def get(self, hash_a: str, hash_b: str, order: str) -> dict | None:
        direct = self._read(self.path_for(self.key(hash_a, hash_b, order)), hash_a, hash_b, order)
        if direct is not None:
            return direct
        if self._index is None:
            self._index = {}
            for path in self.directory.glob("*.json"):
                try:
                    record = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                if not self._matches(record.get("identity") or {}):
                    continue
                ident = (record.get("hash_a"), record.get("hash_b"), record.get("order"))
                if all(isinstance(part, str) for part in ident):
                    self._index[ident] = path
        path = self._index.get((hash_a, hash_b, order))
        if path is None:
            return None
        return self._read(path, hash_a, hash_b, order)

    def put(self, hash_a: str, hash_b: str, order: str, record: dict) -> None:
        payload = {"identity": self.identity, "hash_a": hash_a, "hash_b": hash_b, "order": order, **record}
        path = self.path_for(self.key(hash_a, hash_b, order))
        path.write_text(json.dumps(payload))
        if self._index is not None:
            self._index[(hash_a, hash_b, order)] = path
