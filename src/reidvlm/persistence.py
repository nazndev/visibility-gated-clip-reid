"""Verified run files. Completion is a readable checksum, not a saved percentage."""

from __future__ import annotations

import json
import os
import shutil
import signal
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reidvlm.lvlm.cache import compatible_identity
from reidvlm.runtime import sha256_bytes, sha256_file, sha256_json


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class RunStore:
    """Copy a finished artifact into the run directory and publish it only after read-back."""

    def __init__(self, run_dir: Path, flush_items: int = 10, flush_seconds: float = 60.0):
        self.run_dir = Path(run_dir)
        self.flush_items = flush_items
        self.flush_seconds = flush_seconds
        self._pending: list[dict] = []
        self._last_flush = 0.0

    def prepare(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "judgments").mkdir(exist_ok=True)
        (self.run_dir / "features").mkdir(exist_ok=True)
        (self.run_dir / "items").mkdir(exist_ok=True)
        (self.run_dir / "meta").mkdir(exist_ok=True)
        (self.run_dir / ".incoming").mkdir(exist_ok=True)
        token = sha256_bytes(os.urandom(16))
        probe = self.run_dir / "meta" / "write_probe.txt"
        probe.write_text(token)
        if probe.read_text() != token:
            raise RuntimeError("persistent directory failed read-back")
        probe.unlink()
        return {"writable": True, "free_bytes": shutil.disk_usage(self.run_dir).free}

    def acquire(self) -> None:
        self.prepare()
        lock = self.run_dir / "RUN.lock"
        if lock.exists():
            text = lock.read_text().strip()
            pid = int(text.split()[0]) if text.split() else 0
            if pid != os.getpid() and _pid_alive(pid):
                raise RuntimeError(f"another process holds this run: {pid}")
        lock.write_text(f"{os.getpid()}\n")

    def release(self) -> None:
        lock = self.run_dir / "RUN.lock"
        if lock.is_file() and lock.read_text().startswith(f"{os.getpid()}"):
            lock.unlink()

    def _sidecar(self, path: Path) -> Path:
        return path.with_name(path.name + ".sha256")

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.run_dir).as_posix()

    def is_verified(self, relative: str) -> bool:
        dest = self.run_dir / relative
        sidecar = self._sidecar(dest)
        if not dest.is_file() or not sidecar.is_file():
            return False
        recorded = sidecar.read_text().strip().split()[0]
        return sha256_file(dest) == recorded

    def _quarantine(self, path: Path) -> None:
        if not path.exists():
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path.rename(path.with_name(path.name + f".corrupt-{stamp}"))

    def _note_verified(self, relative: str, digest: str, reused: bool) -> None:
        payload = {
            "relative": relative,
            "sha256": digest,
            "reused": reused,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        raw = json.dumps(payload, sort_keys=True).encode()
        dest = self.run_dir / "meta" / "last_verified.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        self._sidecar(dest).write_text(sha256_bytes(raw) + "\n")

    def publish_bytes(self, relative: str, data: bytes, replace: bool = False) -> dict:
        dest = self.run_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_bytes(data)
        sidecar = self._sidecar(dest)
        if dest.is_file() and sidecar.is_file():
            recorded = sidecar.read_text().strip().split()[0]
            actual = sha256_file(dest)
            if actual == recorded == digest:
                self._note_verified(relative, digest, True)
                return {"relative": relative, "sha256": digest, "reused": True}
            if actual == recorded and actual != digest:
                if not replace:
                    raise RuntimeError(f"refusing to replace a verified artifact: {relative}")
                self._quarantine(dest)
                if sidecar.exists():
                    self._quarantine(sidecar)
            self._quarantine(dest)
            if sidecar.exists():
                self._quarantine(sidecar)
        incoming = self.run_dir / ".incoming" / f"{dest.name}.{os.getpid()}.tmp"
        incoming.parent.mkdir(parents=True, exist_ok=True)
        incoming.write_bytes(data)
        if sha256_file(incoming) != digest:
            incoming.unlink(missing_ok=True)
            raise RuntimeError("local checksum failed before the persistent copy")
        partial = dest.with_name(dest.name + ".partial")
        shutil.copy2(incoming, partial)
        if sha256_file(partial) != digest:
            partial.unlink(missing_ok=True)
            raise RuntimeError("persistent copy checksum failed")
        os.replace(partial, dest)
        incoming.unlink(missing_ok=True)
        if sha256_file(dest) != digest:
            raise RuntimeError("read-back checksum failed")
        sidecar.write_text(digest + "\n")
        if sha256_file(dest) != digest:
            raise RuntimeError("artifact changed after its checksum was recorded")
        self._note_verified(relative, digest, False)
        return {"relative": relative, "sha256": digest, "reused": False}

    def publish_json(self, relative: str, payload: dict, replace: bool = False) -> dict:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return self.publish_bytes(relative, data, replace=replace)

    def read_verified_json(self, relative: str) -> dict | None:
        if not self.is_verified(relative):
            return None
        return json.loads((self.run_dir / relative).read_text())

    def judgment_relative(self, hash_a: str, hash_b: str, order: str) -> str:
        digest = sha256_json({"hash_a": hash_a, "hash_b": hash_b, "order": order})
        return f"judgments/{digest}.json"

    def read_judgment(self, hash_a: str, hash_b: str, order: str) -> dict | None:
        return self.read_verified_json(self.judgment_relative(hash_a, hash_b, order))

    def remember_judgment(self, payload: dict) -> None:
        import time

        self._pending.append(payload)
        if self._last_flush == 0.0:
            self._last_flush = time.monotonic()
        due = (time.monotonic() - self._last_flush) >= self.flush_seconds
        if len(self._pending) >= self.flush_items or due:
            self.flush()

    def flush(self) -> int:
        import time

        wrote = 0
        pending = self._pending
        self._pending = []
        for payload in pending:
            relative = self.judgment_relative(payload["hash_a"], payload["hash_b"], payload["order"])
            self.publish_json(relative, payload)
            wrote += 1
        self._last_flush = time.monotonic()
        return wrote

    def import_judgments(self, source: Path) -> dict:
        """Copy compatible judgment files. Leave the source directory unchanged."""
        imported = reused = incompatible = invalid = 0
        groups: dict[str, int] = {}
        files = sorted(Path(source).glob("*.json"))
        for path in files:
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                invalid += 1
                continue
            identity = record.get("identity") or {}
            key = sha256_json(compatible_identity(identity))
            groups[key] = groups.get(key, 0) + 1
        if not groups:
            return {"imported": 0, "reused": 0, "incompatible": 0, "invalid": invalid, "source_files": 0}
        majority = max(groups, key=groups.get)
        for path in files:
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if sha256_json(compatible_identity(record.get("identity") or {})) != majority:
                incompatible += 1
                continue
            if not all(isinstance(record.get(name), str) for name in ("hash_a", "hash_b", "order")):
                invalid += 1
                continue
            relative = self.judgment_relative(record["hash_a"], record["hash_b"], record["order"])
            result = self.publish_json(relative, record)
            if result["reused"]:
                reused += 1
            else:
                imported += 1
        return {
            "imported": imported,
            "reused": reused,
            "incompatible": incompatible,
            "invalid": invalid,
            "source_files": len(files),
        }

    def status(self) -> dict:
        verified_judgments = []
        invalid = []
        preserved_corrupt = []
        feature_images = 0
        if self.run_dir.is_dir():
            for path in self.run_dir.rglob("*"):
                if not path.is_file():
                    continue
                if ".incoming" in path.parts or path.name.endswith(".sha256"):
                    continue
                relative = self._relative(path)
                if ".corrupt-" in path.name or path.name.endswith(".partial"):
                    preserved_corrupt.append(relative)
                    continue
                if path.name == "RUN.lock":
                    continue
                if self.is_verified(relative):
                    if relative.startswith("judgments/") and relative.endswith(".json"):
                        verified_judgments.append(relative)
                    if relative.startswith("features/") and relative.endswith(".json"):
                        meta = self.read_verified_json(relative)
                        npy = relative[:-5] + ".npy"
                        if meta and self.is_verified(npy):
                            feature_images += int(meta.get("count") or 0)
                elif path.suffix in {".json", ".npy", ".txt"}:
                    invalid.append(relative)
        expected_ids = []
        expected_file = None
        meta = self.run_dir / "meta"
        if meta.is_dir():
            candidates = []
            for path in meta.glob("expected_*.json"):
                relative = self._relative(path)
                if self.is_verified(relative):
                    candidates.append(path)
            if candidates:
                latest = max(candidates, key=lambda item: item.stat().st_mtime_ns)
                expected_file = self._relative(latest)
                payload = self.read_verified_json(expected_file) or {}
                expected_ids = payload.get("ids") or []
        done = set()
        for relative in verified_judgments:
            record = self.read_verified_json(relative) or {}
            done.add((record.get("hash_a"), record.get("hash_b"), record.get("order")))
        remaining = [
            item
            for item in expected_ids
            if (item.get("hash_a"), item.get("hash_b"), item.get("order")) not in done
        ]
        lock_pid = None
        lock = self.run_dir / "RUN.lock"
        if lock.is_file():
            text = lock.read_text().strip().split()
            lock_pid = int(text[0]) if text else None
        last = self.read_verified_json("meta/last_verified.json")
        return {
            "completed_judgments": len(verified_judgments),
            "expected_judgments": len(expected_ids) if expected_file else None,
            "remaining_judgments": len(remaining) if expected_file else None,
            "expected_file": expected_file,
            "verified_feature_images": feature_images,
            "invalid": invalid,
            "preserved_corrupt": preserved_corrupt,
            "last_verified": last,
            "lock_pid": lock_pid,
            "lock_alive": bool(lock_pid and _pid_alive(lock_pid)),
            "free_bytes": shutil.disk_usage(self.run_dir).free if self.run_dir.is_dir() else None,
        }


    def feature_key(self, identity: dict) -> str:
        return sha256_json(identity)[:20]

    def load_feature_table(self, identity: dict) -> dict[str, np.ndarray] | None:
        """Return id -> vector if the sidecar matches. A mismatched identity is refused."""
        key = self.feature_key(identity)
        relative = f"features/{key}.npz"
        dest = self.run_dir / relative
        if not self.is_verified(relative):
            if dest.exists():
                self._quarantine(dest)
            return None
        meta_rel = f"features/{key}.json"
        if not self.is_verified(meta_rel):
            return None
        meta = json.loads((self.run_dir / meta_rel).read_text())
        if meta.get("identity") != identity:
            raise ValueError("Feature cache identity does not match the requested run.")
        payload = np.load(dest)
        ids = [str(item) for item in payload["ids"].tolist()]
        rows = np.asarray(payload["rows"], dtype=np.float32)
        if len(ids) != len(rows):
            raise ValueError("Feature cache row count does not match its identifiers.")
        return {item: row for item, row in zip(ids, rows)}

    def save_feature_table(self, identity: dict, table: dict[str, np.ndarray]) -> dict:
        key = self.feature_key(identity)
        ids = sorted(table)
        rows = np.stack([np.asarray(table[item], dtype=np.float32) for item in ids])
        incoming = self.run_dir / ".incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        array_tmp = incoming / f"{key}.npz"
        np.savez(array_tmp, ids=np.array(ids), rows=rows)
        array_bytes = array_tmp.read_bytes()
        array_tmp.unlink()
        meta = {"identity": identity, "count": len(ids)}
        published = self.publish_bytes(f"features/{key}.npz", array_bytes, replace=True)
        self.publish_json(f"features/{key}.json", meta, replace=True)
        return published


def demo_worker(run_dir: str, total: int, die_after: int | None = None) -> None:
    """Write numbered items. Used by the interruption test, not by the benchmark."""
    store = RunStore(Path(run_dir))
    store.prepare()
    store.acquire()
    try:
        for index in range(total):
            item_id = f"item-{index:04d}"
            relative = f"items/{item_id}.json"
            if store.is_verified(relative):
                continue
            store.publish_json(relative, {"id": item_id, "value": index * 3})
            verified = sum(1 for item in range(total) if store.is_verified(f"items/item-{item:04d}.json"))
            if die_after is not None and verified >= die_after:
                os.kill(os.getpid(), signal.SIGTERM)
    finally:
        store.release()
