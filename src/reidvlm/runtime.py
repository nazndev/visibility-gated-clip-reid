"""Environment records, paths, and content hashes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

PINNED_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "numpy",
    "pillow",
    "pydantic",
    "pyyaml",
    "tqdm",
    "huggingface_hub",
    "gdown",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def source_revision() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return revision + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PINNED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def disk_free_bytes() -> int | None:
    try:
        return int(shutil_disk_free(project_root()))
    except Exception:
        return None


def shutil_disk_free(path: Path) -> int:
    import shutil

    return shutil.disk_usage(path).free


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_report() -> dict:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda,
        }
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "ram_bytes": memory_bytes(),
        "disk_free_bytes": disk_free_bytes(),
        "device": pick_device(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def package_tree_hash() -> str:
    """Hash the package sources so a run can name the code that produced it."""
    digest = hashlib.sha256()
    root = project_root() / "src" / "reidvlm"
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_project_env(root: Path | None = None) -> dict:
    """Load the project dotenv without printing values or replacing a live token.

    ``HF_TOKEN`` already present in the process environment is preserved.
    ``HUGGING_FACE_HUB_TOKEN`` is copied to ``HF_TOKEN`` only when ``HF_TOKEN``
    is empty. Other empty keys in the file fill unset process variables.
    """
    root = root or project_root()
    path = root / ".env"
    token_source = "process_environment" if os.environ.get("HF_TOKEN") else "absent"
    loaded: list[str] = []
    preserved: list[str] = []
    if path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if os.environ.get(key):
                preserved.append(key)
                continue
            os.environ[key] = value
            loaded.append(key)
            if key == "HF_TOKEN" and value:
                token_source = "dotenv"
    if not os.environ.get("HF_TOKEN") and os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
        token_source = "mapped_hugging_face_hub_token"
    return {
        "env_file": path.is_file(),
        "loaded": loaded,
        "preserved": preserved,
        "HF_TOKEN_set": bool(os.environ.get("HF_TOKEN")),
        "token_source": token_source,
    }


def local_model_ready(model_id: str) -> bool:
    """True when the snapshot can be read without a download."""
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_id, local_files_only=True))
    except Exception:
        return False
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def configure_hf_environment(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> dict:
    """Allow a download until the local snapshot is complete, then allow offline use."""
    status = load_project_env()
    ready = local_model_ready(model_id)
    status["local_cache_complete"] = ready
    status["offline_enabled_after_verify"] = False
    if ready and "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"
        status["offline_enabled_after_verify"] = True
    status["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
    return status


def _probe_hub(model_id: str) -> str:
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id)
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 401:
            return "authentication_failed"
        if status_code == 403:
            return "permission_failed"
        name = type(exc).__name__
        if "Offline" in name:
            return "offline_blocks_download"
        if "LocalEntryNotFound" in name:
            return "local_cache_missing"
        return "network_or_other"
    if os.environ.get("HF_TOKEN"):
        return "access_ok"
    return "public_ok_without_token"


def auth_report(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> dict:
    """Check token presence and model access. The token value is not included."""
    status = load_project_env()
    report = {
        "env_file": status["env_file"],
        "HF_TOKEN_set": status["HF_TOKEN_set"],
        "token_source": status["token_source"],
        "model_id": model_id,
    }
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        report["access"] = "not_probed_because_offline"
    else:
        report["access"] = _probe_hub(model_id)
    report["local_cache_complete"] = local_model_ready(model_id)
    report["offline_enabled_after_verify"] = False
    if report["local_cache_complete"] and "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"
        report["offline_enabled_after_verify"] = True
    report["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
    if report["HF_HUB_OFFLINE"] == "1" and not report["local_cache_complete"]:
        report["blocker"] = "offline mode is set and the local model files are incomplete"
    return report


def peak_gpu_bytes() -> int | None:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


def peak_rss_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(usage)
        return int(usage) * 1024
    except Exception:
        return None
