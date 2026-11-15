"""Rank every Market-1501 query. This does not call Qwen."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reidvlm.config import load_config
from reidvlm.data.market1501 import Market1501
from reidvlm.runtime import resolve_path

PYTHON = ROOT / ".venv" / "bin" / "python"
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src")

config = load_config("configs/market1501.yaml")
query_count = len(Market1501.load(resolve_path(config.dataset.root)).query)
print(f"Step 5. Visual ranking of all {query_count} queries. Qwen is not called.")
if not PYTHON.is_file():
    raise SystemExit("The local environment .venv is missing.")
subprocess.check_call(
    [str(PYTHON), "-m", "reidvlm", "evaluate", "--config", "configs/market1501.yaml", "--visual-only"],
    cwd=ROOT,
    env=ENV,
)
