"""Check the dataset counts and load the checkpoint."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src")

print("Step 3. Check the dataset and the model")
if not PYTHON.is_file():
    raise SystemExit("The local environment .venv is missing.")
for command in (
    [str(PYTHON), "-m", "reidvlm", "verify-data", "--config", "configs/market1501.yaml"],
    [str(PYTHON), "-m", "reidvlm", "verify-checkpoint", "--config", "configs/market1501.yaml"],
):
    subprocess.check_call(command, cwd=ROOT, env=ENV)
