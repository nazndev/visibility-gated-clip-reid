"""Download the frozen Qwen judge. This does not run judgments."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "caches" / "huggingface")
sys.path.insert(0, str(ROOT / "src"))

from reidvlm.runtime import load_project_env, local_model_ready

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"

print("Step 4. Download the Qwen judge")
status = load_project_env(ROOT)
print("token set:", status["HF_TOKEN_set"])
if local_model_ready(MODEL_ID):
    print("Qwen is already on disk.")
else:
    print("Downloading Qwen2.5-VL-3B-Instruct. This is several GB.")
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, revision=REVISION)
    print(path)
if not local_model_ready(MODEL_ID):
    raise SystemExit("The Qwen download is incomplete.")
print("Qwen files are complete. No judgments were run.")
