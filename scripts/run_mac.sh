#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo "The local environment .venv is missing."
  exit 1
fi
.venv/bin/python scripts/01_download_dataset.py
.venv/bin/python scripts/02_download_model.py
.venv/bin/python scripts/03_check.py
.venv/bin/python scripts/04_download_qwen.py
.venv/bin/python scripts/05_visual_rank.py
echo "Mac run finished. Dataset, CLIP-ReID, and Qwen are on this laptop. The full Qwen test was not run."
