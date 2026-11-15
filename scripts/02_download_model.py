"""Download the CLIP-ReID checkpoint and check its hash."""

import hashlib
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "Market1501_clipreid_ViT-B-16_60.pth"
MODEL_ID = "1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7"
EXPECTED = "bedc8fcc37f296045df19d901b162dd032bfa85d9b4f2406eff8313536d444dc"

print("Step 2. Download the search model")
if not CHECKPOINT.is_file():
    print("Downloading CLIP-ReID, about 484 MB.")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=MODEL_ID, output=str(CHECKPOINT))
if not CHECKPOINT.is_file():
    raise SystemExit("The checkpoint download did not produce the .pth file.")
digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
print(digest)
if digest != EXPECTED:
    raise SystemExit("Checkpoint hash does not match the official CLIP-ReID file.")
print("Hash matches.")
