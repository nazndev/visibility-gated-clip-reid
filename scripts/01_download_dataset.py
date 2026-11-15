"""Download and extract Market-1501."""

import zipfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Market-1501-v15.09.15"
ARCHIVE = ROOT / "data" / "market1501.zip"
DATASET_ID = "0B8-rUzbwVRk0c054eEozWG9COHM"

print("Step 1. Download the dataset")
if (DATA / "bounding_box_train").is_dir():
    print("Market-1501 is already extracted.")
else:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.is_file():
        print("Downloading Market-1501, about 146 MB.")
        gdown.download(id=DATASET_ID, output=str(ARCHIVE))
    if not ARCHIVE.is_file():
        raise SystemExit("The dataset download did not produce market1501.zip.")
    print("Extracting.")
    with zipfile.ZipFile(ARCHIVE) as package:
        package.extractall(ROOT / "data")
if not (DATA / "bounding_box_train").is_dir():
    raise SystemExit(f"Market-1501 did not extract to {DATA}")
print(DATA)
