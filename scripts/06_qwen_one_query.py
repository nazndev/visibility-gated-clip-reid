"""Ask Qwen to judge one query against five gallery photos.

This is a demonstration. It does not produce a Market-1501 mAP or Rank-1.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "caches" / "huggingface")
sys.path.insert(0, str(ROOT / "src"))

from reidvlm.data.images import load_rgb
from reidvlm.detection.gallery import DemoGallery
from reidvlm.encoders.clip_reid import CLIPReIDEncoder
from reidvlm.lvlm.cache import JudgmentCache
from reidvlm.lvlm.qwen import QwenVLBackend
from reidvlm.retrieval.engine import SearchEngine

REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DATASET = ROOT / "data" / "Market-1501-v15.09.15"
QUERY = DATASET / "query" / "0001_c1s1_001051_00.jpg"
GALLERY = (
    DATASET / "bounding_box_test" / "0001_c2s1_001976_01.jpg",
    DATASET / "bounding_box_test" / "0001_c3s1_000551_03.jpg",
    DATASET / "bounding_box_test" / "0001_c6s3_057292_01.jpg",
    DATASET / "bounding_box_test" / "0004_c2s3_059152_02.jpg",
    DATASET / "bounding_box_test" / "0001_c1s1_001051_03.jpg",
)

print("Step 6. Qwen judges one query. This is not a Market-1501 score.")
print("Query: person 0001, camera 1.")
print("Gallery: three photos of person 0001 from other cameras,")
print("one photo of person 0001 from the same camera, and one photo of person 0004.")
missing = [path for path in (QUERY, *GALLERY) if not path.is_file()]
if missing:
    raise SystemExit("Missing photo: " + ", ".join(str(path) for path in missing))

gallery = DemoGallery()
for path in GALLERY:
    gallery.add_single_person_image(load_rgb(path), source_id=path.name)

encoder = CLIPReIDEncoder(ROOT / "checkpoints" / "Market1501_clipreid_ViT-B-16_60.pth")
backend = QwenVLBackend(revision=REVISION)
engine = SearchEngine(
    gallery.as_gallery_items(),
    encoder=encoder,
    lvlm=backend,
    cache=JudgmentCache(ROOT / "caches" / "judgments", backend.identity()),
    k=5,
    alpha=0.05,
    min_shared_cues=2,
)
result = engine.search(load_rgb(QUERY), mode="proposed")
print(f"Qwen calls: {result.lvlm_calls}. Gate kept {result.gate_accepted_pairs} of {result.judged_pairs} pairs.")
for hit in result.hits:
    print(
        f"rank {hit.revised_rank} was {hit.original_rank}: {hit.source_id} "
        f"gate={hit.gate_accepted} score={hit.corrected_score:.4f}"
    )
    print(f"  {hit.evidence}")
