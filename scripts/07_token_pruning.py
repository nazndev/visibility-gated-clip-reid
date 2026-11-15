"""Show token pruning on one Market-1501 query.

The vision transformer drops patch tokens after layer 6 and still ranks the
same five gallery photos. This is a demonstration, not a Market-1501 score.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reidvlm.data.images import load_rgb
from reidvlm.encoders.clip_reid import CLIPReIDEncoder

DATASET = ROOT / "data" / "Market-1501-v15.09.15"
QUERY = DATASET / "query" / "0001_c1s1_001051_00.jpg"
GALLERY = (
    DATASET / "bounding_box_test" / "0001_c2s1_001976_01.jpg",
    DATASET / "bounding_box_test" / "0001_c3s1_000551_03.jpg",
    DATASET / "bounding_box_test" / "0001_c6s3_057292_01.jpg",
    DATASET / "bounding_box_test" / "0004_c2s3_059152_02.jpg",
    DATASET / "bounding_box_test" / "0001_c1s1_001051_03.jpg",
)
OUT = ROOT / "outputs" / "token_pruning"


def person_camera(path: Path) -> str:
    pid, camera = path.name.split("_")[:2]
    camera = camera.split("s")[0][1:]
    return f"person {pid}, camera {camera}"


def shade_dropped(image: Image.Image, kept, grid_h: int, grid_w: int, patch: int, scale: int = 4) -> Image.Image:
    resized = image.resize((grid_w * patch, grid_h * patch), Image.BILINEAR)
    canvas = resized.resize((resized.width * scale, resized.height * scale), Image.NEAREST).convert("RGBA")
    veil = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(veil)
    kept_ids = {int(index) for index in kept[0].tolist()}
    patches = grid_h * grid_w
    cell = patch * scale
    for index in range(patches):
        if index in kept_ids:
            continue
        row, col = divmod(index, grid_w)
        box = (col * cell, row * cell, (col + 1) * cell, (row + 1) * cell)
        draw.rectangle(box, fill=(0, 0, 0, 170))
    return Image.alpha_composite(canvas, veil).convert("RGB")


def rank(query, gallery) -> list[tuple[float, Path]]:
    scores = gallery @ query
    order = scores.argsort()[::-1]
    return [(float(scores[index]), GALLERY[int(index)]) for index in order]


def main() -> None:
    missing = [path for path in (QUERY, *GALLERY) if not path.is_file()]
    if missing:
        raise SystemExit("Missing photo: " + ", ".join(str(path) for path in missing))
    print("Token pruning on one query. This is not a Market-1501 score.")
    print("Query:", QUERY.name, person_camera(QUERY))
    encoder = CLIPReIDEncoder(ROOT / "checkpoints" / "Market1501_clipreid_ViT-B-16_60.pth")
    query = load_rgb(QUERY)
    gallery_images = [load_rgb(path) for path in GALLERY]

    encoder.set_token_pruning(1.0)
    full_query = encoder.embed_images([query])[0]
    full_gallery = encoder.embed_images(gallery_images)
    full_tokens = encoder.model.image_encoder.last_tokens_out

    encoder.set_token_pruning(0.5, prune_layer=6)
    pruned_query = encoder.embed_images([query])[0]
    kept = encoder.model.image_encoder.last_kept_patch_index.detach().cpu()
    tokens_in = encoder.model.image_encoder.last_tokens_in
    tokens_out = encoder.model.image_encoder.last_tokens_out
    pruned_gallery = encoder.embed_images(gallery_images)

    image = encoder.model.image_encoder
    OUT.mkdir(parents=True, exist_ok=True)
    picture = shade_dropped(query, kept, image.grid_h, image.grid_w, image.patch)
    picture_path = OUT / "0001_c1s1_001051_00_keep50.png"
    picture.save(picture_path)

    print(f"Tokens before pruning: {tokens_in}")
    print(f"Tokens after pruning:  {tokens_out}")
    print(f"Patch tokens kept:     {kept.shape[1]} of {image.grid_h * image.grid_w}")
    print(f"Class token kept:      yes")
    print(f"Prune after layer:     6 of 12")
    print(f"Unpruned token count:  {full_tokens}")
    print(f"Shaded photo:          {picture_path}")
    print()
    print(f"{'gallery':<32} {'full cosine':>12} {'pruned cosine':>14}")
    full_order = rank(full_query, full_gallery)
    pruned_scores = {path.name: score for score, path in rank(pruned_query, pruned_gallery)}
    for score, path in full_order:
        print(f"{path.name:<32} {score:12.4f} {pruned_scores[path.name]:14.4f}  {person_camera(path)}")
    print()
    print("Full order:   " + ", ".join(path.name for _score, path in full_order))
    print("Pruned order: " + ", ".join(path.name for _score, path in rank(pruned_query, pruned_gallery)))


if __name__ == "__main__":
    main()
