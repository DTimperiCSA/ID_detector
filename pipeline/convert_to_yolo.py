# convert_to_yolo.py
import json
import shutil
from pathlib import Path

from paths import COCO_IMAGES, COCO_LABELS, MIDV500_DIR
from PIL import Image
from tqdm import tqdm


def convert_bbox(corners, img_w, img_h):
    """Convert polygon corners to YOLO normalized bbox."""
    # If corners are dicts
    if isinstance(corners[0], dict):
        xs = [p["x"] for p in corners]
        ys = [p["y"] for p in corners]
    # If corners are lists [x, y]
    elif isinstance(corners[0], list) or isinstance(corners[0], tuple):
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
    else:
        raise ValueError(f"Unknown corner format: {corners[0]}")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height


def convert_midv500_to_yolo():
    # Recursively find all JSON files under any ground_truth subfolder
    json_files = list(MIDV500_DIR.rglob("ground_truth/*/*.json"))
    print(f"Found {len(json_files)} annotation files")

    for json_path in tqdm(json_files, desc="Converting to YOLO"):
        try:
            data = json.loads(json_path.read_text())
        except Exception as e:
            print(f"⚠️ JSON read error {json_path}: {e}")
            continue

        # Locate the corresponding image
        subfolder = json_path.parent.name  # e.g., "TS"
        json_parent = json_path.parents[1].parent
        image_dir = json_parent / "images" / subfolder  # images/TS

        img_path = None
        for ext in [".tif", ".png", ".jpg", ".jpeg"]:
            candidate = image_dir / f"{json_path.stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(img_path)
            print(f"⚠️ No matching image for {json_path}")
            continue

        # Load image size
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"⚠️ Image open failed {img_path}: {e}")
            continue

        # Extract polygon corners from JSON
        corners = data.get("quad") or data.get("points") or data.get("value")
        if not corners:
            print(f"⚠️ No bbox in {json_path}")
            continue

        # Convert to YOLO format
        xc, yc, bw, bh = convert_bbox(corners, w, h)

        # Save YOLO label
        out_label = COCO_LABELS / f"{json_path.stem}.txt"
        out_label.parent.mkdir(parents=True, exist_ok=True)
        out_label.write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Copy image (convert extension to .jpg)
        out_img = COCO_IMAGES / f"{img_path.stem}.jpg"
        if not out_img.exists():
            shutil.copy(img_path, out_img)

    print("✅ Conversion complete!")
    print(f"Images saved in: {COCO_IMAGES}")
    print(f"Labels saved in: {COCO_LABELS}")


if __name__ == "__main__":
    convert_midv500_to_yolo()
