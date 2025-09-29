# /home/lucapolenta/Desktop/ID_detector/pipeline/convert_to_yolo.py
import json
from pathlib import Path

from paths import COCO_DIR, COCO_IMAGES, COCO_LABELS, MIDV500_DIR
from tqdm import tqdm


def polygon_to_yolo_bbox(points, width, height):
    """Convert polygon points to YOLO normalized bbox."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height

    return x_center, y_center, w, h


def convert_to_yolo():
    print(f"[INFO] Converting MIDV-500 annotations from {MIDV500_DIR}")

    # Create output dirs
    for d in [COCO_DIR, COCO_IMAGES, COCO_LABELS]:
        d.mkdir(parents=True, exist_ok=True)

    count = 0

    # Iterate all ground_truth.json files
    for gt_file in tqdm(list(MIDV500_DIR.rglob("ground_truth/*.json"))):
        with open(gt_file) as f:
            data = json.load(f)

        # Each JSON corresponds to a single image
        image_path = gt_file.parent.parent / "images" / (gt_file.stem + ".jpg")
        if not image_path.exists():
            continue

        width = data.get("image_size", {}).get("width", 1920)
        height = data.get("image_size", {}).get("height", 1080)

        # Extract document polygon
        document_polygon = None
        for key, value in data.items():
            if isinstance(value, dict) and "quad" in value:
                if key.lower() in ["document", "id_card", "passport", "card"]:
                    document_polygon = value["quad"]
                    break
        if document_polygon is None:
            # fallback: use full image if missing
            document_polygon = [[0, 0], [width, 0], [width, height], [0, height]]

        # Convert to YOLO bbox
        xc, yc, w, h = polygon_to_yolo_bbox(document_polygon, width, height)
        label_line = f"0 {xc} {yc} {w} {h}\n"

        # Copy image to coco/images/all/
        target_img = COCO_IMAGES / image_path.name
        target_label = COCO_LABELS / f"{image_path.stem}.txt"
        if not target_img.exists():
            target_img.write_bytes(image_path.read_bytes())
        target_label.write_text(label_line)
        count += 1

    print(f"[OK] Converted {count} images with YOLO labels.")
    print(f"[INFO] All YOLO txt files are in {COCO_LABELS}")


if __name__ == "__main__":
    convert_to_yolo()
