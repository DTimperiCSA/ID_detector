# paths.py
from pathlib import Path

BASE_DIR = Path("/home/lucapolenta/Desktop/ID_detector")

# MIDV-500 dataset
MIDV500_DIR = BASE_DIR / "midv500"
MIDV500_URL = "https://github.com/fcakyon/midv500/archive/refs/heads/master.zip"

# COCO-style dataset (intermediate format)
COCO_DIR = BASE_DIR / "coco"
COCO_IMAGES = COCO_DIR / "images"
COCO_LABELS = COCO_DIR / "labels"
COCO_JSON = COCO_DIR / "midv500_coco.json"
COCO_YAML = COCO_DIR / "midv500.yaml"

# Train / validation splits
TRAIN_DIR = COCO_IMAGES / "train"
VAL_DIR = COCO_IMAGES / "val"
TRAIN_LABELS = COCO_LABELS / "train"
VAL_LABELS = COCO_LABELS / "val"

# YOLO trained model output
TRAINED_MODEL = BASE_DIR / "train" / "midv500_yolo" / "weights" / "best.pt"

# Pipeline / outputs
PIPELINE_DIR = BASE_DIR / "pipeline"
OUTPUT_DIR = BASE_DIR / "train"
TEST_IMAGES = BASE_DIR / "test_images"

# Create dirs if not exist
for p in [
    COCO_DIR,
    COCO_IMAGES,
    COCO_LABELS,
    TRAIN_DIR,
    VAL_DIR,
    TRAIN_LABELS,
    VAL_LABELS,
    PIPELINE_DIR,
    OUTPUT_DIR,
    TEST_IMAGES,
]:
    p.mkdir(parents=True, exist_ok=True)
