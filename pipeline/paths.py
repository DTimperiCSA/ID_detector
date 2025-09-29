# /home/lucapolenta/Desktop/ID_detector/pipeline/paths.py
from pathlib import Path

BASE_DIR = Path("/home/lucapolenta/Desktop/ID_detector")

# Directories
MIDV500_DIR = BASE_DIR / "midv500"
COCO_DIR = BASE_DIR / "coco"
COCO_IMAGES = COCO_DIR / "images"
COCO_LABELS = COCO_DIR / "labels"

TRAIN_DIR = COCO_IMAGES / "train"
VAL_DIR = COCO_IMAGES / "val"
TRAIN_LABELS = COCO_LABELS / "train"
VAL_LABELS = COCO_LABELS / "val"

COCO_JSON = COCO_DIR / "midv500_coco.json"
COCO_YAML = COCO_DIR / "midv500.yaml"
# Add this line
TRAINED_MODEL = BASE_DIR / "train" / "midv500_yolo" / "weights" / "best.pt"

OUTPUT_DIR = BASE_DIR / "train"

# URLs
MIDV500_URL = "https://github.com/fcakyon/midv500/archive/refs/heads/master.zip"

PIPELINE_DIR = BASE_DIR / "pipeline"
MIDV500_DIR = BASE_DIR / "midv500"
TEST_IMAGES = BASE_DIR / "test_images"
TRAIN_DIR = BASE_DIR / "train"
