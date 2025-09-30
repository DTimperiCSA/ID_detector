# train_yolo.py
import subprocess
from pathlib import Path

from paths import COCO_DIR, COCO_YAML, OUTPUT_DIR

# Create YAML for YOLO
yaml_content = f"""
path: {COCO_DIR}          # root dir containing images/ and labels/
train: train              # relative to path
val: val                  # relative to path
nc: 1
names: ['id_card']
"""

COCO_YAML.parent.mkdir(parents=True, exist_ok=True)
COCO_YAML.write_text(yaml_content)
print(f"[OK] Dataset YAML written at {COCO_YAML}")

# Train YOLOv8
cmd = [
    "yolo",
    "detect",
    "train",
    f"data={COCO_YAML}",
    "model=yolov8n.pt",
    "epochs=50",
    "imgsz=640",
    f"project={OUTPUT_DIR}",
    "name=midv500_yolo",
]

print("[INFO] Running YOLOv8 training...")
subprocess.run(cmd, check=True)
print("[OK] Training finished")
