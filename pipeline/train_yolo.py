# /home/lucapolenta/Desktop/ID_detector/pipeline/train_yolo.py
import os

from paths import COCO_YAML, OUTPUT_DIR

yaml_content = f"""
path: /home/lucapolenta/Desktop/ID_detector/coco
train: images/train
val: images/val
nc: 1
names: ['id_card']
"""
with open(COCO_YAML, "w") as f:
    f.write(yaml_content)

print(f"[OK] Dataset YAML written at {COCO_YAML}")

os.system(
    f"""
yolo detect train \
    data={COCO_YAML} \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640 \
    project={OUTPUT_DIR} \
    name=midv500_yolo
"""
)
