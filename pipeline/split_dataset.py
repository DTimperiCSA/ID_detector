# split_dataset.py
import random
import shutil
from pathlib import Path

from paths import COCO_IMAGES, COCO_LABELS

random.seed(42)

# Use COCO directories
dataset_dir = COCO_IMAGES.parent  # COCO folder containing images/ and labels/
images_dir = COCO_IMAGES  # flat images folder
annotations_dir = COCO_LABELS  # flat labels folder

# Train / val directories
train_dir = dataset_dir / "train"
val_dir = dataset_dir / "val"

# Create necessary subfolders
for d in [train_dir, val_dir]:
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)

# Get all images
images = list(images_dir.glob("*.jpg"))
random.shuffle(images)

# Split 80/20
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]


def copy_files(img_list, target_dir):
    for img_path in img_list:
        label_path = annotations_dir / (img_path.stem + ".txt")
        shutil.copy(img_path, target_dir / "images" / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, target_dir / "labels" / label_path.name)


# Copy files to train / val
copy_files(train_images, train_dir)
copy_files(val_images, val_dir)

print(f"✅ Dataset split complete!")
print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
print(f"Check folders: {train_dir}, {val_dir}")
