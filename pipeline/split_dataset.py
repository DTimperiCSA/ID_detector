# /home/lucapolenta/Desktop/ID_detector/pipeline/split_dataset.py
import shutil
from pathlib import Path

from paths import COCO_IMAGES, COCO_LABELS
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_dataset(train_ratio=0.8):
    print(f"[INFO] Splitting dataset into train/val with ratio {train_ratio}")

    all_images = sorted(list((COCO_IMAGES / "all").glob("*.*")))
    all_labels = sorted(list((COCO_LABELS / "all").glob("*.txt")))

    assert len(all_images) == len(
        all_labels
    ), f"Image/label count mismatch: {len(all_images)} vs {len(all_labels)}"

    # split
    train_imgs, val_imgs = train_test_split(
        all_images, train_size=train_ratio, random_state=42
    )

    for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
        img_out_dir = COCO_IMAGES / split_name
        lbl_out_dir = COCO_LABELS / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(img_list, desc=f"Copying {split_name}"):
            label_path = COCO_LABELS / "all" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(img_path, img_out_dir / img_path.name)
                shutil.copy(label_path, lbl_out_dir / label_path.name)

    print(f"[OK] Split complete.")
    print(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")
    print(f"Images saved under {COCO_IMAGES}, labels under {COCO_LABELS}")


if __name__ == "__main__":
    split_dataset()
