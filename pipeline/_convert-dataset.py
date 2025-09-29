import json
import os

import cv2
import numpy as np
from midv500.utils import (
    create_dir,
    get_bbox_inside_image,
    list_annotation_paths_recursively,
)
from tqdm import tqdm


def convert(root_dir: str, export_dir: str, filename: str = "midv500"):
    create_dir(export_dir)

    images = []
    annotations = []

    annotation_paths = list_annotation_paths_recursively(root_dir)
    print("Converting to COCO format...")
    for ind, rel_annotation_path in enumerate(tqdm(annotation_paths)):
        rel_image_path = rel_annotation_path.replace("ground_truth", "images").replace(
            "json", "tif"
        )
        abs_image_path = os.path.join(root_dir, rel_image_path)
        image = cv2.imread(abs_image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_dict = {
            "file_name": rel_image_path,
            "height": image.shape[0],
            "width": image.shape[1],
            "id": ind,
        }
        images.append(image_dict)

        image_bbox = [0, 0, image.shape[1], image.shape[0]]
        abs_annotation_path = os.path.join(root_dir, rel_annotation_path)
        quad = json.load(open(abs_annotation_path, "r"))
        mask_coords = quad["quad"]

        label_xmin = min([pos[0] for pos in mask_coords])
        label_xmax = max([pos[0] for pos in mask_coords])
        label_ymin = min([pos[1] for pos in mask_coords])
        label_ymax = max([pos[1] for pos in mask_coords])
        label_bbox = get_bbox_inside_image(
            [label_xmin, label_ymin, label_xmax, label_ymax], image_bbox
        )

        label_area = int(
            (label_bbox[2] - label_bbox[0]) * (label_bbox[3] - label_bbox[1])
        )
        label_bbox_coco = [
            label_bbox[0],
            label_bbox[1],
            label_bbox[2] - label_bbox[0],
            label_bbox[3] - label_bbox[1],
        ]

        annotation_dict = {
            "iscrowd": 0,
            "image_id": image_dict["id"],
            "category_id": 1,
            "ignore": 0,
            "id": ind,
            "bbox": label_bbox_coco,
            "area": label_area,
            "segmentation": [[coord for pair in mask_coords for coord in pair]],
        }
        annotations.append(annotation_dict)

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"name": "id_card", "id": 1}],
    }
    export_path = os.path.join(export_dir, filename + "_coco.json")
    with open(export_path, "w") as f:
        json.dump(coco_dict, f, indent=4)


if __name__ == "__main__":
    root_dir = "/home/lucapolenta/Desktop/ID_detector/midv500/"
    export_dir = "/home/lucapolenta/Desktop/ID_detector/coco/"
    convert(root_dir, export_dir)
    print("COCO conversion done!")
