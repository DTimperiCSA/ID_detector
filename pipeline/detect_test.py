# /home/lucapolenta/Desktop/ID_detector/pipeline/detect_test.py

from pathlib import Path

from paths import BASE_DIR, TEST_IMAGES, TRAINED_MODEL
from ultralytics import YOLO


def run_inference():
    print(f"[INFO] Loading YOLO model from {TRAINED_MODEL}")
    model = YOLO(TRAINED_MODEL)

    output_dir = BASE_DIR / "runs" / "detect" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Running detection on test images in {TEST_IMAGES}")
    results = model.predict(
        source=str(TEST_IMAGES),
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        conf=0.25,
        imgsz=640,
        verbose=True,
    )

    print(f"[OK] Detection completed ✅")
    print(f"[INFO] Results saved in: {output_dir}")
    print(f"[INFO] Annotated images in: {output_dir}")
    print(f"[INFO] YOLO labels (txt) in: {output_dir}/labels")


if __name__ == "__main__":
    run_inference()
