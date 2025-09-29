# /home/lucapolenta/Desktop/ID_detector/pipeline/download_dataset.py
import io
import shutil
import zipfile
from pathlib import Path

import requests
from paths import MIDV500_DIR, MIDV500_URL


def download_midv500():
    if MIDV500_DIR.exists():
        print(f"[INFO] MIDV500 already exists at {MIDV500_DIR}")
        return
    print("[INFO] Downloading MIDV500 dataset...")
    response = requests.get(MIDV500_URL)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(MIDV500_DIR.parent)
    # Move folder if needed
    extracted = MIDV500_DIR.parent / "midv500-master"
    if extracted.exists():
        shutil.move(str(extracted), str(MIDV500_DIR))
    print(f"[OK] Dataset ready at {MIDV500_DIR}")


if __name__ == "__main__":
    download_midv500()
