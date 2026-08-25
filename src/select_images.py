import os
import random
import json
from pathlib import Path

random.seed(42)

IMAGENETTE_VAL_DIR = Path("data/raw/imagenette2-160/val")
NUM_IMAGES = 150
OUTPUT_MANIFEST = Path("data/manifests/base_images.json")

# Imagenette class folder names map to WordNet IDs, not human labels
# below mapping is Imagenette's official label set
CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

def collect_images():
    all_images = []
    for class_dir in IMAGENETTE_VAL_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        class_id = class_dir.name
        label = CLASS_NAMES.get(class_id, class_id)
        for img_path in class_dir.glob("*.JPEG"):
            all_images.append({
                "path": str(img_path),
                "class_id": class_id,
                "label": label,
            })
    return all_images

def main():
    all_images = collect_images()
    print(f"Found {len(all_images)} total images across {len(CLASS_NAMES)} classes")

    if len(all_images) < NUM_IMAGES:
        raise ValueError(f"Not enough images found ({len(all_images)}) for requested {NUM_IMAGES}")

    # shuffle then take a roughly even split across classes where possible
    random.shuffle(all_images)
    selected = all_images[:NUM_IMAGES]

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MANIFEST, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"Selected {len(selected)} images, saved manifest to {OUTPUT_MANIFEST}")

    # quick check: class distribution
    from collections import Counter
    counts = Counter(img["label"] for img in selected)
    print("Class distribution in selection:")
    for label, count in counts.items():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main()