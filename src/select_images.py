import os
import random
import json
from pathlib import Path

random.seed(42)

IMAGENETTE_VAL_DIR = Path("data/raw/imagewoof2-160/val")
NUM_IMAGES = 1000
OUTPUT_MANIFEST = Path("data/manifests/base_images.json")

# mapping
CLASS_NAMES = {
    "n02086240": "Shih-Tzu",
    "n02087394": "Rhodesian ridgeback",
    "n02088364": "beagle",
    "n02089973": "English foxhound",
    "n02093754": "Australian terrier",
    "n02096294": "border terrier",
    "n02099601": "golden retriever",
    "n02105641": "Old English sheepdog",
    "n02111889": "Samoyed",
    "n02115641": "dingo",
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