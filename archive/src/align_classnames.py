# align_classnames.py  (run once from the repo root)
import json
import sys
from pathlib import Path

sys.path.insert(0, "refs/tpt")
from data.imagnet_prompts import imagenet_classes   # official TPT class-name list (1,000 classes)

# Imagewoof WordNet IDs -> ImageNet class index, in the pilot's class order
WNID_TO_IDX = {
    "n02086240": 155, "n02087394": 159, "n02088364": 162, "n02089973": 167,
    "n02093754": 193, "n02096294": 182, "n02099601": 207, "n02105641": 229,
    "n02111889": 258, "n02115641": 273,
}

official = {w: imagenet_classes[i] for w, i in WNID_TO_IDX.items()}
for w, name in official.items():
    print(f"{w}  idx {WNID_TO_IDX[w]:>3}  ->  {name}")

Path("data/classnames").mkdir(parents=True, exist_ok=True)
with open("data/classnames/imagewoof.json", "w") as f:
    json.dump([official[w] for w in WNID_TO_IDX], f, indent=2)

Path("data/templates").mkdir(parents=True, exist_ok=True)
with open("data/templates/photo.json", "w") as f:
    json.dump(["a photo of a {}."], f, indent=2)

for mf in ["data/manifests/base_images.json", "data/manifests/corrupted_manifest.json"]:
    with open(mf) as f:
        rows = json.load(f)
    for r in rows:
        r["label"] = official[r["class_id"]]
    with open(mf, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"relabeled {len(rows)} rows in {mf}")