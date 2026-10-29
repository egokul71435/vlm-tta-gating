"""Prepare ImageNet-A / ImageNet-R: download the images and build the class-name,
class-mask and template files the registry points to.

Sources (official repos in refs/, cloned as in the README):
  - 1000 ImageNet class names and the 80 CLIP templates: refs/tpt/data/imagnet_prompts.py
    (TDA's imagenet_classes list is identical)
  - ImageNet-A / -R class masks and the 1000 WNIDs: refs/tpt/data/imagenet_variants.py
  - TDA's 7-template set (ImageNet, -A, -V2): refs/tda/datasets/imagenet_a.py
    (TDA uses the 80 CLIP templates on -R and -Sketch)

Writes (under DATA_ROOT, default ./data):
  classnames/imagenet.json            1000 names, index = ImageNet label
  classnames/imagenet_{a,r}.json      200 names, index = label used by tta.data (sorted WNID folders)
  masks/imagenet_{a,r}_mask.json      the 200 ImageNet indices (ascending)
  templates/clip80.json               80 CLIP templates
  templates/tip7.json                 TDA's 7 templates
  templates/photo.json                "a photo of a {}." (TPT's starting prompt), if missing

Then checks the extracted folders: 200 WNID folders that match the mask, and the
expected image count (ImageNet-A 7,500; ImageNet-R 30,000).

Examples:
  python src_main/scripts/prepare_data.py imagenet_a --download
  python src_main/scripts/prepare_data.py imagenet_a imagenet_r          # metadata + checks only
"""

import argparse
import ast
import json
import tarfile
import urllib.request
from pathlib import Path

from tta.registry import DATA_ROOT, REPO_ROOT

SOURCES = {
    "imagenet_a": {"url": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
                   "folder": "imagenet-a", "n_images": 7500},
    "imagenet_r": {"url": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
                   "folder": "imagenet-r", "n_images": 30000},
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def exec_module(path):
    ns = {}
    exec(Path(path).read_text(), ns)
    return ns


def literal_from(path, name):
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def write_json(rel, obj):
    path = DATA_ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
    print(f"  wrote {path} ({len(obj)} entries)")


def download(url, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / Path(url).name
    if not tar_path.exists():
        print(f"  downloading {url} ...")

        def progress(blocks, block_size, total):
            done = blocks * block_size
            if total > 0 and blocks % 500 == 0:
                print(f"    {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="\r")

        urllib.request.urlretrieve(url, tar_path, progress)
        print()
    print(f"  extracting {tar_path} ...")
    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(dest_dir, filter="data")
        except TypeError:                       # Python without extraction filters
            tar.extractall(dest_dir)
    tar_path.unlink()


def check_folders(name, folder, expected_wnids, n_expected):
    root = DATA_ROOT / folder
    if not root.exists():
        print(f"  {root} not found yet (run with --download, or extract the tar into {DATA_ROOT}/)")
        return False
    wnids = sorted(d.name for d in root.iterdir() if d.is_dir())
    n_images = sum(1 for d in wnids for p in (root / d).iterdir() if p.suffix.lower() in IMG_EXTS)
    ok = wnids == expected_wnids and n_images == n_expected
    print(f"  {root}: {len(wnids)} class folders (match mask: {wnids == expected_wnids}), "
          f"{n_images:,} images (expected {n_expected:,}) -> {'OK' if ok else 'CHECK'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Prepare ImageNet-A / -R.")
    p.add_argument("datasets", nargs="+", choices=sorted(SOURCES))
    p.add_argument("--download", action="store_true", help="Download and extract the images too.")
    p.add_argument("--refs-tpt", default=str(REPO_ROOT / "refs" / "tpt"))
    p.add_argument("--refs-tda", default=str(REPO_ROOT / "refs" / "tda"))
    args = p.parse_args()

    prompts = exec_module(Path(args.refs_tpt) / "data" / "imagnet_prompts.py")
    variants = exec_module(Path(args.refs_tpt) / "data" / "imagenet_variants.py")
    classes, all_wnids = prompts["imagenet_classes"], variants["all_wnids"]
    assert len(classes) == len(all_wnids) == 1000 and all_wnids == sorted(all_wnids)

    masks = {
        "imagenet_a": list(variants["imagenet_a_mask"]),
        "imagenet_r": [i for i, keep in enumerate(variants["imagenet_r_mask"]) if keep],
    }

    print("Shared files:")
    write_json("classnames/imagenet.json", classes)
    write_json("templates/clip80.json", prompts["imagenet_templates"])
    write_json("templates/tip7.json", literal_from(Path(args.refs_tda) / "datasets" / "imagenet_a.py", "template"))
    if not (DATA_ROOT / "templates" / "photo.json").exists():
        write_json("templates/photo.json", ["a photo of a {}."])

    all_ok = True
    for name in args.datasets:
        src, mask = SOURCES[name], masks[name]
        assert len(mask) == 200 and mask == sorted(mask), f"{name}: mask must be 200 ascending indices"
        print(f"\n{name}:")
        write_json(f"classnames/{name}.json", [classes[i] for i in mask])
        write_json(f"masks/{name}_mask.json", mask)
        if args.download:
            download(src["url"], DATA_ROOT)
        all_ok &= check_folders(name, src["folder"], [all_wnids[i] for i in mask], src["n_images"])

    print("\nNext: python src_main/scripts/run_vanilla_clip.py --dataset imagenet_a "
          "--templates data/templates/clip80.json --tag ens80   (target 49.89%)" if all_ok else
          "\nFix the folder check above before running anything.")


if __name__ == "__main__":
    main()