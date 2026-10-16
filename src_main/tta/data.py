"""
Turn a DatasetSpec into an ordered list of samples (id, path, label, meta).
Transforms stay in the method scripts, because TDA and TPT build different views.

Label conventions (the two places loaders usually go wrong):
- WNID folders (ImageNet, -A, -R, -Sketch): label = position in sorted folder order.
  ImageNet indices are assigned in sorted-WNID order, so for -A/-R (200 classes)
  sorted folder i corresponds to the i-th entry of the ascending class mask,
  i.e. to classnames/imagenet_{a,r}.json[i].
- ImageNet-V2 folders are named "0".."999" (the ImageNet index). Sorting them as
  strings gives 0, 1, 10, 100, ... so the label must be int(folder name).
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .registry import DATA_ROOT, DatasetSpec

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    id: str        # path relative to DATA_ROOT; the join key across all results
    path: str      # absolute or cwd-relative path to open
    label: int
    meta: dict = field(default_factory=dict)


def load_samples(spec: DatasetSpec, shuffle_seed: int | None = None, n: int | None = None) -> list[Sample]:
    loaders = {"imagefolder": _imagefolder, "manifest": _manifest, "split_json": _split_json}
    samples = loaders[spec.layout](spec)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(samples)
    return samples[:n] if n else samples


def _rel_id(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(DATA_ROOT.resolve()))
    except ValueError:
        return str(path)


def _imagefolder(spec: DatasetSpec) -> list[Sample]:
    root = spec.image_dir
    folders = sorted(d.name for d in root.iterdir() if d.is_dir())
    classnames = spec.load_classnames()
    if len(folders) != len(classnames):
        raise ValueError(f"{spec.name}: {len(folders)} folders but {len(classnames)} class names")

    if all(f.isdigit() for f in folders):          # ImageNet-V2
        label_of = {f: int(f) for f in folders}
    else:                                          # WNID folders
        label_of = {f: i for i, f in enumerate(folders)}
        mask = spec.load_class_mask()
        if mask is not None and (mask != sorted(mask) or len(mask) != len(folders)):
            raise ValueError(f"{spec.name}: class mask must be ascending and match the folder count")

    samples = []
    for folder in folders:
        for p in sorted((root / folder).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                samples.append(Sample(_rel_id(p), str(p), label_of[folder], {"folder": folder}))
    return samples


def _manifest(spec: DatasetSpec) -> list[Sample]:
    """Pilot manifest. Keeps corrupted_path as the id so the old results still join."""
    entries = json.loads((DATA_ROOT / spec.manifest).read_text())
    classnames = spec.load_classnames()
    samples = []
    for e in entries:
        path = e["corrupted_path"]
        cid = e.get("class_id")
        label = cid if isinstance(cid, int) else classnames.index(e["label"])  # pilot class_id is a WNID
        samples.append(Sample(path, path, int(label), {"corruption_type": e.get("corruption_type")}))
    return samples


def _split_json(spec: DatasetSpec) -> list[Sample]:
    """CoOp-style split file: {"test": [[relative_impath, label, classname], ...], ...}."""
    split = json.loads((DATA_ROOT / spec.split_file).read_text())
    samples = []
    for impath, label, _ in split["test"]:
        p = spec.image_dir / impath
        samples.append(Sample(_rel_id(p), str(p), int(label)))
    return samples