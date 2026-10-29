"""
Dataset registry: one entry per benchmark, so every script takes --dataset <name>
and gets the right images, class names, templates, class mask, TDA config and
method settings. Published targets are ViT-B/16 numbers from the TDA paper
(CLIP = template ensemble), used by the reproduction check.

Paths are relative to DATA_ROOT (env var, default ./data) so the same code runs
on the Mac and the cluster.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))
REPO_ROOT = Path(__file__).resolve().parents[2]  # src_main/tta/registry.py -> repo root
CONFIG_ROOT = REPO_ROOT / "configs"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_root: str              # relative to DATA_ROOT
    layout: str                  # "imagefolder" | "manifest" | "split_json" (CoOp-style splits)
    classnames: str              # json list, index = label id (relative to DATA_ROOT)
    templates: str               # json list of "{}" templates
    tda_config: Optional[str]    # yaml under configs/tda/ (copied from refs/tda/configs)
    tda_n_views: int             # official TDA: 64 on ImageNet-A/V2/R/S, else 1
    tpt_augmix: bool             # official TPT: AugMix only on cross-domain sets
    class_mask: Optional[str] = None   # json list of ImageNet indices (A and R use 200 classes)
    manifest: Optional[str] = None     # for layout="manifest" (pilot)
    split_file: Optional[str] = None   # for layout="split_json"
    targets: dict = field(default_factory=dict)  # published accuracy, % (clip/tda/tpt)

    # --- resolved paths ---
    @property
    def image_dir(self) -> Path:
        return DATA_ROOT / self.image_root

    def load_classnames(self) -> list[str]:
        return json.loads((DATA_ROOT / self.classnames).read_text())

    def load_templates(self) -> list[str]:
        return json.loads((DATA_ROOT / self.templates).read_text())

    def load_class_mask(self) -> Optional[list[int]]:
        if self.class_mask is None:
            return None
        return json.loads((DATA_ROOT / self.class_mask).read_text())

    def tda_config_path(self) -> Optional[Path]:
        return CONFIG_ROOT / "tda" / self.tda_config if self.tda_config else None


# TDA's official templates: its 7-template set on ImageNet, -A and -V2; the 80
# CLIP templates on -R and -Sketch (refs/tda/datasets/*.py). Built by
# src_main/scripts/prepare_data.py. The published CLIP targets below come from
# the TPT paper's 80-template ensemble, so check zero-shot accuracy with
# --templates data/templates/clip80.json.
_TIP7 = "templates/tip7.json"
_CLIP80 = "templates/clip80.json"
_IMAGENET_FAMILY = dict(tpt_augmix=False)

REGISTRY: dict[str, DatasetSpec] = {
    # ---- pilot (kept so the regression check keeps working) ----
    "imagewoof_pilot": DatasetSpec(
        name="imagewoof_pilot",
        image_root="corrupted",
        layout="manifest",
        manifest="manifests/corrupted_manifest.json",
        classnames="classnames/imagewoof.json",
        templates="templates/photo.json",
        tda_config=None,          # pilot alpha=1.0, beta=5.0 (no official config)
        tda_n_views=1,
        tpt_augmix=False,
        targets={"clip": 59.0, "tda": 60.1, "tpt": 61.0},  # our aligned pilot, for regression
    ),

    # ---- ImageNet OOD suite ----
    "imagenet_a": DatasetSpec(
        name="imagenet_a", templates=_TIP7, image_root="imagenet-a", layout="imagefolder",
        classnames="classnames/imagenet_a.json", class_mask="masks/imagenet_a_mask.json",
        tda_config="imagenet_a.yaml", tda_n_views=64,
        targets={"clip": 49.89, "tda": 60.11, "tpt": 54.77}, **_IMAGENET_FAMILY,
    ),
    "imagenet_r": DatasetSpec(
        name="imagenet_r", templates=_CLIP80, image_root="imagenet-r", layout="imagefolder",
        classnames="classnames/imagenet_r.json", class_mask="masks/imagenet_r_mask.json",
        tda_config="imagenet_r.yaml", tda_n_views=64,
        targets={"clip": 77.65, "tda": 80.24, "tpt": 77.06}, **_IMAGENET_FAMILY,
    ),
    "imagenet_v2": DatasetSpec(
        name="imagenet_v2", templates=_TIP7, image_root="imagenetv2-matched-frequency", layout="imagefolder",
        classnames="classnames/imagenet.json",
        tda_config="imagenet_v.yaml", tda_n_views=64,
        targets={"clip": 61.88, "tda": 64.67, "tpt": 63.45}, **_IMAGENET_FAMILY,
    ),
    "imagenet_sketch": DatasetSpec(
        name="imagenet_sketch", templates=_CLIP80, image_root="imagenet-sketch", layout="imagefolder",
        classnames="classnames/imagenet.json",
        tda_config="imagenet_s.yaml", tda_n_views=64,
        targets={"clip": 48.24, "tda": 50.54, "tpt": 47.94}, **_IMAGENET_FAMILY,
    ),
    "imagenet": DatasetSpec(
        name="imagenet", templates=_TIP7, image_root="imagenet/val", layout="imagefolder",
        classnames="classnames/imagenet.json",
        tda_config="imagenet.yaml", tda_n_views=1,
        targets={"clip": 68.34, "tda": 69.51, "tpt": 68.98}, **_IMAGENET_FAMILY,
    ),

    # ---- cross-domain (add the other nine the same way; tpt_augmix=True, tda_n_views=1,
    #      per-dataset templates from refs/tda/datasets/<name>.py, splits from CoOp) ----
    # "dtd": DatasetSpec(name="dtd", image_root="dtd/images", layout="split_json",
    #     split_file="dtd/split_zhou_DescribableTextures.json",
    #     classnames="classnames/dtd.json", templates="templates/dtd.json",
    #     tda_config="dtd.yaml", tda_n_views=1, tpt_augmix=True, targets={...}),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def add_dataset_arg(parser):
    """Shared --dataset flag for every entry script."""
    parser.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    return parser