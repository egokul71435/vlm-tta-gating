"""
Shared helpers for every entry script: device, seeding, CLIP loading, text
weights, uncertainty stats, timing, and one results format.

Results layout:  results/<dataset>/<method>[_<tag>].json   (list of per-image rows)
                 results/<dataset>/<method>[_<tag>].meta.json  (args, git commit, device, summary)
Every row has: id, label, pred, correct. `id` is the sample's path relative to
DATA_ROOT, so results from the Mac and the cluster join on the same key.
"""

import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import open_clip
import torch

MODEL_NAME = "ViT-B-16-quickgelu"
PRETRAINED = "openai"
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "results"))


# ---------------------------------------------------------------- device / seed
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Call before reading a timer, so GPU/MPS work is actually finished."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- CLIP
def load_clip(device: torch.device, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED):
    """Model init draws random numbers, so seed BEFORE calling this when exact
    reproducibility matters (TPT's legacy random prompt init depends on it)."""
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model.to(device).eval(), preprocess, tokenizer


@torch.no_grad()
def build_text_weights(model, tokenizer, classnames, templates, device) -> torch.Tensor:
    """Template-ensembled class embeddings, (D, C), same layout as official TDA's
    clip_classifier: normalize each template embedding, average, renormalize."""
    weights = []
    for name in classnames:
        name = name.replace("_", " ")
        tokens = tokenizer([t.format(name) for t in templates]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0)
        weights.append(emb / emb.norm())
    return torch.stack(weights, dim=1)


# ---------------------------------------------------------------- uncertainty
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


def uncertainty_stats(logits: torch.Tensor, prefix: str) -> dict:
    """Confidence, top-2 margin, entropy and class-count-normalized entropy for
    one image's logits (shape (C,) or (1, C)). Normalized entropy (entropy / ln C)
    is what lets a gate trained on one dataset transfer to another with a
    different number of classes."""
    logits = logits.reshape(-1).float()
    probs = logits.softmax(-1)
    top2 = probs.topk(2).values
    ent = float(softmax_entropy(logits))
    return {
        f"{prefix}_pred": int(probs.argmax()),
        f"{prefix}_conf": float(top2[0]),
        f"{prefix}_margin": float(top2[0] - top2[1]),
        f"{prefix}_entropy": ent,
        f"{prefix}_norm_entropy": ent / math.log(logits.numel()),
    }


# ---------------------------------------------------------------- timing
class Timer:
    """with Timer(device) as t: ...   then t.seconds (synced before stopping)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.seconds = 0.0

    def __enter__(self):
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.seconds = time.perf_counter() - self._t0


# ---------------------------------------------------------------- results I/O
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def results_path(dataset: str, method: str, tag: str | None = None) -> Path:
    name = f"{method}_{tag}" if tag else method
    return RESULTS_ROOT / dataset / f"{name}.json"


def save_results(rows: list[dict], path: Path, args=None, summary: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    meta = {
        "git_commit": git_commit(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / max(len(rows), 1),
        "args": vars(args) if args is not None else None,
        "summary": summary or {},
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_results(path: Path) -> dict[str, dict]:
    """Rows keyed by sample id."""
    return {r["id"]: r for r in json.loads(Path(path).read_text())}