"""Run TDA (Training-free Dynamic Adapter) over a manifest of images.

Mirrors the official implementation (refs/tda: tda_runner.py, utils.py):
  - caches are updated with the CLIP pseudo-label BEFORE the image's own prediction
  - entropy-prioritized dynamic queues (positive + negative caches)
  - negative cache gated on loss / log2(C) (the official nats/bits convention)
  - fixed logit scale of 100
  - text weights = per-class mean of normalized template embeddings, renormalized

Views:
  --n-views 1   one center-cropped view (as described in the TDA paper; official
                code on ImageNet and the 10 cross-domain datasets)
  --n-views 64  center crop + 63 AugMix views, lowest-entropy 10% averaged
                (official code on ImageNet-A / -V2 / -R / Sketch)

Defaults reproduce the Imagewoof pilot: 1 view, one dog template, alpha=1.0,
beta=5.0, manifest order, full precision.

Example (official-style run):
  python src_main/scripts/run_tda.py --manifest data/manifests/imagenet_a.json \
      --classnames data/classnames/imagenet_a.json --templates data/templates/imagenet.json \
      --config refs/tda/configs/imagenet_a.yaml --n-views 64 --shuffle-seed 1 \
      --output results/imagenet_a/tda_64v.json --save-probs
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

LOGIT_SCALE = 100.0
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# --- Pilot defaults (Imagewoof) ---
PILOT_CLASSNAMES = [
    "Shih-Tzu", "Rhodesian ridgeback", "beagle", "English foxhound",
    "Australian terrier", "border terrier", "golden retriever",
    "Old English sheepdog", "Samoyed", "dingo",
]
PILOT_TEMPLATES = ["a photo of a {}, a type of dog."]
# shot_capacity and negative-cache settings match every official config.
# alpha/beta are dataset-specific officially; Imagewoof has no official config,
# so 1.0/5.0 is a documented choice, not a verified default.
PILOT_CONFIG = {
    "positive": {"enabled": True, "shot_capacity": 3, "alpha": 1.0, "beta": 5.0},
    "negative": {
        "enabled": True, "shot_capacity": 2, "alpha": 0.117, "beta": 1.0,
        "entropy_threshold": {"lower": 0.2, "upper": 0.5},
        "mask_threshold": {"lower": 0.03, "upper": 1.0},
    },
}


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------

def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json_list(path, default):
    if path is None:
        return default
    with open(path) as f:
        return json.load(f)


def load_config(path):
    if path is None:
        return PILOT_CONFIG
    import yaml  # pip install pyyaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_augmix_ops(refs_tda):
    """Use the official AugMix operation list so 64-view runs match exactly."""
    ops_dir = Path(refs_tda) / "datasets"
    sys.path.insert(0, str(ops_dir))
    try:
        from augmix_ops import augmentations
    except ImportError as e:
        raise SystemExit(f"Could not import AugMix ops from {ops_dir}/augmix_ops.py: {e}")
    return augmentations


# ----------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])
TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])
PRE_AUGMENT = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
])


def single_view(img):
    return TO_TENSOR(BASE_TRANSFORM(img)).unsqueeze(0)


class AugMixViews:
    """Official AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)."""

    def __init__(self, n_aug, aug_list, severity=1):
        self.n_aug = n_aug
        self.aug_list = aug_list
        self.severity = severity

    def _augmix(self, img):
        x_orig = PRE_AUGMENT(img)
        x_processed = TO_TENSOR(x_orig)
        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))
        mix = torch.zeros_like(x_processed)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, self.severity)
            mix += w[i] * TO_TENSOR(x_aug)
        return m * x_processed + (1 - m) * mix

    def __call__(self, img):
        views = [TO_TENSOR(BASE_TRANSFORM(img))]
        views += [self._augmix(img) for _ in range(self.n_aug)]
        return torch.stack(views)


# ----------------------------------------------------------------------------
# Official TDA logic
# ----------------------------------------------------------------------------

def softmax_entropy(logits):
    """Per-row entropy in nats."""
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


def avg_entropy(logits):
    """Entropy of the averaged prediction across rows (official avg_entropy)."""
    log_p = logits - logits.logsumexp(-1, keepdim=True)
    avg = log_p.logsumexp(0) - math.log(logits.shape[0])
    avg = avg.clamp(min=torch.finfo(avg.dtype).min)
    return -(avg * avg.exp()).sum(-1)


def build_text_weights(model, tokenizer, classnames, templates, device):
    """Official clip_classifier: (dim, C) matrix of class embeddings."""
    weights = []
    with torch.no_grad():
        for name in classnames:
            texts = tokenizer([t.format(name.replace("_", " ")) for t in templates]).to(device)
            emb = model.encode_text(texts)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.mean(0)
            weights.append(emb / emb.norm())
    return torch.stack(weights, dim=1)


def clip_outputs(views, model, text_weights, select_frac=0.1):
    """Official get_clip_logits. views: (V, 3, 224, 224) with V = 1 or 64."""
    feats = model.encode_image(views)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    logits = LOGIT_SCALE * feats @ text_weights
    if feats.shape[0] > 1:
        ent = softmax_entropy(logits)
        idx = ent.argsort()[: int(feats.shape[0] * select_frac)]
        sel = logits[idx]
        feat = feats[idx].mean(0, keepdim=True)          # not renormalized, as in official
        logits = sel.mean(0, keepdim=True)
        loss = avg_entropy(sel)
        probs = sel.softmax(-1).mean(0, keepdim=True)
    else:
        feat = feats
        loss = softmax_entropy(logits)[0]
        probs = logits.softmax(-1)
    pred = int(logits.argmax(-1))
    return feat, logits, float(loss), probs, pred


def update_cache(cache, pred, item, capacity):
    """Official update_cache. item = [feat, loss] or [feat, loss, probs]."""
    entries = cache.setdefault(pred, [])
    if len(entries) < capacity:
        entries.append(item)
    elif item[1] < entries[-1][1]:
        entries[-1] = item
    entries.sort(key=lambda e: e[1])


def cache_logits(feat, cache, alpha, beta, num_classes, mask_thresholds=None):
    """Official compute_cache_logits."""
    keys, values = [], []
    for c in sorted(cache):
        for item in cache[c]:
            keys.append(item[0])
            values.append(item[2] if mask_thresholds else c)
    keys = torch.cat(keys, dim=0).T
    if mask_thresholds:
        v = torch.cat(values, dim=0)
        values = ((v > mask_thresholds[0]) & (v < mask_thresholds[1])).to(feat.dtype)
    else:
        values = F.one_hot(torch.tensor(values, device=feat.device), num_classes).to(feat.dtype)
    affinity = feat @ keys
    return alpha * (-(beta - beta * affinity)).exp() @ values


# ----------------------------------------------------------------------------
# Gate features (computed BEFORE this image updates the caches)
# ----------------------------------------------------------------------------

def gate_stats(feat, pos_cache, neg_cache, pos_cfg, num_classes, clip_pred):
    neg_fill = sum(len(v) for v in neg_cache.values())
    keys = [item[0] for c in pos_cache for item in pos_cache[c]]
    if not keys:
        return {"pos_cache_max_sim": 0.0, "pos_cache_mean_sim": 0.0, "cache_fill": 0,
                "cache_pred": -1, "cache_agrees_with_clip": 0, "neg_cache_fill": neg_fill}
    sims = (feat @ torch.cat(keys, dim=0).T).squeeze(0)
    pos_logits = cache_logits(feat, pos_cache, pos_cfg["alpha"], pos_cfg["beta"], num_classes)
    cache_pred = int(pos_logits.argmax(-1))
    return {
        "pos_cache_max_sim": float(sims.max()),
        "pos_cache_mean_sim": float(sims.mean()),
        "cache_fill": len(keys),
        "cache_pred": cache_pred,
        "cache_agrees_with_clip": int(cache_pred == clip_pred),
        "neg_cache_fill": neg_fill,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run TDA over an image manifest.")
    p.add_argument("--manifest", default="data/manifests/corrupted_manifest.json")
    p.add_argument("--output", default="results/tda_results.json")
    p.add_argument("--vanilla-results", default="results/vanilla_clip_results.json",
                   help="Optional; used for improved/regressed vs. vanilla CLIP.")
    p.add_argument("--classnames", default=None, help="JSON list of class names (default: pilot dogs).")
    p.add_argument("--templates", default=None, help="JSON list of templates with {} (default: pilot dog template).")
    p.add_argument("--config", default=None, help="Official TDA YAML config (default: pilot settings).")
    p.add_argument("--n-views", type=int, default=1, choices=[1, 64])
    p.add_argument("--refs-tda", default="refs/tda", help="Official TDA repo (for AugMix ops).")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--half", action="store_true", help="Half precision (CUDA only), as official.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="Shuffle the stream with this seed (official uses a seeded shuffle). Default: manifest order.")
    p.add_argument("--n", type=int, default=None, help="Use only the first N images (smoke tests).")
    p.add_argument("--save-probs", action="store_true", help="Also save probability vectors (.npz).")
    p.add_argument("--model", default="ViT-B-16-quickgelu")
    p.add_argument("--pretrained", default="openai")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    set_seeds(args.seed)

    classnames = load_json_list(args.classnames, PILOT_CLASSNAMES)
    templates = load_json_list(args.templates, PILOT_TEMPLATES)
    cfg = load_config(args.config)
    num_classes = len(classnames)
    pos_cfg, neg_cfg = cfg["positive"], cfg["negative"]
    mask_thr = (neg_cfg["mask_threshold"]["lower"], neg_cfg["mask_threshold"]["upper"])

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    if args.half and device == "cuda":
        model = model.half()
    dtype = next(model.parameters()).dtype

    text_weights = build_text_weights(model, tokenizer, classnames, templates, device)
    make_views = single_view if args.n_views == 1 else AugMixViews(
        args.n_views - 1, load_augmix_ops(args.refs_tda))

    with open(args.manifest) as f:
        images = json.load(f)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(images)
    if args.n is not None:
        images = images[: args.n]

    vanilla = {}
    if args.vanilla_results and Path(args.vanilla_results).exists():
        with open(args.vanilla_results) as f:
            vanilla = {r["corrupted_path"]: r for r in json.load(f)}

    pos_cache, neg_cache = {}, {}
    results, final_probs_all, clip_probs_all, paths = [], [], [], []

    print(f"TDA | device={device} | views={args.n_views} | classes={num_classes} | "
          f"templates={len(templates)} | alpha={pos_cfg['alpha']} beta={pos_cfg['beta']} | images={len(images)}")

    with torch.no_grad():
        for i, entry in enumerate(images):
            path = entry.get("corrupted_path", entry.get("path"))
            img = Image.open(path).convert("RGB")

            t0 = time.perf_counter()
            views = make_views(img)
            t_views = time.perf_counter() - t0

            views = views.to(device, dtype)
            feat, logits, loss, probs, clip_pred = clip_outputs(views, model, text_weights)
            norm_ent = loss / math.log2(num_classes)            # official convention

            stats = gate_stats(feat, pos_cache, neg_cache, pos_cfg, num_classes, clip_pred)

            if pos_cfg["enabled"]:
                update_cache(pos_cache, clip_pred, [feat, loss], pos_cfg["shot_capacity"])
            if neg_cfg["enabled"] and (neg_cfg["entropy_threshold"]["lower"] < norm_ent
                                       < neg_cfg["entropy_threshold"]["upper"]):
                update_cache(neg_cache, clip_pred, [feat, loss, probs], neg_cfg["shot_capacity"])

            final = logits.clone()
            if pos_cfg["enabled"] and pos_cache:
                final += cache_logits(feat, pos_cache, pos_cfg["alpha"], pos_cfg["beta"], num_classes)
            if neg_cfg["enabled"] and neg_cache:
                final -= cache_logits(feat, neg_cache, neg_cfg["alpha"], neg_cfg["beta"],
                                      num_classes, mask_thr)

            pred_idx = int(final.argmax(-1))
            synchronize(device)
            elapsed = time.perf_counter() - t0

            final_probs = final.float().softmax(-1).squeeze(0).cpu().numpy()
            pred_label = classnames[pred_idx]
            correct = pred_label == entry["label"]
            v = vanilla.get(path, {})
            vanilla_correct = v.get("correct")

            results.append({
                "corrupted_path": path,
                "true_label": entry["label"],
                "corruption_type": entry.get("corruption_type"),
                "predicted_label": pred_label,
                "correct": correct,
                "confidence": float(final_probs.max()),
                "entropy": float(-(final_probs * np.log(np.clip(final_probs, 1e-12, 1))).sum()),
                "vanilla_correct": vanilla_correct,
                "improved_over_vanilla": (correct and not vanilla_correct) if vanilla_correct is not None else None,
                "pos_cache_max_sim": stats["pos_cache_max_sim"],
                "pos_cache_mean_sim": stats["pos_cache_mean_sim"],
                "cache_fill": stats["cache_fill"],
                "cache_agrees_with_clip": stats["cache_agrees_with_clip"],
                "cache_pred_label": classnames[stats["cache_pred"]] if stats["cache_pred"] >= 0 else None,
                "neg_cache_fill": stats["neg_cache_fill"],
                "n_views": args.n_views,
                "view_seconds": t_views,
                "model_seconds": elapsed - t_views,
                "seconds": elapsed,
            })
            if args.save_probs:
                final_probs_all.append(final_probs)
                clip_probs_all.append(probs.float().squeeze(0).cpu().numpy())
                paths.append(path)

            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/{len(images)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    if args.save_probs:
        np.savez_compressed(out.with_suffix(".probs.npz"), paths=np.array(paths),
                            final_probs=np.stack(final_probs_all), clip_probs=np.stack(clip_probs_all))

    acc = np.mean([r["correct"] for r in results])
    secs = np.array([r["seconds"] for r in results])
    view_secs = np.array([r["view_seconds"] for r in results])
    print(f"\nTDA accuracy: {acc:.3f}  ({len(results)} images, {args.n_views} view(s))")
    if vanilla:
        improved = sum(1 for r in results if r["improved_over_vanilla"])
        regressed = sum(1 for r in results if r["vanilla_correct"] and not r["correct"])
        print(f"Improved over vanilla: {improved} | regressed: {regressed}")
    print(f"Time per image: median {np.median(secs)*1000:.1f} ms "
          f"(views {np.median(view_secs)*1000:.1f} ms, model {np.median(secs - view_secs)*1000:.1f} ms)")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()