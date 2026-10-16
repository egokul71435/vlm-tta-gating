"""Run TDA (Training-free Dynamic Adapter) on a registered dataset.

Mirrors the official implementation (refs/tda: tda_runner.py, utils.py):
  - caches are updated with the CLIP pseudo-label BEFORE the image's own prediction
  - entropy-prioritized dynamic queues (positive + negative caches)
  - negative cache gated on loss / log2(C) (the official nats/bits convention)
  - fixed logit scale of 100
  - text weights = per-class mean of normalized template embeddings, renormalized

Everything dataset-specific (images, class names, templates, TDA config, number
of views) comes from the registry entry for --dataset; the flags below override
it for experiments.

Views:
  1   one center-cropped view (as described in the TDA paper; official code on
      ImageNet and the 10 cross-domain datasets)
  64  center crop + 63 AugMix views, lowest-entropy 10% averaged (official code
      on ImageNet-A / -V2 / -R / Sketch)

Examples:
  python src_main/scripts/run_tda.py --dataset imagewoof_pilot
  python src_main/scripts/run_tda.py --dataset imagewoof_pilot --shuffle-seed 1 --tag seed1
  python src_main/scripts/run_tda.py --dataset imagenet_r --shuffle-seed 1 --save-probs
  python src_main/scripts/run_tda.py --dataset imagenet_r --n-views 1 --tag 1view
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from tta.common import (build_text_weights, load_clip, load_results, pick_device, results_path,
                        save_results, seed_everything, softmax_entropy, sync, uncertainty_stats)
from tta.data import load_samples
from tta.registry import REPO_ROOT, add_dataset_arg, get_dataset

LOGIT_SCALE = 100.0
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Used when the dataset has no official TDA config (the Imagewoof pilot).
# shot_capacity and negative-cache settings match every official config;
# alpha/beta are dataset-specific officially, so 1.0/5.0 is a documented choice.
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
# Official TDA logic (unchanged)
# ----------------------------------------------------------------------------

def avg_entropy(logits):
    """Entropy of the averaged prediction across rows (official avg_entropy)."""
    log_p = logits - logits.logsumexp(-1, keepdim=True)
    avg = log_p.logsumexp(0) - math.log(logits.shape[0])
    avg = avg.clamp(min=torch.finfo(avg.dtype).min)
    return -(avg * avg.exp()).sum(-1)


def clip_outputs(views, model, text_weights, select_frac=0.1):
    """Official get_clip_logits. views: (V, 3, 224, 224) with V = 1 or 64.
    Also returns view 0's own logits (for gate features; not used by TDA)."""
    feats = model.encode_image(views)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    logits = LOGIT_SCALE * feats @ text_weights
    view0_logits = logits[:1]
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
    return feat, logits, float(loss), probs, pred, view0_logits


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
        return {"pos_cache_max_sim": 0.0, "pos_cache_mean_sim": 0.0, "pos_cache_sim_clip_pred": 0.0,
                "cache_fill": 0, "cache_pred": -1, "cache_agrees_with_clip": 0, "neg_cache_fill": neg_fill}
    sims = (feat @ torch.cat(keys, dim=0).T).squeeze(0)
    same_class = [item[0] for item in pos_cache.get(clip_pred, [])]
    sim_clip_pred = float((feat @ torch.cat(same_class, dim=0).T).max()) if same_class else 0.0
    pos_logits = cache_logits(feat, pos_cache, pos_cfg["alpha"], pos_cfg["beta"], num_classes)
    cache_pred = int(pos_logits.argmax(-1))
    return {
        "pos_cache_max_sim": float(sims.max()),
        "pos_cache_mean_sim": float(sims.mean()),
        "pos_cache_sim_clip_pred": sim_clip_pred,       # closest cached image of CLIP's predicted class
        "cache_fill": len(keys),
        "cache_pred": cache_pred,
        "cache_agrees_with_clip": int(cache_pred == clip_pred),
        "neg_cache_fill": neg_fill,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run TDA on a registered dataset.")
    add_dataset_arg(p)
    p.add_argument("--n-views", type=int, default=None, choices=[1, 64],
                   help="Default: the dataset's official setting.")
    p.add_argument("--config", default=None, help="TDA YAML (default: the dataset's official config).")
    p.add_argument("--classnames", default=None, help="Override: JSON list of class names.")
    p.add_argument("--templates", default=None, help="Override: JSON list of templates with {}.")
    p.add_argument("--output", default=None, help="Default: results/<dataset>/tda[_<tag>].json")
    p.add_argument("--tag", default=None, help="Suffix for the default output name, e.g. seed1, 1view.")
    p.add_argument("--vanilla-results", default=None,
                   help="Optional; default results/<dataset>/clip.json if it exists.")
    p.add_argument("--refs-tda", default=str(REPO_ROOT / "refs" / "tda"),
                   help="Official TDA repo (for AugMix ops).")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--half", action="store_true", help="Half precision (CUDA only), as official.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="Shuffle the stream with this seed (official uses a seeded shuffle). Default: dataset order.")
    p.add_argument("--n", type=int, default=None, help="Use only the first N images (smoke tests).")
    p.add_argument("--save-probs", action="store_true", help="Also save probability vectors (.npz).")
    p.add_argument("--model", default="ViT-B-16-quickgelu")
    p.add_argument("--pretrained", default="openai")
    return p.parse_args()


def main():
    args = parse_args()
    spec = get_dataset(args.dataset)
    device = pick_device(args.device)
    seed_everything(args.seed)

    classnames = json.loads(Path(args.classnames).read_text()) if args.classnames else spec.load_classnames()
    templates = json.loads(Path(args.templates).read_text()) if args.templates else spec.load_templates()
    n_views = args.n_views or spec.tda_n_views
    cfg = load_config(args.config or spec.tda_config_path())
    num_classes = len(classnames)
    pos_cfg, neg_cfg = cfg["positive"], cfg["negative"]
    mask_thr = (neg_cfg["mask_threshold"]["lower"], neg_cfg["mask_threshold"]["upper"])

    model, _, tokenizer = load_clip(device, args.model, args.pretrained)
    if args.half and device.type == "cuda":
        model = model.half()
    dtype = next(model.parameters()).dtype

    text_weights = build_text_weights(model, tokenizer, classnames, templates, device)
    make_views = single_view if n_views == 1 else AugMixViews(n_views - 1, load_augmix_ops(args.refs_tda))

    samples = load_samples(spec, shuffle_seed=args.shuffle_seed, n=args.n)
    out = Path(args.output) if args.output else results_path(spec.name, "tda", args.tag)

    vanilla_path = Path(args.vanilla_results) if args.vanilla_results else results_path(spec.name, "clip")
    vanilla = load_results(vanilla_path) if vanilla_path.exists() else {}

    pos_cache, neg_cache = {}, {}
    rows, final_probs_all, clip_probs_all, ids = [], [], [], []

    print(f"TDA | dataset={spec.name} | device={device} | views={n_views} | classes={num_classes} | "
          f"templates={len(templates)} | alpha={pos_cfg['alpha']} beta={pos_cfg['beta']} | images={len(samples)}")

    with torch.no_grad():
        for i, s in enumerate(samples):
            img = Image.open(s.path).convert("RGB")

            t0 = time.perf_counter()
            views = make_views(img)
            t_views = time.perf_counter() - t0

            views = views.to(device, dtype)
            feat, logits, loss, probs, clip_pred, view0_logits = clip_outputs(views, model, text_weights)
            norm_ent = loss / math.log2(num_classes)            # official convention

            stats = gate_stats(feat, pos_cache, neg_cache, pos_cfg, num_classes, clip_pred)

            if pos_cfg["enabled"]:
                update_cache(pos_cache, clip_pred, [feat, loss], pos_cfg["shot_capacity"])
            if neg_cfg["enabled"] and (neg_cfg["entropy_threshold"]["lower"] < norm_ent
                                       < neg_cfg["entropy_threshold"]["upper"]):
                update_cache(neg_cache, clip_pred, [feat, loss, probs], neg_cfg["shot_capacity"])

            final = logits.clone()
            pos_max = neg_max = 0.0
            if pos_cfg["enabled"] and pos_cache:
                pos = cache_logits(feat, pos_cache, pos_cfg["alpha"], pos_cfg["beta"], num_classes)
                final += pos
                pos_max = float(pos.max())
            if neg_cfg["enabled"] and neg_cache:
                neg = cache_logits(feat, neg_cache, neg_cfg["alpha"], neg_cfg["beta"], num_classes, mask_thr)
                final -= neg
                neg_max = float(neg.max())

            pred = int(final.argmax(-1))
            sync(device)
            elapsed = time.perf_counter() - t0

            v = vanilla.get(s.id, {})
            vanilla_correct = v.get("correct")
            correct = pred == s.label

            row = {
                "id": s.id,
                "label": s.label,
                "pred": pred,
                "correct": correct,
                "label_name": classnames[s.label],
                "pred_name": classnames[pred],
                **s.meta,
                "stream_index": i,
                "n_views": n_views,
                # CLIP before the caches (what TDA's caches are updated with), and TDA's output
                **uncertainty_stats(logits, "clip"),
                **uncertainty_stats(view0_logits, "clip_view0"),
                **uncertainty_stats(final, "tda"),
                "tda_changed_clip": pred != clip_pred,
                # cache state before this image, and how much each cache moved the logits
                "pos_cache_max_sim": stats["pos_cache_max_sim"],
                "pos_cache_mean_sim": stats["pos_cache_mean_sim"],
                "pos_cache_sim_clip_pred": stats["pos_cache_sim_clip_pred"],
                "cache_fill": stats["cache_fill"],
                "cache_agrees_with_clip": stats["cache_agrees_with_clip"],
                "cache_pred": stats["cache_pred"],
                "neg_cache_fill": stats["neg_cache_fill"],
                "pos_cache_logit_max": pos_max,
                "neg_cache_logit_max": neg_max,
                "vanilla_correct": vanilla_correct,
                "improved_over_vanilla": (correct and not vanilla_correct) if vanilla_correct is not None else None,
                "view_seconds": t_views,
                "model_seconds": elapsed - t_views,
                "seconds": elapsed,
            }
            rows.append(row)
            if args.save_probs:
                final_probs_all.append(final.float().softmax(-1).squeeze(0).cpu().numpy())
                clip_probs_all.append(probs.float().squeeze(0).cpu().numpy())
                ids.append(s.id)

            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/{len(samples)}")

    acc = float(np.mean([r["correct"] for r in rows]))
    secs = np.array([r["seconds"] for r in rows])
    view_secs = np.array([r["view_seconds"] for r in rows])
    summary = {"accuracy": acc, "n_views": n_views, "median_ms": float(np.median(secs) * 1000),
               "config": str(args.config or spec.tda_config_path() or "pilot_default")}
    save_results(rows, out, args, summary)
    if args.save_probs:
        np.savez_compressed(out.with_suffix(".probs.npz"), ids=np.array(ids),
                            final_probs=np.stack(final_probs_all), clip_probs=np.stack(clip_probs_all))

    print(f"\nTDA accuracy: {acc:.3f}  ({len(rows)} images, {n_views} view(s))")
    if spec.targets.get("tda") is not None:
        print(f"Target (registry): {spec.targets['tda']:.2f}%")
    if vanilla:
        improved = sum(1 for r in rows if r["improved_over_vanilla"])
        regressed = sum(1 for r in rows if r["vanilla_correct"] and not r["correct"])
        print(f"Improved over vanilla: {improved} | regressed: {regressed}")
    print(f"Time per image: median {np.median(secs)*1000:.1f} ms "
          f"(views {np.median(view_secs)*1000:.1f} ms, model {np.median(secs - view_secs)*1000:.1f} ms)")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()