"""Zero-shot CLIP on a registered dataset: the reproduction check for the
loaders, and the source of the cheap "vanilla" gate features.

Text weights use the dataset's templates (the 80-template ensemble on the
ImageNet family, matching the published CLIP numbers; "a photo of a {}." on the
pilot). Images use open_clip's standard preprocessing (resize 224 bicubic +
center crop), i.e. the same single view TDA uses.

Examples:
  python src_main/scripts/run_vanilla_clip.py --dataset imagewoof_pilot
  python src_main/scripts/run_vanilla_clip.py --dataset imagenet_a
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tta.common import (build_text_weights, load_clip, pick_device, results_path, save_results,
                        seed_everything, uncertainty_stats)
from tta.data import load_samples
from tta.registry import add_dataset_arg, get_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot CLIP on a registered dataset.")
    add_dataset_arg(p)
    p.add_argument("--classnames", default=None, help="Override: JSON list of class names.")
    p.add_argument("--templates", default=None, help="Override: JSON list of templates with {}.")
    p.add_argument("--output", default=None, help="Default: results/<dataset>/clip[_<tag>].json")
    p.add_argument("--tag", default=None)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=None, help="Use only the first N images (smoke tests).")
    p.add_argument("--no-embeddings", action="store_true", help="Skip saving image embeddings.")
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

    model, preprocess, tokenizer = load_clip(device, args.model, args.pretrained)
    text_weights = build_text_weights(model, tokenizer, classnames, templates, device)   # (D, C)
    logit_scale = model.logit_scale.exp()

    samples = load_samples(spec, n=args.n)
    out = Path(args.output) if args.output else results_path(spec.name, "clip", args.tag)

    print(f"CLIP zero-shot | dataset={spec.name} | device={device} | classes={len(classnames)} | "
          f"templates={len(templates)} | images={len(samples)}")

    rows, embeddings, ids = [], [], []
    with torch.no_grad():
        for i, s in enumerate(samples):
            img = Image.open(s.path).convert("RGB")
            feats = model.encode_image(preprocess(img).unsqueeze(0).to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cos_sims = (feats @ text_weights).squeeze(0)        # cosine similarity to each class
            logits = logit_scale * cos_sims
            pred = int(logits.argmax())

            rows.append({
                "id": s.id,
                "label": s.label,
                "pred": pred,
                "correct": pred == s.label,
                "label_name": classnames[s.label],
                "pred_name": classnames[pred],
                **s.meta,
                **uncertainty_stats(logits, "clip"),
                "text_alignment": float(cos_sims.max()),     # closeness to the nearest class prompt
            })
            if not args.no_embeddings:
                embeddings.append(feats.squeeze(0).float().cpu().numpy())
                ids.append(s.id)

            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/{len(samples)}")

    acc = float(np.mean([r["correct"] for r in rows]))
    save_results(rows, out, args, {"accuracy": acc, "templates": len(templates)})

    print(f"\nCLIP zero-shot accuracy: {acc:.3f}  ({len(rows)} images)")
    if spec.targets.get("clip") is not None:
        print(f"Target (registry): {spec.targets['clip']:.2f}%")
    groups = defaultdict(list)
    for r in rows:
        if r.get("corruption_type"):
            groups[r["corruption_type"]].append(r["correct"])
    for name, vals in sorted(groups.items()):
        print(f"  {name}: {np.mean(vals):.3f} ({len(vals)} images)")
    print(f"Saved results to {out}")

    if not args.no_embeddings:
        emb_path = out.with_name(out.stem + "_embeddings.npz")
        np.savez_compressed(emb_path, ids=np.array(ids), embeddings=np.stack(embeddings))
        print(f"Saved {len(ids)} image embeddings to {emb_path}")


if __name__ == "__main__":
    main()