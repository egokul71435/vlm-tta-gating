"""Run ZERO (Farina et al., NeurIPS 2024) on a registered dataset.

Mirrors the official implementation (refs/zero: ttas/zero.py, ttas/base.py,
data/augs.py, utils/tools.py):
  - 64 views: the original image (Resize 224 bicubic + CenterCrop 224) plus 63
    RandomResizedCrop(224) + horizontal flip views. No AugMix, on any dataset.
  - zero-shot CLIP logits with the prompt "a photo of a {class}." (logit scale 100)
  - keep the 30% most confident views (lowest entropy of the softmax)
  - "zero temperature": each kept view gives a one-hot vote for its top class;
    predict the most-voted class
  - ties: the first lower-ranked view whose top class is tied decides (official
    greedy_break); if none does, the tied class with the highest logit in the
    next-ranked view (the official fallback's intent; its code indexes with ~ints)
  - no backpropagation, no state carried across images

One deliberate difference, logged both ways: official ZERO runs under fp16 autocast,
so its "zero temperature" is fp16's epsilon (~9.8e-4) applied to unscaled cosine
logits. Views whose top-2 classes are close then cast fractional votes. The main
prediction here (pred / zero_*) uses exact one-hot votes, as the paper defines;
zero_fp16_* emulates the official code's fp16 voting on the same views (fp16
cosine logits, view ranking on fp16 logits / fp16 temperature, soft votes), so
either can be reported (--fp16-votes makes it the main prediction). The emulation
covers the voting arithmetic only: CLIP's forward pass here runs in fp32 unless
--half is used on CUDA, so it is close to, not bit-exact with, the official run.

Also logs the ingredients of CAS's gate (arXiv 2609.08367), which uses the same
kind of view statistics: agreement between the lowest-entropy 10% of views and the
original view, by top-1 label and by cosine similarity of probability vectors.
CAS's code is not released yet; with --save-view-probs the selected views'
probabilities are saved so its exact score can be recomputed later. Note: CAS's
paper generates its views with AugMix; these are ZERO's crop + flip views.

Examples:
  python src_main/scripts/run_zero.py --dataset imagewoof_pilot
  python src_main/scripts/run_zero.py --dataset imagenet_a --save-probs --save-view-probs
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from tta.common import (build_text_weights, load_clip, load_results, pick_device, results_path,
                        save_results, seed_everything, softmax_entropy, sync, uncertainty_stats)
from tta.data import load_samples
from tta.registry import add_dataset_arg, get_dataset

LOGIT_SCALE = 100.0          # CLIP's temperature 0.01, as in official ZERO
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ----------------------------------------------------------------------------
# Views (official data/augs.py: MultiViewCropAndFlip)
# ----------------------------------------------------------------------------

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])
TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])
CROP_AND_FLIP = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
])


def zero_views(img, n_views=64):
    views = [TO_TENSOR(BASE_TRANSFORM(img))]
    views += [TO_TENSOR(CROP_AND_FLIP(img)) for _ in range(n_views - 1)]
    return torch.stack(views)


# ----------------------------------------------------------------------------
# ZERO and CAS ingredients from the views' zero-shot logits
# ----------------------------------------------------------------------------

FP16_EPS = float(torch.finfo(torch.float16).eps)   # 9.765625e-4


def zero_from_logits(logits, gamma=0.3, fp16_votes=False, temp16=None):
    """Official ZERO on the zero-shot logits of all views (row 0 = the original image).
    fp16_votes=True emulates the official code under fp16 autocast: cosine logits in
    fp16, views ranked by the entropy of fp16 (cosine / temperature) with the
    temperature itself computed in fp16 (temp16, ~0.010002), and votes =
    softmax(cosine / fp16 eps) instead of one-hot."""
    n, c = logits.shape
    if fp16_votes:
        cos16 = (logits / LOGIT_SCALE).half()            # fp16 cosine logits, as under autocast
        logits = (cos16 / temp16).float()                 # official: (l / self.model.temp); ranking + kept-view probs
        order = softmax_entropy(logits).argsort()
        tb_logits = cos16.float()                         # official greedy_break uses the unscaled fp16 cosines
    else:
        order = softmax_entropy(logits).argsort()
        tb_logits = logits
    k = max(1, int(n * gamma))
    sel = order[:k]
    if fp16_votes:
        votes = (cos16[sel] / FP16_EPS).float().softmax(-1).sum(0)
    else:
        votes = torch.bincount(logits[sel].argmax(-1), minlength=c).float()
    top = votes.max()
    ties = (votes == top).nonzero().flatten().tolist()
    pred, tie_rule = ties[0], "none"
    if len(ties) > 1:
        rest = tb_logits[order[k:]].argmax(-1).tolist()
        hit = next((t for t in rest if t in ties), None)
        if hit is not None:
            pred, tie_rule = hit, "greedy"
        else:
            row = tb_logits[order[min(k, n - 1)]]
            pred, tie_rule = max(ties, key=lambda t: float(row[t])), "fallback"
    ranked = votes.sort(descending=True).values
    second = ranked[1] if c > 1 else torch.tensor(0.0)
    return {
        "pred": int(pred), "votes": votes, "k": k, "tie_rule": tie_rule,
        "agreement": float(top / k),                    # share of kept views voting for the winner
        "vote_margin": float((ranked[0] - second) / k),
        "sel_avg_probs": logits[sel].softmax(-1).mean(0),
    }


def cas_components(logits, rho=0.1):
    """CAS gate ingredients: the lowest-entropy rho of views vs. the original view,
    computed with view 0 allowed in the selected set and with it excluded (it
    trivially agrees with itself; the CAS paper's choice is unconfirmed until its
    code is released). Also returns the top-(k+1) views' probabilities and indices,
    enough to recompute either variant later."""
    probs = logits.softmax(-1)
    order = softmax_entropy(logits).argsort()
    k = max(1, int(logits.shape[0] * rho))
    p0 = probs[0]

    def agree(idx):
        label = (probs[idx].argmax(-1) == p0.argmax()).float().mean()
        cos = torch.nn.functional.cosine_similarity(probs[idx], p0.unsqueeze(0), dim=-1).mean()
        return float(label), float(cos)

    incl = agree(order[:k])
    excl = agree(order[order != 0][:k])
    top = order[: k + 1]
    return {"label_agree": incl[0], "prob_cos": incl[1],
            "label_agree_excl0": excl[0], "prob_cos_excl0": excl[1],
            "view_probs": probs[top], "view_idx": top}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run ZERO on a registered dataset.")
    add_dataset_arg(p)
    p.add_argument("--classnames", default=None, help="Override: JSON list of class names.")
    p.add_argument("--prompt", default="a photo of a {}.", help='Official: "a photo of a {}." (ctx_init a_photo_of_a).')
    p.add_argument("--output", default=None, help="Default: results/<dataset>/zero[_<tag>].json")
    p.add_argument("--tag", default=None, help="Suffix for the default output name, e.g. smoke.")
    p.add_argument("--vanilla-results", default=None,
                   help="Optional; default results/<dataset>/clip.json if it exists.")
    p.add_argument("--n-views", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.3, help="Share of most confident views kept (official 0.3).")
    p.add_argument("--cas-rho", type=float, default=0.1, help="Share of views CAS compares (0.1 per the CAS paper).")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--half", action="store_true", help="Half precision (CUDA only).")
    p.add_argument("--fp16-votes", action="store_true",
                   help="Use the official code's fp16 soft votes as the main prediction (both are logged).")
    p.add_argument("--seed", type=int, default=1, help="Official ZERO scripts use seed 1.")
    p.add_argument("--n", type=int, default=None, help="Use only the first N images (smoke tests).")
    p.add_argument("--save-probs", action="store_true",
                   help="Also save ZERO's vote shares, kept-view average and view-0 probabilities (.npz).")
    p.add_argument("--save-view-probs", action="store_true",
                   help="Also save the CAS-selected views' probabilities (large on 1,000-class sets).")
    p.add_argument("--model", default="ViT-B-16-quickgelu")
    p.add_argument("--pretrained", default="openai")
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_views < 4:
        raise SystemExit("--n-views must be at least 4 (ZERO keeps 30% of the views).")
    spec = get_dataset(args.dataset)
    device = pick_device(args.device)
    seed_everything(args.seed)

    classnames = json.loads(Path(args.classnames).read_text()) if args.classnames else spec.load_classnames()
    model, _, tokenizer = load_clip(device, args.model, args.pretrained)
    if args.half and device.type == "cuda":
        model = model.half()
    dtype = next(model.parameters()).dtype
    text_weights = build_text_weights(model, tokenizer, classnames, [args.prompt], device)   # (D, C)
    temp16 = 1 / model.logit_scale.detach().half().exp()   # official fp16 temperature (~0.010002)

    # ZERO resets per image, so stream order does not matter; dataset order is kept.
    samples = load_samples(spec, n=args.n)
    out = Path(args.output) if args.output else results_path(spec.name, "zero", args.tag)
    vanilla_path = Path(args.vanilla_results) if args.vanilla_results else results_path(spec.name, "clip")
    vanilla = load_results(vanilla_path) if vanilla_path.exists() else {}

    print(f"ZERO | dataset={spec.name} | device={device} | views={args.n_views} | gamma={args.gamma} | "
          f"prompt='{args.prompt}' | classes={len(classnames)} | images={len(samples)}")

    rows, ids, votes_all, avg_all, view0_all, cas_views_all, cas_idx_all = [], [], [], [], [], [], []
    with torch.no_grad():
        for i, s in enumerate(samples):
            img = Image.open(s.path).convert("RGB")

            t0 = time.perf_counter()
            views = zero_views(img, args.n_views)
            t_views = time.perf_counter() - t0

            feats = model.encode_image(views.to(device, dtype))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = (LOGIT_SCALE * feats @ text_weights).float()
            sync(device)
            t_encoded = time.perf_counter() - t0

            zero = zero_from_logits(logits, args.gamma, fp16_votes=args.fp16_votes, temp16=temp16)
            sync(device)
            elapsed = time.perf_counter() - t0

            other = zero_from_logits(logits, args.gamma, fp16_votes=not args.fp16_votes, temp16=temp16)
            hard, soft = (other, zero) if args.fp16_votes else (zero, other)
            cas = cas_components(logits, args.cas_rho)
            view_ent = softmax_entropy(logits)
            view_pred = logits.argmax(-1)
            zs_pred = int(view_pred[0])
            pred = zero["pred"]
            correct = pred == s.label
            v = vanilla.get(s.id, {})
            vanilla_correct = v.get("correct")

            rows.append({
                "id": s.id,
                "label": s.label,
                "pred": pred,
                "correct": correct,
                "label_name": classnames[s.label],
                "pred_name": classnames[pred],
                **s.meta,
                "n_views": args.n_views,
                # zero-shot original view, and ZERO's own confidence signals
                **uncertainty_stats(logits[0], "zs"),
                "zero_agreement": zero["agreement"],
                "zero_vote_margin": zero["vote_margin"],
                "zero_tie_rule": zero["tie_rule"],
                "zero_changed_zs": pred != zs_pred,
                "zero_votes": "fp16" if args.fp16_votes else "one-hot",
                # both vote rules on the same views (paper: one-hot; official code on GPU: fp16 soft)
                "zero_hard_pred": hard["pred"],
                "zero_hard_correct": hard["pred"] == s.label,
                "zero_hard_tie_rule": hard["tie_rule"],
                "zero_fp16_pred": soft["pred"],
                "zero_fp16_correct": soft["pred"] == s.label,
                "zero_fp16_agreement": soft["agreement"],
                "zero_fp16_tie_rule": soft["tie_rule"],
                **uncertainty_stats(torch.log(zero["sel_avg_probs"].clamp_min(1e-12)), "zero_avg"),
                # all-view statistics, and CAS gate ingredients
                "view_entropy_mean": float(view_ent.mean()),
                "view_entropy_std": float(view_ent.std()),
                "view_agreement": float((view_pred == view_pred[0]).float().mean()),
                "cas_label_agree": cas["label_agree"],
                "cas_prob_cos": cas["prob_cos"],
                "cas_label_agree_excl0": cas["label_agree_excl0"],
                "cas_prob_cos_excl0": cas["prob_cos_excl0"],
                "vanilla_correct": vanilla_correct,
                "improved_over_vanilla": (correct and not vanilla_correct) if vanilla_correct is not None else None,
                "view_seconds": t_views,
                "encode_seconds": t_encoded - t_views,
                "vote_seconds": elapsed - t_encoded,
                "seconds": elapsed,
            })
            if args.save_probs:
                ids.append(s.id)
                votes_all.append((zero["votes"] / zero["k"]).cpu().numpy())
                avg_all.append(zero["sel_avg_probs"].cpu().numpy())
                view0_all.append(logits[0].softmax(-1).cpu().numpy())
            if args.save_view_probs:
                cas_views_all.append(cas["view_probs"].cpu().numpy().astype(np.float16))
                cas_idx_all.append(cas["view_idx"].cpu().numpy())

            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/{len(samples)}")

    acc = float(np.mean([r["correct"] for r in rows]))
    acc_hard = float(np.mean([r["zero_hard_correct"] for r in rows]))
    acc_fp16 = float(np.mean([r["zero_fp16_correct"] for r in rows]))
    secs = np.array([r["seconds"] for r in rows])
    ties = sum(r["zero_tie_rule"] != "none" for r in rows)
    fallbacks = sum(r["zero_tie_rule"] == "fallback" for r in rows)
    ties_hard = sum(r["zero_hard_tie_rule"] != "none" for r in rows)
    ties_fp16 = sum(r["zero_fp16_tie_rule"] != "none" for r in rows)
    summary = {"accuracy": acc, "accuracy_one_hot_votes": acc_hard, "accuracy_fp16_votes": acc_fp16,
               "main_votes": "fp16" if args.fp16_votes else "one-hot",
               "gamma": args.gamma, "ties": ties, "tie_fallbacks": fallbacks,
               "ties_one_hot_votes": ties_hard, "ties_fp16_votes": ties_fp16,
               "median_ms": float(np.median(secs) * 1000)}
    save_results(rows, out, args, summary)
    if args.save_probs:
        np.savez_compressed(out.with_suffix(".probs.npz"), ids=np.array(ids), zero_votes=np.stack(votes_all),
                            zero_sel_avg_probs=np.stack(avg_all), zeroshot_view0_probs=np.stack(view0_all))
    if args.save_view_probs:
        np.savez_compressed(out.with_suffix(".viewprobs.npz"), ids=np.array([r["id"] for r in rows]),
                            cas_view_probs=np.stack(cas_views_all), cas_view_idx=np.stack(cas_idx_all))

    print(f"\nZERO accuracy: {acc:.3f}  ({len(rows)} images; {ties} ties broken, {fallbacks} by the fallback rule)")
    print(f"  one-hot votes (paper): {acc_hard:.3f} ({ties_hard} ties) | "
          f"fp16 soft votes (official code on GPU): {acc_fp16:.3f} ({ties_fp16} ties)")
    if vanilla:
        improved = sum(1 for r in rows if r["improved_over_vanilla"])
        regressed = sum(1 for r in rows if r["vanilla_correct"] and not r["correct"])
        print(f"Improved over vanilla: {improved} | regressed: {regressed}")
    print(f"Time per image: median {np.median(secs)*1000:.1f} ms "
          f"(views {np.median([r['view_seconds'] for r in rows])*1000:.1f}, "
          f"encode {np.median([r['encode_seconds'] for r in rows])*1000:.1f}, "
          f"vote {np.median([r['vote_seconds'] for r in rows])*1000:.1f})")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()