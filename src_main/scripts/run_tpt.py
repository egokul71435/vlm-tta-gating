"""Run TPT (Test-time Prompt Tuning) over a manifest of images.

Mirrors the official implementation (refs/tpt: tpt_classification.py, clip/custom_clip.py):
  - 4 learnable context vectors initialized from the token embeddings of "a photo of a"
  - prompts are "[ctx x4] {class name}."
  - 64 views: view 0 = Resize(224, bicubic) + CenterCrop(224); 63 augmented views
    (random resized crop + flip; AugMix with --augmix, as official uses on the
    cross-domain datasets)
  - lowest-entropy 10% of views (6 of 64) selected from the pre-tuning logits
  - loss = entropy of the averaged prediction over the selected views
  - AdamW, lr 5e-3, 1 step; prompt and optimizer reset for every image
  - final prediction from view 0 only, with the tuned prompt

Legacy flags reproduce the original Imagewoof pilot (random init, dog suffix,
random view 0, averaged selected views):
  --ctx-init random --suffix ", a type of dog." --view0 random --final selected_avg

Gate features come from the PRE-tuning logits. In official mode those are plain
zero-shot predictions ("a photo of a {class}."), so they cost nothing extra.
"""

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PILOT_CLASSNAMES = [
    "Shih-Tzu", "Rhodesian ridgeback", "beagle", "English foxhound",
    "Australian terrier", "border terrier", "golden retriever",
    "Old English sheepdog", "Samoyed", "dingo",
]


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


def load_json_list(path, default):
    if path is None:
        return default
    with open(path) as f:
        return json.load(f)


def load_augmix_ops(refs_tpt):
    ops_dir = Path(refs_tpt) / "data"
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
    transforms.RandomResizedCrop(224),          # default scale (0.08, 1.0), as official
    transforms.RandomHorizontalFlip(),
])


class TPTViews:
    """Official AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=...)."""

    def __init__(self, n_aug=63, aug_list=None, view0="center", severity=1):
        self.n_aug = n_aug
        self.aug_list = aug_list or []
        self.view0 = view0
        self.severity = severity

    def _augmented(self, img):
        x_orig = PRE_AUGMENT(img)
        x_processed = TO_TENSOR(x_orig)
        if not self.aug_list:
            return x_processed
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
        first = TO_TENSOR(BASE_TRANSFORM(img)) if self.view0 == "center" else self._augmented(img)
        return torch.stack([first] + [self._augmented(img) for _ in range(self.n_aug)])


# ----------------------------------------------------------------------------
# Prompt learner and text encoder
# ----------------------------------------------------------------------------

class PromptLearner(nn.Module):
    """Official TPT PromptLearner (class token at the end), adapted to open_clip."""

    def __init__(self, model, tokenizer, classnames, ctx_init="a_photo_of_a", n_ctx=4, suffix="."):
        super().__init__()
        device = model.token_embedding.weight.device
        dtype = model.token_embedding.weight.dtype
        ctx_dim = model.ln_final.weight.shape[0]
        classnames = [c.replace("_", " ") for c in classnames]

        if ctx_init and ctx_init.lower() != "random":
            ctx_words = ctx_init.replace("_", " ")
            n_ctx = len(ctx_words.split(" "))
            with torch.no_grad():
                emb = model.token_embedding(tokenizer([ctx_words]).to(device)).type(dtype)
            ctx_vectors = emb[0, 1:1 + n_ctx, :].detach().clone()
            prefix_text = ctx_words
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)   # CPU, as in the old pilot
            nn.init.normal_(ctx_vectors, std=0.02)
            ctx_vectors = ctx_vectors.to(device)
            prefix_text = " ".join(["X"] * n_ctx)

        prompts = [f"{prefix_text} {name}{suffix}" for name in classnames]
        tokenized = tokenizer(prompts).to(device)
        with torch.no_grad():
            emb = model.token_embedding(tokenized).type(dtype)

        self.register_buffer("token_prefix", emb[:, :1, :])            # SOS
        self.register_buffer("token_suffix", emb[:, 1 + n_ctx:, :])    # class name, ".", EOS, padding
        self.register_buffer("ctx_init_state", ctx_vectors.clone())
        self.ctx = nn.Parameter(ctx_vectors)
        self.tokenized_prompts = tokenized
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx

    def reset(self):
        with torch.no_grad():
            self.ctx.copy_(self.ctx_init_state)

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.attn_mask = getattr(model, "attn_mask", None)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)       # open_clip is batch-first
        x = self.ln_final(x)
        return x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


# ----------------------------------------------------------------------------
# Official TPT math
# ----------------------------------------------------------------------------

def softmax_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


def avg_entropy(logits):
    log_p = logits - logits.logsumexp(-1, keepdim=True)
    avg = log_p.logsumexp(0) - np.log(logits.shape[0])
    avg = avg.clamp(min=torch.finfo(avg.dtype).min)
    return -(avg * avg.exp()).sum(-1)


def text_features(prompt_learner, text_encoder):
    t = text_encoder(prompt_learner(), prompt_learner.tokenized_prompts)
    return t / t.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run TPT over an image manifest.")
    p.add_argument("--manifest", default="data/manifests/corrupted_manifest.json")
    p.add_argument("--output", default="results/tpt_results.json")
    p.add_argument("--vanilla-results", default="results/vanilla_clip_results.json")
    p.add_argument("--tda-results", default="results/tda_results.json",
                   help="Optional; used to print the TPT/TDA cost ratio.")
    p.add_argument("--classnames", default=None, help="JSON list of class names (default: pilot dogs).")
    p.add_argument("--ctx-init", default="a_photo_of_a", help='Official default; "random" for the legacy pilot.')
    p.add_argument("--suffix", default=".", help='Text after the class name; legacy pilot: ", a type of dog."')
    p.add_argument("--view0", default="center", choices=["center", "random"])
    p.add_argument("--final", default="view0", choices=["view0", "selected_avg"])
    p.add_argument("--augmix", action="store_true", help="AugMix views (official on cross-domain datasets).")
    p.add_argument("--refs-tpt", default="refs/tpt", help="Official TPT repo (for AugMix ops).")
    p.add_argument("--n-views", type=int, default=64)
    p.add_argument("--selection-p", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--probe-k", type=int, default=8, help="Views used for the cheap probe features.")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--amp", action="store_true", help="Mixed precision with gradient scaling (CUDA only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=None, help="Use only the first N images (smoke tests).")
    p.add_argument("--save-probs", action="store_true", help="Also save probability vectors (.npz).")
    p.add_argument("--model", default="ViT-B-16-quickgelu")
    p.add_argument("--pretrained", default="openai")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    use_amp = args.amp and device == "cuda"

    # Seed BEFORE creating the model: model init draws random numbers, so this
    # order is what makes the legacy flags reproduce the old pilot exactly.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    classnames = load_json_list(args.classnames, PILOT_CLASSNAMES)

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    prompt_learner = PromptLearner(model, tokenizer, classnames, args.ctx_init, suffix=args.suffix).to(device)
    text_encoder = TextEncoder(model).to(device)
    logit_scale = model.logit_scale.exp().detach()

    aug_list = load_augmix_ops(args.refs_tpt) if args.augmix else []
    make_views = TPTViews(args.n_views - 1, aug_list, args.view0)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", init_scale=1000) if use_amp else None

    with open(args.manifest) as f:
        images = json.load(f)
    if args.n is not None:
        images = images[: args.n]

    vanilla = {}
    if args.vanilla_results and Path(args.vanilla_results).exists():
        with open(args.vanilla_results) as f:
            vanilla = {r["corrupted_path"]: r for r in json.load(f)}

    print(f"TPT | device={device} | amp={use_amp} | views={args.n_views} | augmix={args.augmix} | "
          f"ctx_init={args.ctx_init} | view0={args.view0} | final={args.final} | images={len(images)}")

    results, final_probs_all, zs_probs_all, paths = [], [], [], []

    for i, entry in enumerate(images):
        path = entry.get("corrupted_path", entry.get("path"))
        prompt_learner.reset()
        optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=args.lr)

        img = Image.open(path).convert("RGB")
        t0 = time.perf_counter()

        views = make_views(img)
        t_views = time.perf_counter() - t0

        views = views.to(device)
        with torch.no_grad(), autocast():
            img_feats = model.encode_image(views)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        synchronize(device)
        t_encoded = time.perf_counter() - t0

        # --- test-time tuning ---
        selected_idx, init_logits = None, None
        for _ in range(args.steps):
            with autocast():
                logits = logit_scale * img_feats @ text_features(prompt_learner, text_encoder).T
            logits = logits.float()
            if selected_idx is None:
                init_logits = logits.detach()
                ent = softmax_entropy(init_logits)
                selected_idx = ent.argsort()[: int(logits.shape[0] * args.selection_p)]
            loss = avg_entropy(logits[selected_idx])
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # --- prediction with the tuned prompt ---
        with torch.no_grad(), autocast():
            tuned_text = text_features(prompt_learner, text_encoder)
            if args.final == "view0":
                final_probs = (logit_scale * img_feats[:1] @ tuned_text.T).float().softmax(-1)[0]
            else:
                probs = (logit_scale * img_feats @ tuned_text.T).float().softmax(-1)
                final_probs = probs[selected_idx].mean(0)
        final_probs = final_probs.cpu().numpy()
        pred_idx = int(final_probs.argmax())
        synchronize(device)
        elapsed = time.perf_counter() - t0

        # --- gate features from the pre-tuning logits (zero-shot in official mode) ---
        init_ent = softmax_entropy(init_logits)
        init_pred = init_logits.argmax(-1)
        k = min(args.probe_k, init_logits.shape[0])
        zs_probs = init_logits[0].softmax(-1).cpu().numpy()

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
            # pre-tuning view statistics (all views): costly, need the 64 views
            "view_entropy_std": float(init_ent.std()),
            "view_entropy_mean": float(init_ent.mean()),
            "view_agreement": float((init_pred == init_pred[0]).float().mean()),
            # cheap probe: first k views only
            "probe_entropy_mean": float(init_ent[:k].mean()),
            "probe_entropy_std": float(init_ent[:k].std()),
            "probe_agreement": float((init_pred[:k] == init_pred[0]).float().mean()),
            "n_views": args.n_views,
            "view_seconds": t_views,
            "encode_seconds": t_encoded - t_views,
            "tune_predict_seconds": elapsed - t_encoded,
            "view_feature_seconds": t_encoded,      # views + encoding (kept for compatibility)
            "seconds": elapsed,
        })
        if args.save_probs:
            final_probs_all.append(final_probs)
            zs_probs_all.append(zs_probs)
            paths.append(path)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    if args.save_probs:
        np.savez_compressed(out.with_suffix(".probs.npz"), paths=np.array(paths),
                            final_probs=np.stack(final_probs_all), zeroshot_view0_probs=np.stack(zs_probs_all))

    acc = np.mean([r["correct"] for r in results])
    secs = np.array([r["seconds"] for r in results])
    print(f"\nTPT accuracy: {acc:.3f}  ({len(results)} images)")
    if vanilla:
        improved = sum(1 for r in results if r["improved_over_vanilla"])
        regressed = sum(1 for r in results if r["vanilla_correct"] and not r["correct"])
        print(f"Improved over vanilla: {improved} | regressed: {regressed}")
    print(f"Time per image: median {np.median(secs)*1000:.1f} ms "
          f"(views {np.median([r['view_seconds'] for r in results])*1000:.1f}, "
          f"encode {np.median([r['encode_seconds'] for r in results])*1000:.1f}, "
          f"tune+predict {np.median([r['tune_predict_seconds'] for r in results])*1000:.1f})")
    tda_path = Path(args.tda_results) if args.tda_results else None
    if tda_path and tda_path.exists():
        with open(tda_path) as f:
            tda_secs = [r["seconds"] for r in json.load(f) if "seconds" in r]
        if tda_secs:
            print(f"TPT / TDA cost ratio (median): {np.median(secs) / np.median(tda_secs):.1f}x")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()