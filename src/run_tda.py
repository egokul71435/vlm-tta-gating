import json
import torch
import numpy as np
import open_clip
from pathlib import Path
from PIL import Image

MANIFEST = Path("data/manifests/corrupted_manifest.json")
VANILLA_RESULTS = Path("results/vanilla_clip_results.json")
OUTPUT_PATH = Path("results/tda_results.json")
DEVICE = "mps"

CLASS_LABELS = [
    "Shih-Tzu", "Rhodesian ridgeback", "beagle", "English foxhound",
    "Australian terrier", "border terrier", "golden retriever",
    "Old English sheepdog", "Samoyed", "dingo",
]

# --- Positive cache hyperparameters ---
# shot_capacity=3 is constant across all of TDA's official dataset configs.
# alpha/beta ARE dataset-specific in the reference (e.g. imagenet.yaml uses
# 2.0/5.0, food101.yaml uses 1.0/1.0, imagenet_r.yaml uses 1.0/8.0).
# Imagewoof has no official TDA config, so these are an arbitrary but
# documented choice, not a verified default.
POS_CACHE_CAPACITY = 3
POS_ALPHA = 1.0
POS_BETA = 5.0

# --- Negative cache hyperparameters ---
# Near-constant across TDA's official configs, per refs/tda.
NEG_CACHE_CAPACITY = 2
NEG_ALPHA = 0.117
NEG_BETA = 1.0
NEG_ENTROPY_THRESHOLD = (0.2, 0.5)   # normalized entropy range to qualify for negative caching
NEG_MASK_THRESHOLD = (0.03, 1.0)     # prob range defining "ruled out" classes

# NOTE: the exact negative-mask construction below is a best-effort
# reconstruction from thresholds only, not verified line-for-line against
# refs/tda's actual mask logic. Confirm before treating results as final.


def entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


class PositiveCache:
    """Per-class queue of (feature, entropy) pairs. New entries replace the
    highest-entropy (least confident) existing entry in that class once
    the queue is full, keeping the cache biased toward confident examples."""

    def __init__(self, num_classes, capacity):
        self.capacity = capacity
        self.cache = {c: [] for c in range(num_classes)}

    def update(self, class_idx, feature, entropy_val):
        entries = self.cache[class_idx]
        if len(entries) < self.capacity:
            entries.append((feature, entropy_val))
        else:
            max_idx = max(range(len(entries)), key=lambda i: entries[i][1])
            if entropy_val < entries[max_idx][1]:
                entries[max_idx] = (feature, entropy_val)

    def compute_logits(self, image_feature, num_classes, beta):
        logits = torch.zeros(num_classes, device=image_feature.device)
        for class_idx, entries in self.cache.items():
            if not entries:
                continue
            feats = torch.stack([e[0] for e in entries])  # (n_entries, dim)
            sims = feats @ image_feature  # cosine similarity, features already normalized
            affinities = torch.exp(-beta * (1 - sims))
            logits[class_idx] = affinities.sum()
        return logits


class NegativeCache:
    """Per-predicted-class queue of (feature, entropy, negative_mask) triples.
    negative_mask marks which classes this image likely does NOT belong to,
    used to subtract evidence for those classes on future similar images."""

    def __init__(self, num_classes, capacity, entropy_threshold, mask_threshold):
        self.capacity = capacity
        self.num_classes = num_classes
        self.entropy_lower, self.entropy_upper = entropy_threshold
        self.mask_lower, self.mask_upper = mask_threshold
        self.cache = {c: [] for c in range(num_classes)}

    def maybe_update(self, pred_idx, feature, entropy_val, probs):
        max_entropy = np.log2(self.num_classes)  # refs/tda normalizes by log2, not ln
        norm_entropy = entropy_val / max_entropy
        if not (self.entropy_lower < norm_entropy < self.entropy_upper):
            return  # only moderately-uncertain predictions qualify

        mask = ((probs > self.mask_lower) & (probs < self.mask_upper)).float()

        entries = self.cache[pred_idx]
        item = (feature, entropy_val, mask)
        if len(entries) < self.capacity:
            entries.append(item)
        else:
            entries.sort(key=lambda e: e[1])
            if entropy_val < entries[-1][1]:
                entries[-1] = item

    def compute_logits(self, image_feature, beta):
        logits = torch.zeros(self.num_classes, device=image_feature.device)
        for entries in self.cache.values():
            if not entries:
                continue
            feats = torch.stack([e[0] for e in entries])
            masks = torch.stack([e[2] for e in entries])
            sims = feats @ image_feature
            affinities = torch.exp(-beta * (1 - sims))
            logits += (affinities.unsqueeze(1) * masks).sum(dim=0)
        return logits


def main():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    model = model.to(DEVICE).eval()

    prompts = [f"a photo of a {label}, a type of dog." for label in CLASS_LABELS]
    text_tokens = tokenizer(prompts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    with open(MANIFEST) as f:
        images = json.load(f)
        # images = images[:20] # temporary: test on a small subset first
    with open(VANILLA_RESULTS) as f:
        vanilla_results = {r["corrupted_path"]: r for r in json.load(f)}

    pos_cache = PositiveCache(len(CLASS_LABELS), POS_CACHE_CAPACITY)
    neg_cache = NegativeCache(len(CLASS_LABELS), NEG_CACHE_CAPACITY,
                               NEG_ENTROPY_THRESHOLD, NEG_MASK_THRESHOLD)
    results = []

    with torch.no_grad():
        for i, entry in enumerate(images):
            img = Image.open(entry["corrupted_path"]).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

            image_features = model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_feature = image_features.squeeze(0)

            logit_scale = model.logit_scale.exp()
            clip_logits = (logit_scale * image_feature @ text_features.T)
            clip_probs = torch.softmax(clip_logits, dim=-1)

            # Update caches with the model's own (pseudo-labeled) prediction
            # BEFORE computing this sample's cache-augmented logits — matches
            # refs/tda's ordering, where a sample's own feature can already
            # influence its own cache-boosted prediction.
            pseudo_label_idx = int(torch.argmax(clip_probs).item())
            pseudo_entropy = entropy(clip_probs.cpu().numpy())
            pos_cache.update(pseudo_label_idx, image_feature.clone(), pseudo_entropy)
            neg_cache.maybe_update(pseudo_label_idx, image_feature.clone(), pseudo_entropy, clip_probs)

            pos_logits = pos_cache.compute_logits(image_feature, len(CLASS_LABELS), POS_BETA)
            neg_logits = neg_cache.compute_logits(image_feature, NEG_BETA)
            combined_logits = clip_logits + POS_ALPHA * pos_logits - NEG_ALPHA * neg_logits
            combined_probs = torch.softmax(combined_logits, dim=-1).cpu().numpy()

            pred_idx = int(np.argmax(combined_probs))
            pred_label = CLASS_LABELS[pred_idx]
            correct = pred_label == entry["label"]

            vanilla = vanilla_results.get(entry["corrupted_path"], {})
            vanilla_correct = vanilla.get("correct", None)

            results.append({
                "corrupted_path": entry["corrupted_path"],
                "true_label": entry["label"],
                "corruption_type": entry["corruption_type"],
                "predicted_label": pred_label,
                "correct": correct,
                "confidence": float(np.max(combined_probs)),
                "entropy": entropy(combined_probs),
                "vanilla_correct": vanilla_correct,
                "improved_over_vanilla": (correct and not vanilla_correct) if vanilla_correct is not None else None,
            })

            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/{len(images)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    accuracy = sum(r["correct"] for r in results) / len(results)
    improved_count = sum(1 for r in results if r["improved_over_vanilla"])
    regressed_count = sum(1 for r in results if r["vanilla_correct"] and not r["correct"])
    print(f"\nTDA accuracy: {accuracy:.3f}")
    print(f"Images improved over vanilla: {improved_count}")
    print(f"Images regressed vs vanilla: {regressed_count}")
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()