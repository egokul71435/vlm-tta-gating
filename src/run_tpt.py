import json
import copy
import torch
import torch.nn as nn
import numpy as np
import open_clip
from pathlib import Path
from PIL import Image
from torchvision import transforms

MANIFEST = Path("data/manifests/corrupted_manifest.json")
VANILLA_RESULTS = Path("results/vanilla_clip_results.json")
OUTPUT_PATH = Path("results/tpt_results.json")
DEVICE = "mps"

CLASS_LABELS = [
    "Shih-Tzu", "Rhodesian ridgeback", "beagle", "English foxhound",
    "Australian terrier", "border terrier", "golden retriever",
    "Old English sheepdog", "Samoyed", "dingo",
]

N_CTX = 4          # number of learnable context tokens
N_AUGMENTATIONS = 63  # + 1 original = 64 views, matching TPT paper
TOP_K_FRACTION = 0.10  # keep top 10% most confident views
LR = 0.005
N_STEPS = 1        # TPT paper uses a single optimization step per image


def entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


class PromptLearner(nn.Module):
    """Learnable context vectors prepended to class name embeddings.
    Simplification: unified context shared across all classes (as in
    original TPT), not per-class conditioned context."""

    def __init__(self, classnames, clip_model, tokenizer, n_ctx=4):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.token_embedding.weight.dtype
        device = clip_model.token_embedding.weight.device

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name}, a type of dog." for name in classnames]
        tokenized_prompts = tokenizer(prompts).to(device) 


        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])       # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # class name + EOS + pad
        self.tokenized_prompts = tokenized_prompts
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = getattr(clip_model, "attn_mask", None)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
    

def get_base_transform():
    # Matches TPT's base_transform: random crop + flip, applied before
    # CLIP's own preprocess (tensor conversion + normalization). Verified
    # against refs/tpt/data/datautils.py's AugMixAugmenter — for
    # ImageNet-scale OOD sets (the closest analog to this pilot), TPT's
    # augmix flag is False, so it also reduces to plain crop+flip.
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ])


def main():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    model = model.to(DEVICE).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)  # only the prompt context gets trained

    base_transform = get_base_transform()

    prompt_learner = PromptLearner(CLASS_LABELS, model, tokenizer, n_ctx=N_CTX).to(DEVICE)
    text_encoder = TextEncoder(model).to(DEVICE)
    initial_ctx_state = copy.deepcopy(prompt_learner.state_dict())

    tokenized_prompts = prompt_learner.tokenized_prompts.to(DEVICE)

    with open(MANIFEST) as f:
        images = json.load(f)
        # images = images[:20] # temporary: test on a small subset first
    with open(VANILLA_RESULTS) as f:
        vanilla_results = {r["corrupted_path"]: r for r in json.load(f)}

    top_k = max(1, int((N_AUGMENTATIONS + 1) * TOP_K_FRACTION))
    results = []

    for i, entry in enumerate(images):
        # Reset prompt to its initial state before every image — TPT adapts
        # per-image and does not carry state across images.
        prompt_learner.load_state_dict(initial_ctx_state)
        optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=LR)

        img = Image.open(entry["corrupted_path"]).convert("RGB")

        views = [preprocess(base_transform(img)) for _ in range(N_AUGMENTATIONS + 1)]
        views_tensor = torch.stack(views).to(DEVICE)

        with torch.no_grad():
            image_features_all = model.encode_image(views_tensor)
            image_features_all = image_features_all / image_features_all.norm(dim=-1, keepdim=True)

        for _ in range(N_STEPS):
            prompts = prompt_learner()
            text_features = text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features_all @ text_features.T
            probs = torch.softmax(logits, dim=-1)

            with torch.no_grad():
                entropies = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                selected_idx = torch.topk(-entropies, k=top_k).indices

            avg_prob = probs[selected_idx].mean(dim=0)
            loss = -(avg_prob * torch.log(avg_prob.clamp_min(1e-12))).sum()
            # print(f"  loss: {loss.item():.4f}")  # check if loss actually decreases

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prompts = prompt_learner()
            text_features = text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features_all @ text_features.T
            probs = torch.softmax(logits, dim=-1)
            final_probs = probs[selected_idx].mean(dim=0).cpu().numpy()

        pred_idx = int(np.argmax(final_probs))
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
            "confidence": float(np.max(final_probs)),
            "entropy": entropy(final_probs),
            "vanilla_correct": vanilla_correct,
            "improved_over_vanilla": (correct and not vanilla_correct) if vanilla_correct is not None else None,
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    accuracy = sum(r["correct"] for r in results) / len(results)
    improved_count = sum(1 for r in results if r["improved_over_vanilla"])
    print(f"\nTPT accuracy: {accuracy:.3f}")
    print(f"Images improved over vanilla CLIP: {improved_count}")
    print(f"Saved results to {OUTPUT_PATH}")

    regressed_count = sum(1 for r in results if r["vanilla_correct"] and not r["correct"])
    print(f"Images regressed vs vanilla: {regressed_count}")
    

if __name__ == "__main__":
    main()