import json
import torch
import numpy as np
import open_clip
from pathlib import Path
from PIL import Image

MANIFEST = Path("data/manifests/corrupted_manifest.json")
OUTPUT_CSV = Path("results/vanilla_clip_results.json")
EMBEDDINGS_OUTPUT = Path("results/vanilla_clip_embeddings.npz")
DEVICE = "mps"

# Imagenette classes, in a fixed order matching CLASS_NAMES from select_images.py
CLASS_LABELS = [
    "Shih-Tzu", "Rhodesian ridgeback", "beagle", "English foxhound",
    "Australian terrier", "border terrier", "golden retriever",
    "Old English sheepdog", "Samoyed", "dingo",
]

def entropy(probs):
    # Shannon entropy of a probability distribution, in nats
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def main():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    model = model.to(DEVICE).eval()

    # build and encode text prompts once
    prompts = [f"a photo of a {label}" for label in CLASS_LABELS]
    text_tokens = tokenizer(prompts).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    with open(MANIFEST) as f:
        images = json.load(f)

    results = []
    embedding_store = {}

    with torch.no_grad():
        for i, entry in enumerate(images):
            img = Image.open(entry["corrupted_path"]).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

            image_features = model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding_store[entry["corrupted_path"]] = image_features.squeeze(0).cpu().numpy()

            cos_sims = (image_features @ text_features.T).squeeze(0)  # raw cosine similarity to each class prompt
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * cos_sims
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            sorted_probs = np.sort(probs)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1])       # top-1 minus top-2 probability
            text_alignment = float(cos_sims.max().item())           # closeness to the nearest class prompt

            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_LABELS[pred_idx]
            correct = pred_label == entry["label"]

            results.append({
                "corrupted_path": entry["corrupted_path"],
                "true_label": entry["label"],
                "corruption_type": entry["corruption_type"],
                "predicted_label": pred_label,
                "correct": correct,
                "confidence": float(np.max(probs)),
                "entropy": entropy(probs),
                "margin": margin,
                "text_alignment": text_alignment,
            })

            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1}/{len(images)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w") as f:
        json.dump(results, f, indent=2)

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\nVanilla CLIP zero-shot accuracy: {accuracy:.3f}")
    print(f"Saved results to {OUTPUT_CSV}")

    # quick breakdown by corruption type
    for ctype in ["gaussian_blur", "gaussian_noise"]:
        subset = [r for r in results if r["corruption_type"] == ctype]
        acc = sum(r["correct"] for r in subset) / len(subset)
        print(f"  {ctype}: {acc:.3f} ({len(subset)} images)")

    EMBEDDINGS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBEDDINGS_OUTPUT,
        **{path.replace("/", "__"): emb for path, emb in embedding_store.items()}
    )
    print(f"Saved {len(embedding_store)} image embeddings to {EMBEDDINGS_OUTPUT}")

if __name__ == "__main__":
    main()