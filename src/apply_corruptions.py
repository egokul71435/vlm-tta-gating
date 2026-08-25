import json
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.filters import gaussian

random_seed = 42  # already baked into manifest's shuffle order

INPUT_MANIFEST = Path("data/manifests/base_images.json")
OUTPUT_DIR = Path("data/corrupted")
OUTPUT_MANIFEST = Path("data/manifests/corrupted_manifest.json")
SEVERITY = 3

# sigma values per severity level, matching ImageNet-C's gaussian_blur convention
BLUR_SIGMAS = {1: 0.5, 2: 0.75, 3: 1.0, 4: 1.25, 5: 1.5}
# std-dev values per severity level, matching ImageNet-C's gaussian_noise convention
NOISE_STDS = {1: 0.04, 2: 0.08, 3: 0.12, 4: 0.16, 5: 0.20}

def apply_gaussian_blur(img_np, severity):
    sigma = BLUR_SIGMAS[severity]
    img_float = img_np.astype(np.float64) / 255.0
    blurred = gaussian(img_float, sigma=sigma, channel_axis=-1)
    return np.clip(blurred * 255, 0, 255).astype(np.uint8)

def apply_gaussian_noise(img_np, severity):
    std = NOISE_STDS[severity]
    img_float = img_np.astype(np.float64) / 255.0
    noise = np.random.normal(0, std, img_float.shape)
    noisy = img_float + noise
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

def main():
    np.random.seed(42)  # reproducible noise

    with open(INPUT_MANIFEST) as f:
        images = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    midpoint = len(images) // 2
    blur_images = images[:midpoint]
    noise_images = images[midpoint:]

    manifest = []

    for group, corruption_name, corrupt_fn in [
        (blur_images, "gaussian_blur", apply_gaussian_blur),
        (noise_images, "gaussian_noise", apply_gaussian_noise),
    ]:
        for entry in group:
            img_path = Path(entry["path"])
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)

            corrupted_np = corrupt_fn(img_np, SEVERITY)
            corrupted_img = Image.fromarray(corrupted_np)

            out_filename = f"{corruption_name}_{img_path.stem}.jpg"
            out_path = OUTPUT_DIR / out_filename
            corrupted_img.save(out_path)

            manifest.append({
                "original_path": str(img_path),
                "corrupted_path": str(out_path),
                "class_id": entry["class_id"],
                "label": entry["label"],
                "corruption_type": corruption_name,
                "severity": SEVERITY,
            })

    with open(OUTPUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Applied corruptions to {len(manifest)} images")
    print(f"  {len(blur_images)} -> gaussian_blur")
    print(f"  {len(noise_images)} -> gaussian_noise")
    print(f"Saved manifest to {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()