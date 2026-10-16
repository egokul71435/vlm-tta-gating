# Cost-Aware Routing Between Test-Time Adaptation Methods for VLMs

Test-time adaptation (TTA) recovers CLIP accuracy under distribution shift, but methods differ widely in cost. TPT (test-time prompt tuning) runs 64 augmented views plus a gradient step per image; TDA (training-free dynamic adapter) can run on a single view with a small cache. This project routes each image: cheap TDA by default, expensive TPT only when a nearly free gate predicts the image needs it.

## Status

- **Both methods verified against the official code** (logic matches; differences are documented run settings). Published-number reproduction is pending GPU access.
- **Pilot (Imagewoof, 1,000 images) re-run with the verified implementations and shared prompts.** A single free signal, CLIP's top-2 margin, predicts which images need TPT (AUC ~0.86, permutation p < 0.005). On this pilot TPT and TDA differ by under a point, so the accuracy at stake is small.
- **Next:** repo restructure, ImageNet-A/-R loaders and zero-shot check, then benchmark runs (ImageNet OOD suite, then the 10 cross-domain datasets). Target: CVPR 2027.

## Pilot setup

- **Data:** [Imagewoof](https://github.com/fastai/imagenette) validation images (10 dog breeds), 1,000 images, half gaussian blur and half gaussian noise (scikit-image implementation of ImageNet-C severity conventions). Switched from Imagenette after CLIP was too accurate there for corruption to matter.
- **Model:** CLIP ViT-B/16 (open_clip `ViT-B-16-quickgelu`, OpenAI weights), on Apple M2 Max (MPS).
- **Class names:** official ImageNet names for the 10 classes (from the TPT repo), e.g. "Shih Tzu", "Beagle".
- **Prompt:** `a photo of a {}.` for all three methods (the same prompt official TPT starts from).
- **TDA:** 1 view; pilot α=1.0, β=5.0 (Imagewoof has no official config; untuned); manifest order plus 3 shuffle seeds.
- **TPT:** official defaults (see below), crop + flip views (official setting for ImageNet-derived data).

## Pilot results

| Method | Accuracy | Notes |
|---|---|---|
| Vanilla CLIP | 59.0% | blur 63.6%, noise 54.4% |
| TDA (1 view) | 60.1% | 59.9 / 60.6 / 60.5% across 3 shuffle seeds |
| TPT (official) | 61.0% | +2.0 over vanilla (51 improved, 31 regressed) |
| Oracle (best of TPT/TDA per image) | 64.8% | 3.8 pts above TPT |

**Per-image outcomes:** both right 563, both wrong 352, only TPT right 47, only TDA right 38. Only the 47 "TPT-only" images (4.7%) are worth paying TPT's cost for; 38 of them (81%) are noise-corrupted.

### Cost (median per image, M2 Max)

| Strategy | Accuracy | Time per image |
|---|---|---|
| Always TDA (1 view) | 60.1% | ~25 ms |
| Always TPT | 61.0% | ~600 ms (~24x TDA): views 47, encoding 469, tuning + prediction 83 |
| Cost-aware oracle (TPT only on the 4.7% that need it) | 64.8% | ~54 ms (~11x cheaper than always-TPT) |

On this 10-class dataset, encoding TPT's 64 views dominates its cost. TPT's tuning step runs the text encoder over one prompt per class, so on 1,000-class datasets it will be far heavier. Laptop timings are hardware-specific; GPU costs will be measured per dataset.

### Routing

The gate scores each image by how likely it is to need TPT (TPT right, TDA wrong) using cheap signals from vanilla CLIP and TDA's cache; the top-scoring X% go to TPT, the rest to TDA. Cost accounting is conservative: TDA runs on every image, routed images pay for TPT on top.

| Gate | AUC | Avg. precision (base rate 0.047) | Acc @20% to TPT | Acc @30% | Matches always-TPT at |
|---|---|---|---|---|---|
| Low top-2 margin (single threshold) | 0.861 | 0.156 | 60.5% | 60.8% | 25% |
| Low confidence (single threshold) | 0.853 | 0.164 | 60.6% | 60.9% | 40% |
| High entropy (single threshold) | 0.816 | 0.142 | 60.3% | 60.9% | 45% |
| Logistic regression, cheap scalars | 0.844 ± 0.008 | 0.142 | 60.8% | 60.9% | 25% |
| Gradient boosting, cheap scalars | 0.835 ± 0.009 | 0.149 | 61.2% | 60.8% | 15% |
| Logistic regression, PCA-5 CLIP embeddings | 0.688 | 0.100 | 60.9% | 61.3% | 25% |
| Random routing | 0.5 | 0.047 | 60.3% | 60.4% | — |

Learned gates use 5-fold cross-validation averaged over 5 fold splits; thresholds need no training.

- **The ranking signal is real:** permutation test on the logistic-regression AUC (200 label shuffles): observed 0.847 vs. shuffled mean 0.501, max 0.635; **p < 0.005**.
- **Top-2 margin carries it.** Vanilla-uncertainty features alone reach AUC 0.853 (slightly above all features, 0.844); TDA cache features alone reach 0.702 but add nothing on top of uncertainty. Margin has by far the largest coefficient (−2.39), and a margin threshold alone is the best single rule.
- **Accuracy gains are small here.** TPT beats TDA by only 0.9 pts, so routing gains ~0.5–0.9 pts over random routing and roughly matches TPT at 15–40% TPT. Those budget numbers are noisy (a few images move them); the benchmarks, with larger method gaps, will decide whether routing matters in practice.

## Why these results differ from the first pilot

The first pilot (kept in `results/pilot_v1/`) reported TDA 62.5%, TPT 64.0% and a 5.8-pt oracle gap. Three things changed:

1. **Shared prompts (the main reason).** The first pilot gave the methods different text prompts: vanilla CLIP used "a photo of a {class}", while TDA and TPT used "... {class}, a type of dog." Telling CLIP the classes are dog breeds raises accuracy by itself, so part of TDA's and TPT's apparent gain over vanilla came from the better prompt rather than from adaptation. With one prompt for all three, TDA's gain over vanilla drops from 4.0 to 1.1 pts.
2. **TPT now follows the official implementation.** The first pilot's TPT used a random prompt initialization, a random first view, and averaged six augmented views for its prediction. Official TPT initializes from "a photo of a", uses a center-cropped first view, and predicts from that view alone. (Legacy flags reproduce the first pilot exactly: 64.0%.)
3. **Official class names** ("Shih Tzu" rather than "Shih-Tzu", etc.), matching what the benchmarks will use.

TDA's cache logic was already equivalent to the official code; its rewrite reproduces the first pilot exactly (62.5%) under the old settings.

**Superseded results:** the gate experiments from the first pilot (disagreement-case classifiers, the embedding MLP, cost-shaped RL, and the claim that TDA cache agreement adds unique signal) were run on the old implementations and prompts. They are not reported here; RL and the cascade will be re-run on the benchmarks.

## Implementations and verification

Both were compared line by line against the official repositories ([TPT](https://github.com/azshue/TPT), [TDA](https://github.com/kdiaaa/tda)) and reviewed independently.

**TDA** (`src_main/scripts/run_tda.py`)
- Matches official: CLIP pseudo-label updates both caches before the image's own prediction; entropy-ordered queues (3 positive / 2 negative per class); negative cache gated on entropy in nats divided by log2(classes) (official convention, so the normalized value tops out near 0.69); fixed logit scale 100; text weights averaged over a template list.
- `--n-views 1` (as described in the TDA paper; official code on ImageNet and the cross-domain sets) or `--n-views 64` (center crop + 63 AugMix views, lowest-entropy 10% averaged; official code on ImageNet-A, -V2, -R and Sketch, though the paper does not mention it).
- Deviations: fp32 by default (official fp16; `--half` on CUDA); image order is a seeded shuffle rather than official's exact sequence (order moves pilot accuracy by a few tenths); pilot α/β untuned.

**TPT** (`src_main/scripts/run_tpt.py`)
- Matches official: 4 context vectors from "a photo of a"; prompts "[context] {class}."; view 0 = resize + center crop; 63 random-resized-crop + flip views (AugMix with `--augmix`, official on cross-domain sets); lowest-entropy 6 of 64 views; loss = entropy of their averaged prediction; AdamW lr 5e-3, 1 step; reset per image; final prediction from view 0. The open_clip text path matches open_clip's own encoding exactly (max difference 0.0).
- Encodes views once instead of every step (same predictions, slightly cheaper).
- Deviations: fp32 by default (official fp16 with a gradient scaler; `--amp` on CUDA); seed 42 vs. official 0 (only the random views differ).
- Gate features come from the pre-tuning logits, which in default mode are zero-shot predictions, so they cost nothing extra.

**Still to verify:** published accuracy on ImageNet-A and -R (ViT-B/16 targets: TDA with 64 views 60.11% / 80.24%; TPT 54.77% / 77.06%), and GPU cost per dataset.

## Setup

```bash
conda create -n vlm-tta python=3.11 -y
conda activate vlm-tta
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())"

# official implementations, used for verification and AugMix operations (gitignored)
git clone https://github.com/azshue/TPT refs/tpt
git clone https://github.com/kdiaaa/tda refs/tda
```

## Running the pilot

```bash
python src/run_vanilla_clip.py
python src_main/scripts/run_tda.py --classnames data/classnames/imagewoof.json --templates data/templates/photo.json
python src_main/scripts/run_tpt.py --classnames data/classnames/imagewoof.json
python src/build_win_labels.py
python src/compute_oracle.py
python src/cost_aware_routing.py
```

Data preparation (`src/select_images.py`, `src/apply_corruptions.py`) and the class-name alignment step have already been run; their outputs live in `data/manifests/`, `data/classnames/` and `data/templates/`.

| Step | Script | Output |
|---|---|---|
| Select base images | `src/select_images.py` | `data/manifests/base_images.json` |
| Apply corruptions | `src/apply_corruptions.py` | `data/corrupted/`, `data/manifests/corrupted_manifest.json` |
| Vanilla CLIP | `src/run_vanilla_clip.py` | `results/vanilla_clip_results.json`, embeddings |
| TDA | `src_main/scripts/run_tda.py` | `results/tda_results.json` |
| TPT | `src_main/scripts/run_tpt.py` | `results/tpt_results.json` |
| Merge outcomes + gate features | `src/build_win_labels.py` | `results/win_labels.json` |
| Oracle | `src/compute_oracle.py` | console |
| Routing curves, feature analysis, threshold baselines | `src/cost_aware_routing.py` | `results/cost_curve.png`, `results/threshold_baseline.png` |

A restructure into a configurable package (dataset registry, device and path options, one results format) is in progress.

## Notes

- Results from the first pilot are preserved in `results/pilot_v1/`; earlier gate-search scripts are in `src/archive/`.
- `imagecorruptions` was replaced with a direct scikit-image implementation due to packaging issues.