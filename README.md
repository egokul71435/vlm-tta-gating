# Cost-Aware Routing Between Test-Time Adaptation Methods for VLMs

Test-time adaptation (TTA) recovers CLIP accuracy under distribution shift, but methods differ widely in cost. TPT (test-time prompt tuning) runs 64 augmented views plus a gradient step per image; TDA (training-free dynamic adapter) can run on a single view with a small cache. This project routes each image: cheap TDA by default, expensive TPT only when a nearly free gate predicts the image needs it.

## Status

- **Both methods verified against the official code** (logic matches; differences are documented run settings). Published-number reproduction is pending GPU access.
- **Pilot (Imagewoof, 1,000 images) re-run with the verified implementations and shared prompts.** A single free signal, CLIP's top-2 margin, predicts which images need TPT (AUC ~0.86, permutation p < 0.005). On this pilot TPT and TDA differ by under a point, so the accuracy at stake is small.
- **Learned and RL gates compared on the corrected pilot:** a small MLP scorer (AUC 0.815) and cost-shaped REINFORCE (useful only in a narrow penalty range) never beat the simple scorers; accuracy-only RL and disagreement-trained MLPs do not rank the images that need TPT. A margin threshold remains the method; the others are comparisons. Attention-based gates are a possible follow-up if the benchmarks show the simple gates leaving signal behind.
- **Repo restructured:** any registered dataset runs end to end with `./run.sh <dataset>`. The pilot reproduces exactly through the new pipeline (identical per-image predictions, outcome counts and routing numbers).
- **Compute:** access to NCSA DeltaAI through the lab's ACCESS allocation (Delta GPU hours are used up).
- **Next:** ImageNet-A data preparation and zero-shot check (target 49.89%), then TDA (1 and 64 views) and TPT on ImageNet-A, ZERO as a second expensive method, then the rest of the ImageNet OOD suite and the 10 cross-domain datasets. Target: CVPR 2027.

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
| Always TDA (1 view) | 60.1% | ~25–35 ms |
| Always TPT | 61.0% | ~600–770 ms (~22–24x TDA); in the faster run: views 47, encoding 469, tuning + prediction 83 |
| Cost-aware oracle (TPT only on the 4.7% that need it) | 64.8% | ~11x cheaper than always-TPT |

Ranges span two runs of the same code; laptop timings vary with load and temperature, while the ratio stays stable. On this 10-class dataset, encoding TPT's 64 views dominates its cost. TPT's tuning step runs the text encoder over one prompt per class, so on 1,000-class datasets it will be far heavier. Paper costs will be measured on GPU, on an otherwise idle machine, after warm-up.

### Routing

The gate scores each image by how likely it is to need TPT (TPT right, TDA wrong) using cheap signals from vanilla CLIP and TDA's cache; the top-scoring X% go to TPT, the rest to TDA. Cost accounting is conservative: TDA runs on every image, routed images pay for TPT on top.

| Gate | AUC | Avg. precision (base rate 0.047) | Acc @20% to TPT | Acc @30% | Matches always-TPT at |
|---|---|---|---|---|---|
| Low top-2 margin (single threshold) | 0.861 | 0.156 | 60.5% | 60.8% | 25% |
| Low confidence (single threshold) | 0.853 | 0.164 | 60.6% | 60.9% | 40% |
| High entropy (single threshold) | 0.816 | 0.142 | 60.3% | 60.9% | 45% |
| Logistic regression, cheap scalars | 0.844 ± 0.008 | 0.142 | 60.8% | 60.9% | 25% |
| Gradient boosting, cheap scalars | 0.835 ± 0.009 | 0.149 | 61.2% | 60.8% | 15% |
| MLP, cheap scalars | 0.815 ± 0.007 | 0.130 | 61.0% | 60.6% | 15% |
| Logistic regression, PCA-5 CLIP embeddings | 0.688 ± 0.014 | 0.100 | 60.9% | 61.3% | 25% |
| MLP, PCA-5 CLIP embeddings | 0.646 ± 0.019 | 0.080 | 60.6% | 60.6% | not by 50% |
| MLP, cheap scalars + PCA-5 embeddings | 0.783 ± 0.012 | 0.132 | 60.5% | 60.5% | not by 50% |
| Random routing | 0.5 | 0.047 | 60.3% | 60.4% | — |

Learned gates use 5-fold cross-validation averaged over 5 fold splits; thresholds need no training. The MLP has one hidden layer of 32 units, L2 weight decay, and positives up-weighted by the negative-to-positive ratio (the same balance as the logistic regression).

- **The ranking signal is real:** permutation test on the logistic-regression AUC (200 label shuffles): observed 0.847 vs. shuffled mean 0.501, max 0.635; **p < 0.005**.
- **Top-2 margin carries it.** Vanilla-uncertainty features alone reach AUC 0.853 (slightly above all features, 0.844); TDA cache features alone reach 0.702 but add nothing on top of uncertainty. Margin has by far the largest coefficient (−2.39), and a margin threshold alone is the best single rule.
- **Nonlinear gates don't help here.** The MLP ranks below the linear model on every input set (0.815 vs. 0.844 on cheap scalars), and adding embeddings lowers its AUC further (0.783). With 47 positive images, extra capacity mostly fits noise; the signal is close to one number. The MLP's real test is the benchmarks, with far more images per dataset and several shift types.
- **Accuracy gains are small here.** TPT beats TDA by only 0.9 pts, so routing gains ~0.5–0.9 pts over random routing and roughly matches TPT at 15–40% TPT. Those budget numbers are noisy (a few images move them); the benchmarks, with larger method gaps, will decide whether routing matters in practice.

### How the gates are trained and tested

All learned gates use stratified 5-fold cross-validation over the 1,000 pilot images: each fold trains on 800 images and tests on the other 200, so every image is scored exactly once by a model that never saw it. AUC and the budget curves are computed over those 1,000 held-out scores. The procedure is repeated over 5 fold splits (3 for the RL gates) and averaged; the ± values are the spread across splits. Feature scaling and PCA are fit on each training fold only.

| Gate | Trained on (per fold) | Tested on (per fold) | Notes |
|---|---|---|---|
| Thresholds (margin, confidence, entropy, cache rule) | — (no training) | all 1,000 | Direction fixed in advance: more uncertain → TPT |
| Logistic regression, gradient boosting, MLP | 800 (~38 need TPT) | 200 (~9–10 need TPT) | Target: needs TPT (TPT right, TDA wrong) |
| Winner MLP | ~68 disagreement images among the 800 (~38 TPT-only, ~30 TDA-only) | 200 | Never sees an image where the methods agree |
| REINFORCE, accuracy-only (λ = 0) | 800, but only the ~68 disagreement images move the policy | 200 | Where both methods agree, both actions score the same |
| REINFORCE, cost-shaped (λ > 0) | 800; every image contributes | 200 | The penalty pushes every image toward TDA |
| Feature analysis coefficients | all 1,000 (one fit) | — | Descriptive only; the ablations and the permutation test use the cross-validation above |

> **Limitation: the pilot is small.** Each gate learns from about 38 positive images per fold (47 in total), so differences of a few tenths of a point are a few images, and the ± values reflect only fold-to-fold variation, understating the real uncertainty. Everything here is also trained and tested within one dataset; whether a gate transfers across datasets (train on some benchmarks, test on another) is untested. The benchmarks, with thousands of images per dataset, are where these comparisons count.

### RL and RL-adjacent gates

Same cheap features, folds and cost accounting as above (`cost_aware_routing.py --rl`; 5-fold, 3 fold splits). Every image's outcome is known for both methods, so the gate is a contextual bandit with full feedback; REINFORCE uses the mean reward of the two actions as its baseline.

**Cost-shaped REINFORCE** (reward = TDA right, or TPT right − λ when choosing TPT), greedy decisions:

| λ | Images to TPT | Accuracy | Cost (ms/image) | AUC of P(TPT) |
|---|---|---|---|---|
| 0 (accuracy-only) | 69.8% | 61.1% | 606 | 0.43 |
| 0.02 | 18.3% | 61.0% | 182 | 0.78 |
| 0.05 | 11.1% | 60.5% | 125 | 0.74 |
| ≥ 0.10 | 0–0.2% | 60.1% (always TDA) | 35–37 | — |

**As ranking scores over the same budgets:**

| Gate | AUC | AP | Acc @10% | Acc @20% | Acc @30% | Matches always-TPT at |
|---|---|---|---|---|---|---|
| Logistic regression, cheap scalars (reference) | 0.843 | 0.140 | 60.2% | 60.7% | 60.8% | 25% |
| Accuracy-only REINFORCE | 0.432 | 0.041 | 60.3% | 60.5% | 60.6% | not by 50% |
| Winner MLP (disagreements only), cheap scalars | 0.465 | 0.043 | 60.1% | 60.2% | 60.3% | not by 50% |
| Winner MLP (disagreements only), PCA-5 embeddings | 0.588 | 0.056 | 60.4% | 60.7% | 60.7% | 35% |

- **Cost-shaped RL only works in a narrow band of λ.** At λ = 0.02 it matches always-TPT with 18% of images sent to TPT (182 vs. 819 ms per image in this run, about 4.5x cheaper), level with the scorers at that budget (60.7–61.2%). Above about 0.1, it never routes: the expected advantage of TPT over TDA on any image rarely exceeds λ, because only 4.7% of images need TPT and 3.8% need TDA.
- **Accuracy-only RL learns "mostly TPT"**, not which images need it. It sends 70% of images to TPT, roughly matching always-TPT accuracy at ~17x TDA's cost, and its probabilities rank needs-TPT images worse than chance.
- **The winner MLP answers the wrong question.** Trained on the 85 images where the methods disagree, it predicts which method wins given a disagreement, not whether one will occur, so it cannot pick out those images among all 1,000.
- **Conclusion:** no RL or RL-adjacent gate beats a single margin threshold, and each RL budget needs its own training run and λ, while a scorer covers every budget with one threshold. Differences of 0.3 pts are 3 images on this pilot. RL stays as a comparison; the benchmarks decide whether it is worth revisiting.
- **Why:** with both outcomes known and one decision per image, the cost-shaped objective (accuracy − λ × cost) is maximized by routing whenever the predicted advantage of TPT over TDA exceeds λ. A supervised scorer estimates that directly; REINFORCE estimates it through sampled actions and noisier gradients, so at best it matches. The cost-shaped objective stays as the paper's framing of the problem.

### Future consideration: attention-based gates

Not built yet. Worth trying only if, on the benchmarks, the MLP or gradient boosting clearly beat the margin threshold, i.e. there is nonlinear signal the simple gates miss.

- **What it would attend over** (in order of promise): CLIP's top-k class text embeddings (which classes the image is confused between, and how similar they are, beyond the single margin number); TDA's cache entries (a learned look-up in place of TDA's fixed similarity kernel); recent images in the stream (context about the current shift); CLIP's patch tokens.
- **Cost:** the first is free (computable from saved image embeddings and text weights, no reruns); the cache version needs TDA rerun with its cache contents logged (cheap with 1 view); patch tokens need re-encoding.
- **Risks:** too little data on the pilot (about 85 informative images), and every extra component in the gate is something reviewers will ask us to justify; a threshold doing the work is a strength for a cost-focused paper.

## Why these results differ from the first pilot

The first pilot (kept in `results/pilot_v1/`) reported TDA 62.5%, TPT 64.0% and a 5.8-pt oracle gap. Three things changed:

1. **Shared prompts (the main reason).** The first pilot gave the methods different text prompts: vanilla CLIP used "a photo of a {class}", while TDA and TPT used "... {class}, a type of dog." Telling CLIP the classes are dog breeds raises accuracy by itself, so part of TDA's and TPT's apparent gain over vanilla came from the better prompt rather than from adaptation. With one prompt for all three, TDA's gain over vanilla drops from 4.0 to 1.1 pts.
2. **TPT now follows the official implementation.** The first pilot's TPT used a random prompt initialization, a random first view, and averaged six augmented views for its prediction. Official TPT initializes from "a photo of a", uses a center-cropped first view, and predicts from that view alone. (Legacy flags reproduce the first pilot exactly: 64.0%.)
3. **Official class names** ("Shih Tzu" rather than "Shih-Tzu", etc.), matching what the benchmarks will use.

TDA's cache logic was already equivalent to the official code; its rewrite reproduces the first pilot exactly (62.5%) under the old settings.

**Superseded results:** the gate experiments from the first pilot (disagreement-case classifiers, the embedding MLP, cost-shaped RL, and the claim that TDA cache agreement adds unique signal) were run on the old implementations and prompts. The RL, cost-shaped RL and winner-MLP gates have been re-run on the corrected pilot (above), without the first pilot's TPT 64-view features, which cost nearly all of TPT's view encoding and so cannot be part of a cheap gate. The cascade will be built on the benchmarks.

## Implementations and verification

Both were compared line by line against the official repositories ([TPT](https://github.com/azshue/TPT), [TDA](https://github.com/kdiaaa/tda)) and reviewed independently.

**TDA** (`src_main/scripts/run_tda.py`)
- Matches official: CLIP pseudo-label updates both caches before the image's own prediction; entropy-ordered queues (3 positive / 2 negative per class); negative cache gated on entropy in nats divided by log2(classes) (official convention, so the normalized value tops out near 0.69); fixed logit scale 100; text weights averaged over a template list.
- Views follow the dataset's official setting by default: 1 view (as described in the TDA paper; official code on ImageNet and the cross-domain sets) or 64 views (center crop + 63 AugMix views, lowest-entropy 10% averaged; official code on ImageNet-A, -V2, -R and Sketch, though the paper does not mention it). `--n-views` overrides.
- Deviations: fp32 by default (official fp16; `--half` on CUDA); image order is a seeded shuffle rather than official's exact sequence (order moves pilot accuracy by a few tenths); pilot α/β untuned.

**TPT** (`src_main/scripts/run_tpt.py`)
- Matches official: 4 context vectors from "a photo of a"; prompts "[context] {class}."; view 0 = resize + center crop; 63 random-resized-crop + flip views (AugMix on the cross-domain sets, as official; `--augmix` / `--no-augmix` override); lowest-entropy 6 of 64 views; loss = entropy of their averaged prediction; AdamW lr 5e-3, 1 step; reset per image; final prediction from view 0. The open_clip text path matches open_clip's own encoding exactly (max difference 0.0).
- Encodes views once instead of every step (same predictions, slightly cheaper).
- Deviations: fp32 by default (official fp16 with a gradient scaler; `--amp` on CUDA); seed 42 vs. official 0 (only the random views differ).
- Gate features come from the pre-tuning logits, which in default mode are zero-shot predictions, so they cost nothing extra.

**Still to verify:** zero-shot CLIP and published TDA/TPT accuracy on ImageNet-A and -R (ViT-B/16 targets: CLIP 49.89% / 77.65%; TDA with 64 views 60.11% / 80.24%; TPT 54.77% / 77.06%), and GPU cost per dataset.

## Repository layout

```
run.sh                  pipeline runner (one dataset, chosen stages)
pyproject.toml          installs the tta package (pip install -e .)
src_main/
  tta/
    registry.py         one entry per dataset: images, class names, templates, mask, TDA config, view settings, published targets
    data.py             dataset -> ordered samples (id, path, label); ImageFolder, manifest and CoOp-split layouts
    common.py           device, seeding, CLIP loading, text weights, uncertainty stats, results I/O
  scripts/
    run_vanilla_clip.py zero-shot CLIP (reproduction check + vanilla gate features + embeddings)
    run_tda.py          TDA
    run_tpt.py          TPT
    build_win_labels.py merge the three runs per image; outcome labels, gate features, headroom
    cost_aware_routing.py  gates (logreg, GB, MLP), budget curves, feature analysis, threshold baselines, RL gates (--rl)
configs/tda/            official TDA configs (copied from refs/tda/configs)
data/                   classnames/, templates/, masks/, manifests/ are tracked; images are not
results/<dataset>/      per-dataset outputs (see below)
archive/                first-pilot scripts and gate searches (frozen; some no longer run)
refs/                   official TPT and TDA repos (gitignored)
```

## Setup

```bash
conda create -n vlm-tta python=3.11 -y
conda activate vlm-tta
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())"
python -c "import tta.registry as r; print(sorted(r.REGISTRY))"

# official implementations, used for verification and AugMix operations (gitignored)
git clone https://github.com/azshue/TPT refs/tpt
git clone https://github.com/kdiaaa/tda refs/tda
```

`DATA_ROOT` (default `data`) and `RESULTS_ROOT` (default `results`) can be set as environment variables, e.g. to point at a cluster's data directory.

## Running

Run from the repo root. `run.sh` runs any subset of the five stages for one registered dataset and saves each stage's output to `results/<dataset>/logs/<stage>.log`:

```bash
chmod +x run.sh                                  # once
./run.sh imagewoof_pilot                         # all stages: clip tda tpt labels route
./run.sh imagenet_a clip                         # zero-shot check only
N=100 ./run.sh imagenet_a clip tda               # smoke test on the first 100 images
./run.sh imagewoof_pilot labels route            # re-analyze saved results (no model)
RESULTS_ROOT=/tmp/test N=20 ./run.sh imagewoof_pilot clip tda tpt labels   # test without touching results/
```

For variants, call the scripts directly; `--tag` keeps their outputs separate:

```bash
python src_main/scripts/run_tda.py --dataset imagewoof_pilot --shuffle-seed 1 --tag seed1
python src_main/scripts/run_tda.py --dataset imagenet_a --n-views 1 --tag 1view
python src_main/scripts/build_win_labels.py --dataset imagenet_a --tda-tag 1view --out-tag 1view
python src_main/scripts/cost_aware_routing.py --dataset imagenet_a --tag 1view --skip-permutation
python src_main/scripts/cost_aware_routing.py --dataset imagewoof_pilot --rl        # add RL gates (--rl-only: just those)
python src_main/scripts/run_tpt.py --dataset imagewoof_pilot --ctx-init random --suffix ", a type of dog." \
    --view0 random --final selected_avg --tag legacy        # first-pilot TPT
```

| Stage | Script | Output in `results/<dataset>/` |
|---|---|---|
| `clip` | `run_vanilla_clip.py` | `clip.json`, `clip_embeddings.npz` |
| `tda` | `run_tda.py` | `tda.json` |
| `tpt` | `run_tpt.py` | `tpt.json` |
| `labels` | `build_win_labels.py` | `win_labels.json` (prints headroom: TPT-only %, TDA-only %, oracle) |
| `route` | `cost_aware_routing.py` | `cost_curve.png`, `threshold_baseline.png`, `rl_gates.png` with `--rl` (prints gate tables) |

Only the model stages need a GPU or long runtimes. Routing can be simulated exactly from saved outputs, because TDA runs on every image regardless of routing (its cache evolves the same way) and TPT resets after every image; the analysis stages therefore take seconds to minutes and never rerun a model.

### Results format

Each run writes a list of per-image rows plus a `.meta.json` (git commit, arguments, accuracy). Every row has `id` (image path relative to `DATA_ROOT`, the join key across methods), `label`, `pred`, `correct`, `label_name`, `pred_name`, and method signals:

- **CLIP:** `clip_conf`, `clip_margin`, `clip_entropy`, `clip_norm_entropy` (entropy / ln C, comparable across datasets), `text_alignment`.
- **TDA:** the same stats for CLIP as TDA sees it (`clip_*`, `clip_view0_*`) and for TDA's output (`tda_*`); whether TDA changed CLIP's prediction; cache state before the image (similarities, fill, agreement with CLIP); how much each cache moved the logits; stream position; timings.
- **TPT:** zero-shot stats before tuning (`zs_*`); TPT's output stats (`tpt_*`); whether tuning changed the prediction; the loss before the update; view and 8-view probe entropy/agreement; timings.

### Adding a dataset

1. Put the images under `data/` and the class names (JSON list, index = label), templates and, if needed, a class mask under `data/classnames/`, `data/templates/`, `data/masks/`.
2. Add a `DatasetSpec` to `src_main/tta/registry.py` (layout, paths, TDA config, `tda_n_views`, `tpt_augmix`, published targets).
3. `./run.sh <dataset> clip` and compare to the published CLIP number before running anything else.

Label conventions are handled in `data.py`: WNID folders are labeled in sorted order (matching ascending class masks); ImageNet-V2's numeric folders are labeled by their number.

### Pilot data

The pilot's data preparation (`archive/src/select_images.py`, `apply_corruptions.py`, `align_classnames.py`) has already been run; its outputs live in `data/manifests/`, `data/classnames/` and `data/templates/`. The run before the restructure is preserved in `results/pilot_v2/` as the regression reference; `results/imagewoof_pilot/` holds the same results in the current format.

## Notes

- Results from the first pilot are preserved in `results/pilot_v1/`; earlier gate-search scripts are in `archive/src/archive/`.
- `imagecorruptions` was replaced with a direct scikit-image implementation due to packaging issues.