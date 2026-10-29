# Cost-Aware Routing Between Test-Time Adaptation Methods for VLMs

Test-time adaptation (TTA) recovers CLIP accuracy under distribution shift, but methods differ widely in cost. Per-image methods such as ZERO and TPT (test-time prompt tuning) encode 64 augmented views of every image, and TPT adds a gradient step; TDA (training-free dynamic adapter) can run on a single view with a small cache. This project routes each image: 1-view TDA on every image, and an expensive method only on the images a nearly free gate flags as uncertain.

## Status

- **ImageNet-A done end to end on the Mac (7,500 images).** Every method reproduces its published number: CLIP 49.9% (49.89%), TDA with 64 views 60.1% (60.11%), ZERO 59.2% (59.77% in TTA-VLM), TPT 54.6% (54.77%).
- **Main result.** A label-free gate that sends low-CLIP-margin images to ZERO is the best option between 31 and ~430 ms per image. It comes within 1 point of always-ZERO at 57% of ZERO's cost and ties it at 82%.
- **Against CAS** (closest prior work, adapt-or-skip): its gate has to encode ZERO's views for every image, so it saves almost nothing on ZERO. At the same accuracy (~58.6%) our gate costs 287 ms vs. 464 ms per image.
- **What didn't help:** learned, MLP and RL gates over a plain margin threshold; averaging TDA's and ZERO's probabilities; a TPT step after ZERO.
- **Open:** 64-view TDA (60.1% at 651 ms) is above every routing option on ImageNet-A. Proposed next: apply TDA's cache to ZERO's views on routed images.
- **Next:** NCSA DeltaAI runs (lab ACCESS allocation) for ImageNet-R, V2, Sketch, ImageNet and 3–4 cross-domain sets. Target: CVPR 2027.

## Methods

| Role | Method | Script |
|---|---|---|
| Cheap, every image | TDA, 1 view | `run_tda.py --n-views 1` |
| Expensive (main) | ZERO: 64 views, keep the 30% most confident, majority vote, no backprop | `run_zero.py` |
| Expensive (second) | TPT: one step of prompt tuning on 64 views | `run_tpt.py` |
| Gate | Low CLIP top-2 margin; no labels or training (threshold from a quantile of the stream) | `cost_aware_routing.py`, `fusion_cascade.py` |
| Baselines | Always-TDA / ZERO / TPT, 64-view TDA, CAS-style adapt-or-skip, random routing | `fusion_cascade.py` |

**Cost accounting is conservative:** TDA runs on every image (its cache must see the whole stream, and the gate's signals come from its pass), and routed images pay for the expensive method on top. Reported costs are mean time per image on an Apple M2 Max (MPS) unless noted.

**Routing is simulated exactly from saved outputs.** TDA's cache evolves the same way whatever the gate does, and ZERO and TPT reset after every image, so one full run of each method gives the per-image result at every budget. All analysis stages take seconds to minutes and never rerun a model.

## ImageNet-A results

7,500 images, CLIP ViT-B/16, fixed shuffled stream order (`--shuffle-seed 1`).

| Method | Accuracy | Published | Median time per image |
|---|---|---|---|
| CLIP, 80 prompts | 49.9% | 49.89% | — |
| CLIP, TDA's 7 prompts (gate input) | 50.7% | — | — |
| CLIP, "a photo of a {}." | 47.8% | — | — |
| TDA, 1 view | 51.1% | — | 30 ms |
| TDA, 64 views (official setting) | 60.1% | 60.11% | 641 ms |
| ZERO | 59.2% | 59.77% (TTA-VLM) | 451 ms |
| TPT | 54.6% | 54.77% | 1,323 ms |

ZERO with the official code's fp16 soft votes gives 59.3%; the main number uses exact one-hot votes, as the paper defines.

### Headroom

| | TDA → ZERO | TDA → TPT |
|---|---|---|
| Only the expensive method right | 14.2% (1,067) | 7.7% (581) |
| Only TDA right | 6.0% (453) | ~4.2% |
| Expensive method's lead over TDA | 8.1 pts | 3.5 pts |
| Oracle (perfect gate) | 65.3% at 97 ms | 58.8% at 134 ms |

### Accuracy against cost

| Option | Accuracy | Mean cost per image |
|---|---|---|
| 1-view TDA | 51.1% | 31 ms |
| Margin gate → ZERO, 30% routed | 56.6% | 171 ms |
| Margin gate → ZERO, 50% routed | 58.3% | 264 ms |
| Margin gate → ZERO, 75% routed | 59.2% | 380 ms |
| Always ZERO | 59.2% | 465 ms |
| CAS-style gate with ZERO, any budget | 58.3–59.3% | 462–496 ms |
| Margin gate → TPT, 30% routed | 53.6% | 430 ms |
| Margin gate → TPT, 60% routed | 54.7% | 829 ms |
| Always TPT | 54.6% | 1,330 ms |
| Average of TDA and ZERO, every image | 57.7% | 496 ms |
| Average of ZERO and TPT, every image | 59.1% | 1,795 ms |
| Cascade: 70% to ZERO, a quarter of those to TPT | 57.3% | 509 ms |
| 64-view TDA | 60.1% | 651 ms |

### Gates

5-fold cross-validation (6,000 train / 1,500 test per fold), averaged over 5 fold splits (3 for RL).

| Gate | AUC, needs ZERO | AUC, needs TPT |
|---|---|---|
| Low CLIP margin (threshold) | 0.748 | 0.764 |
| Low TDA margin (threshold) | 0.743 | 0.753 |
| Logistic regression, cheap scalars (CLIP uncertainty + TDA cache) | 0.751 | 0.782 |
| Logistic regression, cheap scalars + TDA's own output | 0.757 | 0.790 |
| Logistic regression, + PCA-5 CLIP embeddings | 0.760 | 0.792 |
| Gradient boosting / MLP, cheap + TDA output | 0.753 / 0.738 | 0.783 / 0.766 |
| CLIP embeddings alone (PCA-5) | 0.54–0.56 | 0.55–0.56 |
| Cost-shaped REINFORCE, λ = 0.02 | 0.714 (58.9% at 304 ms) | — |
| Accuracy-only REINFORCE / winner MLP | 0.61 / 0.58 | — |

### Findings

- **The gate works and stays simple.** A plain low-margin threshold is within ~0.01–0.03 AUC of every learned gate. TDA's own confidence adds ~0.01, CLIP embeddings are near chance, and RL lands on the same accuracy-cost curve. Same conclusion as the pilot.
- **Why TPT is easier to match than ZERO.** To tie the expensive method a gate must recover its whole net lead over TDA. TPT leads by 3.5 points, so catching about half of its wins is enough (tie at ~55–60% routed). ZERO leads by 8.1 points, so the gate needs most of its 1,067 wins, which are spread through the ranking (tie at 75%).
- **Routing never beats always-ZERO** (peak 59.3% vs. a 65.3% oracle): on the 6% of images where only TDA is right, TDA looks as uncertain as on the rest.
- **Averaging hurts.** TDA's probabilities are much more peaked than ZERO's average over its kept views, so the weaker method dominates (57.7% vs. 59.2%). The best weighting (0.75 on ZERO) peaks at 59.5%, but that weight was picked on the same images. Routed images keep ZERO's answer.
- **TPT adds nothing after ZERO.** Every share sent on to TPT costs more and lowers accuracy; the two use the same kind of views and make the same mistakes.
- **CAS saves nothing on ZERO,** consistent with the 1.09x saving CAS reports for ZERO. CAS's code is unreleased; the CAS-style baseline follows its paper using the view-consistency signals logged by `run_zero.py` (on ZERO's crop + flip views, where CAS uses AugMix).
- **ZERO is the right main method:** 4.6 points more accurate than TPT at about a third of the cost.

### Next on ImageNet-A

- **TDA's cache on ZERO's views.** 64-view TDA's gain comes from many views plus the stream cache; ZERO has only the views. On routed images ZERO has already encoded 64 views, so applying TDA's cache to them costs almost nothing extra. Updating the cache from 1-view features on every image keeps the simulation exact. One new script and about an hour on the Mac.
- **Fusion, done honestly.** Choose the weight on another dataset (ImageNet-R) and apply it to ImageNet-A, or calibrate each method's probabilities without labels before averaging. Keep fusion only if it gains ≥0.5 points on datasets it wasn't tuned on.

## Pilot (Imagewoof, 1,000 images)

- **Data:** [Imagewoof](https://github.com/fastai/imagenette) validation images (10 dog breeds), half Gaussian blur and half Gaussian noise (scikit-image implementation of ImageNet-C severity conventions).
- **Model:** CLIP ViT-B/16 (open_clip `ViT-B-16-quickgelu`, OpenAI weights), Apple M2 Max (MPS).
- **Prompt:** `a photo of a {}.` for all methods; official ImageNet class names ("Shih Tzu", "Beagle").
- **TDA:** 1 view, α = 1.0, β = 5.0 (no official config for Imagewoof; untuned). **TPT:** official defaults, crop + flip views.

| Method | Accuracy | Notes |
|---|---|---|
| Vanilla CLIP | 59.0% | blur 63.6%, noise 54.4% |
| TDA (1 view) | 60.1% | 59.9 / 60.6 / 60.5% across 3 shuffle seeds; ~25–35 ms/image |
| TPT (official) | 61.0% | ~600–770 ms/image (~22–24x TDA) |
| Oracle (best of TPT/TDA per image) | 64.8% | 3.8 pts above TPT |

Per-image outcomes: both right 563, both wrong 352, only TPT right 47 (4.7%), only TDA right 38.

| Gate (needs TPT) | AUC | Acc @20% to TPT | Matches always-TPT at |
|---|---|---|---|
| Low top-2 margin (threshold) | 0.861 | 60.5% | 25% |
| Logistic regression, cheap scalars | 0.844 ± 0.008 | 60.8% | 25% |
| Gradient boosting, cheap scalars | 0.835 ± 0.009 | 61.2% | 15% |
| MLP, cheap scalars | 0.815 ± 0.007 | 61.0% | 15% |
| Cost-shaped REINFORCE, λ = 0.02 | 0.78 | 61.0% at 18% routed | 18% |
| Accuracy-only REINFORCE | 0.43 | routes 70% to TPT | — |
| Winner MLP (disagreements only) | 0.47–0.59 | 60.2–60.7% | 35% or never |
| Random routing | 0.5 | 60.3% | — |

- **The ranking signal is real:** permutation test on the logistic-regression AUC (200 label shuffles), observed 0.847 vs. shuffled mean 0.501; p < 0.005.
- **Margin carries it.** CLIP-uncertainty features alone reach AUC 0.853; TDA's cache features alone reach 0.702 and add nothing on top.
- **RL can at best match a scorer.** With both outcomes known and one decision per image, the cost-shaped objective (accuracy − λ × cost) is maximized by routing whenever the predicted advantage of the expensive method exceeds λ. A supervised scorer estimates that directly; REINFORCE gets there through noisier sampled gradients, and each budget needs its own λ and training run. Accuracy-only RL learns "mostly TPT"; the winner MLP learns who wins given a disagreement, not whether one will happen.
- **Limitation:** each gate learned from ~38 positives per fold, and TPT led TDA by only 0.9 points, so ImageNet-A is the real test.

**Why these numbers differ from the first pilot** (`results/pilot_v1/`: TDA 62.5%, TPT 64.0%): the first pilot gave TDA and TPT the prompt "…, a type of dog." while vanilla CLIP used "a photo of a {}", so part of their gain came from the prompt; its TPT also used a random prompt initialization, a random first view and a six-view average. Legacy flags reproduce the first pilot exactly.

**Future consideration: attention-based gates** (over CLIP's top-k class embeddings, TDA's cache entries or recent images). Worth building only if a nonlinear gate clearly beats the margin threshold on the benchmarks; so far none has.

## Implementations and verification

Each was compared line by line against the official repository ([TPT](https://github.com/azshue/TPT), [TDA](https://github.com/kdiaaa/tda), [ZERO](https://github.com/FarinaMatteo/zero)) and reviewed independently; ImageNet-A reproduces every published number.

**TDA** (`run_tda.py`)
- Matches official: the CLIP pseudo-label updates both caches before the image's own prediction; entropy-ordered queues (3 positive / 2 negative per class); negative-cache gating on entropy in nats divided by log2(classes) (official convention); logit scale 100; text weights averaged over the dataset's templates (TDA's 7 on ImageNet/-A/-V2, the 80 CLIP templates on -R/Sketch).
- Views follow the dataset's official setting by default: 1 view (ImageNet and the cross-domain sets) or 64 views (center crop + 63 AugMix views, lowest-entropy 10% averaged) on ImageNet-A, -V2, -R and Sketch. The TDA paper does not mention the 64 views; they come from `build_test_data_loader` in its `utils.py`. `--n-views` overrides.
- Use `--shuffle-seed`: official TDA shuffles the stream, and ImageNet-A otherwise loads in class order.
- Deviations: fp32 by default (official fp16; `--half` on CUDA); seeded shuffle rather than official's exact sequence; pilot α/β untuned.

**TPT** (`run_tpt.py`)
- Matches official: 4 context vectors from "a photo of a"; view 0 = resize + center crop; 63 random-resized-crop + flip views (AugMix on the cross-domain sets); lowest-entropy 6 of 64 views; entropy loss on their average; AdamW lr 5e-3, 1 step; reset per image; prediction from view 0.
- Encodes views once instead of every step (same predictions, slightly cheaper). Deviations: fp32 by default (`--amp` on CUDA); seed 42 vs. official 0.

**ZERO** (`run_zero.py`)
- Matches official: center view + 63 RandomResizedCrop(224) + flip views, no AugMix on any dataset; prompt "a photo of a {}."; keep the 30% most confident views (19 of 64); predict the most-voted class; official greedy tie-break; seed 1.
- One deliberate difference, logged both ways: official ZERO runs under fp16 autocast, so its "zero temperature" becomes soft votes. The main prediction uses exact one-hot votes (as the paper defines); `zero_fp16_*` emulates the official fp16 voting on the same views (verified identical to a port of the official code on 3,000 near-tied cases). `--fp16-votes` makes it the main prediction. ImageNet-A: 59.24% one-hot, 59.35% fp16.
- Also logs CAS's gate ingredients (agreement between the lowest-entropy 10% of views and the original view, by label and by probability cosine) and, with `--save-view-probs`, the selected views' probabilities so CAS's exact score can be recomputed when its code is released.

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
    prepare_data.py     download ImageNet-A/-R; build class names, masks and templates from the official repos; check folders
    run_vanilla_clip.py zero-shot CLIP (reproduction check, gate features, embeddings)
    run_tda.py          TDA (1 or 64 views)
    run_zero.py         ZERO (+ CAS gate signals)
    run_tpt.py          TPT
    summarize.py        one table of every finished run for a dataset -> summary.md / summary.csv
    build_win_labels.py merge CLIP, TDA and one expensive method (--expensive zero|tpt); outcome labels, gate features, headroom
    cost_aware_routing.py  gates (thresholds, logreg, GB, MLP, RL with --rl), budget curves 0-100%, feature analysis
    fusion_cascade.py   fusion rules, TDA -> ZERO -> TPT cascade, CAS-style baselines, accuracy-cost Pareto front
configs/tda/            official TDA configs
data/                   classnames/, templates/, masks/, manifests/ are tracked; images are not
results/<dataset>/      per-dataset outputs (see below)
archive/                first-pilot scripts and gate searches (frozen)
refs/                   official TPT, TDA and ZERO repos (gitignored)
```

## Setup

```bash
conda create -n vlm-tta python=3.11 -y
conda activate vlm-tta
pip install -r requirements.txt
pip install -e .
python -c "import tta.registry as r; print(sorted(r.REGISTRY))"

# official implementations, used for verification, data preparation and AugMix (gitignored)
git clone https://github.com/azshue/TPT refs/tpt
git clone https://github.com/kdiaaa/tda refs/tda
git clone https://github.com/FarinaMatteo/zero refs/zero

# ImageNet-A (and -R): images, class names, masks, templates
python src_main/scripts/prepare_data.py imagenet_a --download
```

`DATA_ROOT` (default `data`) and `RESULTS_ROOT` (default `results`) can be set as environment variables, e.g. to point at a cluster's data directory.

## Running

### ImageNet-A, as run for the results above

```bash
export PYTHONUNBUFFERED=1
# zero-shot check (prints the published target)
./run.sh imagenet_a clip

# model runs (~5.5 h on an M2 Max); parentheses so every stage goes to the log
( python src_main/scripts/run_tda.py --dataset imagenet_a --n-views 1 --tag 1view --shuffle-seed 1 --save-probs && \
  python src_main/scripts/run_tda.py --dataset imagenet_a --shuffle-seed 1 --save-probs && \
  python src_main/scripts/run_zero.py --dataset imagenet_a --save-probs --save-view-probs && \
  python src_main/scripts/run_tpt.py --dataset imagenet_a --save-probs ) 2>&1 | tee results/imagenet_a/overnight.log

# analysis (seconds to minutes, no model)
python src_main/scripts/summarize.py --dataset imagenet_a
python src_main/scripts/build_win_labels.py --dataset imagenet_a --expensive zero --tda-tag 1view --out-tag 1view
python src_main/scripts/build_win_labels.py --dataset imagenet_a --expensive tpt  --tda-tag 1view --out-tag 1view
python src_main/scripts/cost_aware_routing.py --dataset imagenet_a --tag zero_1view --skip-permutation --rl
python src_main/scripts/cost_aware_routing.py --dataset imagenet_a --tag 1view --skip-permutation
python src_main/scripts/fusion_cascade.py --dataset imagenet_a --tda-tag 1view
```

On a Mac, keep it awake during long runs with `caffeinate -dimsu` in a second terminal (Ctrl+C to stop).

### run.sh

`run.sh` runs any subset of the stages for one registered dataset with default settings and saves each stage's output to `results/<dataset>/logs/<stage>.log`:

```bash
chmod +x run.sh                                  # once
./run.sh imagewoof_pilot                         # clip tda tpt labels route
./run.sh imagenet_a summary                      # table of every finished run
N=100 ./run.sh imagenet_a clip tda               # smoke test on the first 100 images
RESULTS_ROOT=/tmp/test N=20 ./run.sh imagewoof_pilot clip tda tpt labels   # test without touching results/
```

For variants (1-view TDA, ZERO, `--expensive zero`, tags), call the scripts directly as above; `--tag` keeps outputs separate. `cost_aware_routing.py --gate-features cheap` uses the pilot's feature set (without TDA's own output) for the feature analysis and RL gates.

### Outputs in `results/<dataset>/`

| Script | Output |
|---|---|
| `run_vanilla_clip.py` | `clip.json`, `clip_embeddings.npz` |
| `run_tda.py` | `tda[_<tag>].json`, `.probs.npz` with `--save-probs` |
| `run_zero.py` | `zero.json`, `zero.probs.npz` (votes, kept-view average, view-0 probabilities), `zero.viewprobs.npz` (CAS views) |
| `run_tpt.py` | `tpt.json`, `tpt.probs.npz` |
| `summarize.py` | `summary.md`, `summary.csv` |
| `build_win_labels.py` | `win_labels[_<tag>].json` (prefixed `zero_` with `--expensive zero`); prints headroom |
| `cost_aware_routing.py` | `cost_curve_<tag>.png`, `threshold_baseline_<tag>.png`, `rl_gates_<tag>.png` |
| `fusion_cascade.py` | `fusion_cascade.json`, `fusion_cascade.png` |

Each run also writes a `.meta.json` (git commit, arguments, accuracy, summary). Every row has `id` (image path relative to `DATA_ROOT`, the join key across methods), `label`, `pred`, `correct`, and method signals:

- **CLIP:** `clip_conf`, `clip_margin`, `clip_entropy`, `clip_norm_entropy` (entropy / ln C, comparable across datasets), `text_alignment`.
- **TDA:** CLIP as TDA sees it (`clip_*`, `clip_view0_*`), TDA's output (`tda_*`), whether TDA changed CLIP's prediction, cache state before the image, how much each cache moved the logits, stream position, timings.
- **ZERO:** vote agreement and margin, tie handling, one-hot and fp16 predictions, view statistics, CAS signals (`cas_*`), timings (views, encoding, vote).
- **TPT:** zero-shot stats before tuning (`zs_*`), TPT's output (`tpt_*`), whether tuning changed the prediction, view and probe statistics, timings.

`win_labels` rows carry generic fields for the router (`expensive`, `exp_correct`, `exp_seconds`, `exp_pred`, `needs_exp` = expensive method right and TDA wrong) plus the method's own fields; TPT files also keep `needs_tpt`, and older files without `exp_*` are read as TPT.

### Adding a dataset

1. Put the images under `data/` and the class names (JSON list, index = label), templates and, if needed, a class mask under `data/classnames/`, `data/templates/`, `data/masks/` (`prepare_data.py` does this for ImageNet-A/-R).
2. Add a `DatasetSpec` to `src_main/tta/registry.py` (layout, paths, TDA config, `tda_n_views`, `tpt_augmix`, published targets).
3. `./run.sh <dataset> clip` and compare to the published CLIP number before running anything else.

Label conventions are handled in `data.py`: WNID folders are labeled in sorted order (matching ascending class masks); ImageNet-V2's numeric folders are labeled by their number.

## Notes

- The pilot's data preparation (`archive/src/select_images.py`, `apply_corruptions.py`, `align_classnames.py`) has already been run; its outputs live in `data/`. `results/pilot_v2/` holds the pre-restructure run used as the regression reference; `results/pilot_v1/` the first pilot.
- `imagecorruptions` was replaced with a direct scikit-image implementation due to packaging issues.
- Mac timings vary with load and temperature (~10–20%); paper costs will be measured on GPU, on an otherwise idle machine, after warm-up.