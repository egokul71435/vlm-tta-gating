# Meta-Learned Test-Time Adaptation: Strategy Selection for VLMs

Pilot project exploring whether a lightweight, meta-learned gate can predict, per image, which test-time adaptation (TTA) strategy (TPT vs. TDA) will work better for a CLIP-based vision-language model under domain shift, without manual per-domain tuning.

## Status: routing the most uncertain images to TPT matches TPT's accuracy with 15–25% of images sent to TPT (~3.3–5x cheaper); a simple threshold does as well as a learned gate

| n | Vanilla | TPT | TDA | Oracle | Oracle gap |
|---|---|---|---|---|---|
| 150 | 57.3% | 56.7% | 61.3% | 66.7% | 5.3pt |
| 300 | 56.3% | 58.0% | 61.0% | 64.7% | 3.7pt |
| 500 | 57.0% | 56.4% | 61.6% | 68.2% | 6.6pt |
| 1000 | 58.5% | 64.0% | 62.5% | 69.8% | 5.8pt |

The n=1000 row comes from a seeded run (`torch.manual_seed(42)` in `run_tpt.py`/`run_tda.py`) and is reproducible. The n=150/300/500 rows predate the seeding fix, so TPT/TDA numbers there may shift slightly on a rerun. The qualitative pattern (TDA > vanilla, stable oracle gap) is what to take from them.

## Gate results

All numbers are from the seeded n=1,000 run. Of the 1,000 images:

| Case | Images | Does the gate's choice matter? |
|---|---|---|
| Both TPT and TDA correct | 567 | No |
| Both wrong | 302 | No |
| Only TPT correct | 73 | Yes |
| Only TDA correct | 58 | Yes |

Only the **131 disagreement cases** carry information about which strategy to pick, so gates are trained and compared there. Overall accuracy follows directly: (567 + disagreement cases the gate gets right) / 1,000.

### Summary

| Gate | Disagreement cases (n=131) | Overall (n=1,000) | Share of oracle headroom captured* |
|---|---|---|---|
| Vanilla CLIP (no adaptation) | — | 58.5% | — |
| Always TDA | 44.3% | 62.5% | — |
| **Always TPT (baseline)** | **55.7%** | **64.0%** | 0% |
| Entropy-only classifier (best: logistic regression) | 55.0% | 63.9% | ~0% |
| RL gate (REINFORCE) | ~50–57%† | 63.3–64.1% | ~0% |
| Embedding MLP (PCA-5) | 59.8% | ~64.5% | ~9% |
| Gradient boosting (entropy + view-spread) | 60.7% | ~64.7% | ~11% |
| **Oracle (perfect per-image choice)** | **100%** | **69.8%** | 100% |

\*Headroom = oracle minus always-TPT: 5.8 points overall, or 58 images. A gate's share = (its overall accuracy − 64.0%) / 5.8 points.
†The RL gate was evaluated on all 1,000 images; its disagreement-case accuracy is derived from its overall accuracy using the formula above.

**Takeaway:** the oracle shows up to 5.8 points of overall gain is available. The best gates so far capture roughly a tenth of it: about +4–5 points on the disagreement cases, or about +0.5–0.7 points overall.

### Gate 1: Classic classifiers (`src/fit_gate_final.py`)

- **Setup:** 4 feature sets × 4 models (logistic regression, random forest, gradient boosting, k-NN) = 16 configurations, trained on the 131 disagreement cases and evaluated with leave-one-out cross-validation.
- **Features tried:** pre-adaptation entropy; + confidence; + corruption type; entropy + TPT view-entropy spread (the mean and std of CLIP's entropy across TPT's 64 augmented views, before adaptation).
- **Outcome:** only gradient boosting on entropy + view-spread beat the baseline: **60.7% vs. 55.7%**, averaged over 7 seeds (std 0.004). All entropy-only variants were at or below baseline.
- **Cost note:** the view-spread features need TPT's 64 forward passes, so this gate pays a large part of TPT's cost before deciding.
- **Earlier false positive:** a random forest result (65.5%) didn't replicate once TPT/TDA augmentation was seeded.

### Gate 2: RL gate (`src/train_rl_gate.py`, `main`)

- **Setup:** a contextual bandit. A small network (4 scalar features → 16 hidden → 2 actions) picks TPT or TDA per image and receives reward 1 if that strategy was correct. Trained with REINFORCE on all 1,000 images, using the average reward of both options as a variance-reducing baseline. Evaluated with 5-fold cross-validation.
- **Outcome:** **63.3–64.1% overall vs. 64.0% always-TPT** across runs, and 1/5 seeds above baseline in the stability check. No gain.
- **Likely reasons:** 87% of images give identical reward for either action (no learning signal), and sampling actions adds noise that supervised training avoids.

### Gate 3: Embedding MLP (`src/train_rl_gate.py`, `run_embedding_experiment`)

- **Setup:** supervised classifier on vanilla CLIP image embeddings (512-dim), PCA-reduced to 5 dims inside each fold (≈30% of variance retained), MLP with 64 hidden units, weight decay 0.1, 30 epochs. Trained on the 131 disagreement cases, evaluated with 5-fold cross-validation.
- **Outcome:** **59.8% vs. 55.7%**, averaged over 7 seeds (std 0.013), with all 7 above baseline.
- **What didn't work:** raw 512 dims memorized the training data (100% train vs. ~64% test); training on all 1,000 images collapsed to always predicting TPT; 30 PCA dims overfit to 47%.
- **Cost note:** the embedding comes from vanilla CLIP's single forward pass, so this gate is nearly free to run.

### Cost (measured)

Median wall-clock time per image on an M2 Max (MPS), excluding image loading:

| Strategy | Accuracy | Time per image |
|---|---|---|
| Always TDA | 62.5% | 29.3 ms |
| Always TPT | 64.0% | 524.7 ms (~18x TDA) |
| Cost-aware oracle (TPT only where it alone is right: 7.3% of images) | 69.8% | 65.4 ms (8x cheaper than always-TPT) |

86% of TPT's cost (449 ms) is generating its 64 augmented views and their forward pass. Gates that use view-spread features (gradient boosting, RL) pay that cost on every image just to decide, so they can't deliver meaningful savings. Gates built on cheap signals (vanilla CLIP outputs, TDA cache statistics, CLIP embeddings) add almost no overhead. Wall-clock ratios are hardware-specific.

### Gate 4: Cost-aware routing (`src/cost_aware_routing.py`)

Instead of asking "which strategy is better?", this gate asks **"does this image need TPT?"** (`needs_tpt` = TPT right and TDA wrong, 73 of 1,000 images). A scorer ranks images by that probability using only cheap features. For a given TPT budget, the top-scoring images go to TPT and the rest to TDA.

- **Features (cheap, from vanilla CLIP and TDA's cache):** entropy, confidence, top-2 margin, text alignment, positive-cache max/mean similarity, cache-agrees-with-CLIP, cache fill, negative-cache fill. No view-spread features.
- **Models:** logistic regression and gradient boosting on three feature sets (scalars, PCA-5 embeddings, both). All six are reported; settings were fixed in advance.
- **Evaluation:** 5-fold cross-validation, averaged over 5 different fold splits, using all 1,000 images.
- **Cost accounting (conservative):** TDA runs on every image, since its cache must keep updating and the cheap features come from its pass. Routed images pay for TPT on top. Mean per-image times are used, so the numbers differ slightly from the median-based table above.

**Best configuration: logistic regression on cheap scalars** (AUC 0.757 ± 0.007, average precision 0.161 vs. 0.073 at random):

| Strategy | Images sent to TPT | Accuracy | Cost per image |
|---|---|---|---|
| Always TDA | 0% | 62.5% | 30 ms |
| Gate | 20% | 63.8% | ~138 ms |
| **Gate** | **25%** | **≥64.0% (matches always-TPT)** | **~165 ms (~3.3x cheaper)** |
| Gate | 30% | 64.7% | ~192 ms (~2.8x cheaper) |
| Always TPT | 100% | 64.0% | 537 ms |
| Cost-aware oracle | 7.3% | 69.8% | 70 ms |

All six configurations beat random routing at every budget. Plot: `results/cost_curve.png`.

**What drives it** (feature analysis, logistic regression):
- **Vanilla CLIP uncertainty carries most of the signal:** alone it reaches AUC 0.750, vs. 0.757 with all features.
- **TDA cache agreement is the one cache feature that adds unique signal:** removing it drops AUC by 0.017. When the cache disagrees with CLIP, the image is more likely to need TPT.
- The other cache features are largely redundant with uncertainty.
- In short, **TPT is needed when CLIP is torn between classes and TDA's cache disagrees with CLIP.**

**Caveats:** "matches at 25%" sits right at the line (64.3% vs. 64.0% is about 3 images), so "25–30%" is the safer claim. There's a large gap to the oracle: most top-ranked images don't actually need TPT. Confirmation on new images or another dataset is still pending.

#### Simple-threshold baselines (`threshold_baselines` in `src/cost_aware_routing.py`)

No training: each rule ranks images by one feature and sends the top X% to TPT. Directions were fixed in advance (more uncertain → TPT).

| Rule | AUC | AP | Acc @20% | Acc @30% | Matches always-TPT at |
|---|---|---|---|---|---|
| High entropy | 0.754 | 0.163 | 63.8% | 63.8% | 25% |
| Low confidence | 0.757 | 0.158 | 64.0% | 64.2% | 20% |
| Low margin | 0.755 | 0.166 | 64.2% | 64.5% | 20% |
| Cache disagrees, then entropy | 0.746 | 0.165 | 64.1% | 64.5% | **15%** |
| Learned gate (logreg, cheap scalars) | 0.757 | 0.161 | 63.8% | 64.7% | 25% |

- **At this scale, a single uncertainty threshold does as well as the learned gate.** Differences are 0.2–0.5 points (2–5 images), within noise.
- **The two-signal rule** (send images where TDA's cache disagrees with CLIP, then the most uncertain) matches TPT with only 15% of images going to TPT, the cheapest point so far, and is fully interpretable. It was designed after the feature analysis, so it needs confirming on new data.
- **What this means for the contribution:** the value is in the cost-aware analysis and the finding that uncertainty plus cache agreement predict when TPT is needed, not in a complex gate. Learned gates may help again with more data and richer features.

Plot: `results/threshold_baseline.png`.

- **Positioning:** the learned scorer is simpler than RL (one model covers every budget via a cutoff), but simple thresholds match both at this scale (see below). The main method is therefore the routing approach itself, with a simple rule as the default; learned gates are revisited at larger scale.

## Statistical checks

| Check | Applied to | What it shows | Result |
|---|---|---|---|
| Cross-validation | All gates | Scores on images not used for training | Numbers above |
| Seed stability | Gradient boosting, RL, MLP | Result doesn't depend on one random initialization | Stable for all three |
| Permutation test (200 label shuffles) | Gradient boosting, MLP | Whether the gap over baseline beats chance | p = 0.045 (GB), p = 0.030 (MLP, single seed) |
| Permutation test on AUC (200 shuffles of `needs_tpt`) | Cost-aware gate (logreg, cheap scalars) | Whether the scorer ranks images better than chance | Observed AUC 0.770 vs. shuffled max 0.634; **p < 0.005** (survives correction for the 6 configurations) |

**Caveats:**
- The caveats below apply to the disagreement-case gates (1–3). With 131 cases, accuracy has a standard error of about ±4 points, so a ~5-point gap is about one standard error.
- The permutation p-values treat each gate as the only one tried. Gradient boosting was the best of 16 configurations, and the MLP's PCA size and regularization were tuned on the same folds. Correcting for 16 tries would require p < 0.003, so both results are **nominally significant**.
- Seed checks vary model initialization, not the data split, so they show stability, not significance.

**Next steps:** a cost-shaped RL gate (reward = correct − λ × cost) as a comparison against the cost-aware scorer; then scaling and cross-dataset tests (e.g., train on ImageNet-R, test on ImageNet-A or Sketch) once GPU access is available.

Note: n=150/300/500/1000 are nested samples (same seed for image selection), not independent replications.

## Setup

```bash
conda create -n vlm-tta python=3.11 -y
conda activate vlm-tta
pip install -r requirements.txt
```

Confirm MPS is available (Apple Silicon):
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Reference implementations (not project dependencies, used for verification):
```bash
git clone https://github.com/azshue/TPT refs/tpt
git clone https://github.com/kdiaaa/tda refs/tda
```

## Pipeline

Run the full pipeline in order with:
```bash
./run_pipeline.sh
```

Or run steps in order:

| Step | Script | Output |
|---|---|---|
| 1. Select base images | `src/select_images.py` | `data/manifests/base_images.json` |
| 2. Apply corruptions | `src/apply_corruptions.py` | `data/corrupted/`, `data/manifests/corrupted_manifest.json` |
| 3. Vanilla CLIP baseline | `src/run_vanilla_clip.py` | `results/vanilla_clip_results.json`, `results/vanilla_clip_embeddings.npz` |
| 4. Run TPT | `src/run_tpt.py` | `results/tpt_results.json` |
| 5. Run TDA | `src/run_tda.py` | `results/tda_results.json` |
| 6. Merge results | `src/build_win_labels.py` | `results/win_labels.json` (incl. `needs_tpt` label, cheap features, timings) |
| 7. Test gate signal (search + stability check) | `src/fit_gate_final.py` | printed to console |
| 8. Compute oracle baseline | `src/compute_oracle.py` | printed to console |
| 9. RL / embedding-based gates + permutation tests | `src/train_rl_gate.py` | printed to console |
| 10. Cost-aware routing curve | `src/cost_aware_routing.py` | `results/cost_curve.png` |

Earlier iterations of the gate-signal search (`fit_gate_pilot_v2/v3/v4.py`, `check_gb_stability.py`) are preserved in `src/archive/` for reference — `fit_gate_final.py` consolidates the full search and the winning result.

Dataset used: [Imagewoof](https://github.com/fastai/imagenette) (10 dog breeds), scaled from 150 to 1,000 images, corrupted with gaussian blur/noise. Switched from Imagenette after finding its classes too easy to distinguish for corruption to matter (92-97% baseline accuracy even at high severity).


## Repo structure
```
data/
├── raw/            # downloaded datasets (gitignored)
├── manifests/       # image selection + corruption metadata
└── corrupted/       # generated corrupted images (gitignored, regenerable)
results/             # per-stage outputs (JSON) + plots
src/                 # pipeline scripts
├── archive/         # superseded iterations of the gate-signal search, kept for reference
refs/                # cloned reference implementations (gitignored)
spec/                # project spec (to be added later)
```

## Notes

- TPT and TDA implementations were built independently and verified line-by-line against their official repos.
- `imagecorruptions` was replaced with a direct `scikit-image` implementation due to unresolved packaging issues (see relevant commit for details).