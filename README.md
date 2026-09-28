# Meta-Learned Test-Time Adaptation: Strategy Selection for VLMs

Pilot project exploring whether a lightweight, meta-learned gate can predict, per image, which test-time adaptation (TTA) strategy (TPT vs. TDA) will work better for a CLIP-based vision-language model under domain shift, without manual per-domain tuning.

## Status: pilot complete — real signal found (gradient boosting on scalar features; MLP on PCA-compressed embeddings)

| n | Vanilla | TPT | TDA | Oracle | Oracle gap |
|---|---|---|---|---|---|
| 150 | 57.3% | 56.7% | 61.3% | 66.7% | 5.3pt |
| 300 | 56.3% | 58.0% | 61.0% | 64.7% | 3.7pt |
| 500 | 57.0% | 56.4% | 61.6% | 68.2% | 6.6pt |
| 1000 | 58.5% | 64.0% | 62.5% | 69.8% | 5.8pt |

The n=1000 row comes from a seeded run (`torch.manual_seed(42)` in `run_tpt.py`/`run_tda.py`) and is reproducible. The n=150/300/500 rows predate the seeding fix, so TPT/TDA numbers there may shift slightly on a rerun. The qualitative pattern (TDA > vanilla, stable oracle gap) is what to take from them.

**Oracle gap is real and stable** across a 6.7x increase in data (150→1000 images): TPT and TDA disagree often enough, and by enough margin, that a gate correctly routing between them would meaningfully beat either strategy alone.

**Signal search:** pre-adaptation entropy alone (tested with logistic regression, random forest, gradient boosting, k-NN, at 4 data scales) did not reliably beat a majority-class baseline. Adding TPT's internal view-entropy spread (the spread of entropy across TPT's 64 augmented views, before adaptation) as a second feature, combined with gradient boosting specifically, does: **60.7% average LOO accuracy vs. 55.7% majority baseline**, stable across 7 random seeds (std = 0.004). Simpler models (logistic regression) and simpler feature sets (entropy alone) did not find this, it required both the richer feature and a non-linear model.

**One earlier false positive was caught and ruled out:** an initial random forest result (65.5%) did not replicate once TPT/TDA's random augmentation was seeded for reproducibility, a reminder that promising-looking small-sample results need a stability check before being trusted.

**Conclusion:** the strategy-selection opportunity is real (oracle gap), and a modest, suggestive predictive signal exists. Next step is a lightweight learned gate (small MLP / prediction head) rather than full MAML meta-training, plus scaling data and shift types to test whether the signal holds up with more informative examples.

### Update: RL and embedding-based gates

Two further gate designs were tested (`src/train_rl_gate.py`):

- **RL (REINFORCE, contextual-bandit framing)** on the scalar features: stable across seeds but lands at the best-fixed-strategy baseline (63.3-64.1% across runs vs. 64.0% on all 1,000 images; individual runs flipped between "beats" and "doesn't beat" by ~0.1 points, which is noise). No clear gain.
- **Supervised MLP on CLIP image embeddings, PCA-reduced to 5 dims**, trained on the TPT/TDA disagreement cases: **59.8% mean vs. 55.7% baseline, 7/7 seeds beat baseline (std = 0.013)**, comparable to the gradient boosting result above. Raw 512-dim and 30-dim inputs collapsed to the majority class or overfit badly on ~131 examples; 5 dims and the regularization strength were selected by trial and error against the same CV folds, so 59.8% is likely optimistic; the 7-seed check varies network initialization but not the data split.

**Caveat:** both gate results (gradient boosting and the embedding MLP) are measured on the 131 disagreement cases only, where exactly one strategy is right. End-to-end over all 1,000 images, a ~5-point gain on those cases is roughly 6-7 extra correct images, or about +0.7 points over always-TPT, so the practical gain is small so far. Statistically, with 131 cases the standard error on accuracy is ~4 points, so a ~5-point gap over baseline is about one standard error. The 7-seed check varies model initialization, not the data split, and this combination was the best of ~16 feature/model pairs tried, so treat it as suggestive rather than established. Permutation tests (200 label shuffles, same CV procedure) give p = 0.045 for gradient boosting and p = 0.030 for the embedding MLP (single seed, 58.8%). Both are nominally significant, but neither survives correction for the ~16 configurations tried, so they need confirmation on more data. Scaling data and shift types is the next step.

**Baseline definition:** on the 131 disagreement cases, the 55.7% baseline is "always pick TPT" (TPT alone is right on 73 of 131). On all 1,000 images, the baseline is always-TPT at 64.0%.

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
| 6. Merge results | `src/build_win_labels.py` | `results/win_labels.json` |
| 7. Test gate signal (search + stability check) | `src/fit_gate_final.py` | printed to console |
| 8. Compute oracle baseline | `src/compute_oracle.py` | printed to console |
| 9. RL / embedding-based gates + permutation tests | `src/train_rl_gate.py` | printed to console |

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