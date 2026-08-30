# Meta-Learned Test-Time Adaptation: Strategy Selection for VLMs

Pilot project exploring whether a lightweight, meta-learned gate can predict, per image, which test-time adaptation (TTA) strategy (TPT vs. TDA) will work better for a CLIP-based vision-language model under domain shift, without manual per-domain tuning.

## Status: pilot complete — real signal found (gradient boosting + entropy + TPT view-spread)

| n | Vanilla | TPT | TDA | Oracle | Oracle gap |
|---|---|---|---|---|---|
| 150 | 57.3% | 56.7% | 61.3% | 66.7% | 5.3pt |
| 300 | 56.3% | 58.0% | 61.0% | 64.7% | 3.7pt |
| 500 | 57.0% | 56.4% | 61.6% | 68.2% | 6.6pt |
| 1000 | 58.5% | 61.6%–64.0%* | 62.5% | 69.1%–69.8%* | 5.8–6.6pt |

\*TPT/TDA use random augmentation; early runs (before a fixed seed was added) show minor run-to-run variance — see Notes.

**Oracle gap is real and stable** across a 6.7x increase in data (150→1000 images): TPT and TDA disagree often enough, and by enough margin, that a gate correctly routing between them would meaningfully beat either strategy alone.

**Signal search:** pre-adaptation entropy alone (tested with logistic regression, random forest, gradient boosting, k-NN, at 4 data scales) did not reliably beat a majority-class baseline. Adding TPT's internal view-entropy spread (the spread of entropy across TPT's 64 augmented views, before adaptation) as a second feature, combined with gradient boosting specifically, does: **60.7% average LOO accuracy vs. 55.7% majority baseline**, stable across 7 random seeds (std = 0.004). Simpler models (logistic regression) and simpler feature sets (entropy alone) did not find this — it required both the richer feature and a non-linear model.

**One earlier false positive was caught and ruled out:** an initial random forest result (65.5%) did not replicate once TPT/TDA's random augmentation was seeded for reproducibility — a reminder that promising-looking small-sample results need a stability check before being trusted.

**Conclusion:** the strategy-selection opportunity is real (oracle gap), and a genuine, if modest, predictive signal exists — but it required a non-linear model and a richer feature than simple entropy to surface. This supports moving to the full MAML-based gate, which can learn this kind of non-linear, multi-signal relationship directly rather than requiring hand-picked features.

Note: n=150/300/500/1000 are nested samples (same seed for image selection), not independent replications. TPT/TDA augmentation is now seeded (`torch.manual_seed(42)`) for reproducibility — some earlier-cited numbers predate this fix and may not exactly reproduce.

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
| 3. Vanilla CLIP baseline | `src/run_vanilla_clip.py` | `results/vanilla_clip_results.json` |
| 4. Run TPT | `src/run_tpt.py` | `results/tpt_results.json` |
| 5. Run TDA | `src/run_tda.py` | `results/tda_results.json` |
| 6. Merge results | `src/build_win_labels.py` | `results/win_labels.json` |
| 7. Test gate signal (search + stability check) | `src/fit_gate_final.py` | printed to console |
| 8. Compute oracle baseline | `src/compute_oracle.py` | printed to console |

Earlier iterations of the gate-signal search (`fit_gate_pilot_v2/v3/v4.py`, `check_gb_stability.py`) are preserved in `src/archive/` for reference — `fit_gate_final.py` consolidates the full search and the winning result.

Dataset used: [Imagewoof](https://github.com/fastai/imagenette) (10 dog breeds), 150 images, corrupted with gaussian blur/noise. Switched from Imagenette after finding its classes too easy to distinguish for corruption to matter (92-97% baseline accuracy even at high severity).

## Repo structure

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