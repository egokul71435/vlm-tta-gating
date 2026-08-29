# Meta-Learned Test-Time Adaptation: Strategy Selection for VLMs

Pilot project exploring whether a lightweight, meta-learned gate can predict, per image, which test-time adaptation (TTA) strategy (TPT vs. TDA) will work better for a CLIP-based vision-language model under domain shift, without manual per-domain tuning.

## Status: scaling in progress -> entropy signal trending upward, still inconclusive (n=150 → 300 → 500)

| n | Vanilla | TPT | TDA | Oracle | Oracle gap | Disagreement cases | Entropy-gate LOO acc | Majority baseline |
|---|---|---|---|---|---|---|---|---|
| 150 | 57.3% | 56.7% | 61.3% | 66.7% | 5.3pt | 23 | 56.5% | 65.2% |
| 300 | 56.3% | 58.0% | 61.0% | 64.7% | 3.7pt | 31 | 54.8% | 64.5% |
| 500 | 57.0% | 56.4% | 61.6% | 68.2% | 6.6pt | 92 | 60.9% | 64.1% |

**Read:** TDA consistently outperforms TPT and vanilla CLIP across all scales tested. The entropy-based gate is trending toward the majority-class baseline as data scales (gap has narrowed from -8.7pt at n=150 to -3.2pt at n=500), suggesting the earlier inconclusive results were at least partly a data-volume limitation. Not yet conclusive — the trend needs to continue at larger scale to confirm entropy is a real, usable signal rather than noise settling down.

A multi-feature version (entropy + confidence + corruption type) has not consistently outperformed entropy alone across scales and likely overfits at these sample sizes; entropy alone remains the more trustworthy signal for now.

Methodology note: `fit_gate_pilot.py` and `fit_gate_pilot_v2.py` were found to disagree on entropy-only accuracy (63.0% vs. 60.9%) due to a feature-scaling inconsistency between the two scripts. Standardized both to apply `StandardScaler` before fitting — 60.9% is the corrected, trustworthy number.

Note: n=150/300/500 are nested samples (same random seed, larger sets are supersets of smaller ones), not independent replications.

**Next:** continue scaling toward ~800-1000 images to see whether the entropy signal fully closes the gap to baseline, or plateaus below it.


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
| 7. Test gate signal | `src/fit_gate_pilot.py` | `results/entropy_vs_winner.png` |
| 8. Compute oracle baseline | `src/compute_oracle.py` | printed to console |

Dataset used: [Imagewoof](https://github.com/fastai/imagenette) (10 dog breeds), 150 images, corrupted with gaussian blur/noise. Switched from Imagenette after finding its classes too easy to distinguish for corruption to matter (92-97% baseline accuracy even at high severity).

## Repo structure

```
data/
├── raw/            # downloaded datasets (gitignored)
├── manifests/       # image selection + corruption metadata
└── corrupted/       # generated corrupted images (gitignored, regenerable)
results/             # per-stage outputs (JSON) + plots
src/                 # pipeline scripts
refs/                # cloned reference implementations (gitignored)
spec/                # project spec (to be added later) 
```

## Notes

- TPT and TDA implementations were built independently and verified line-by-line against their official repos.
- `imagecorruptions` was replaced with a direct `scikit-image` implementation due to unresolved packaging issues (see relevant commit for details).