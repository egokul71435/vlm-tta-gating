# Meta-Learned Test-Time Adaptation: Strategy Selection for VLMs

Pilot project exploring whether a lightweight, meta-learned gate can predict, per image, which test-time adaptation (TTA) strategy (TPT vs. TDA) will work better for a CLIP-based vision-language model under domain shift, without manual per-domain tuning.

## Status: pilot phase in progress

A small end-to-end pilot has been run and validated.

**TL;DR of pilot results:** TPT and TDA meaningfully disagree on which images they get right (23/150 disagreement cases), which supports the core hypothesis that a gate has something to learn. A simple logistic-regression gate using pre-adaptation entropy (and confidence, corruption type) did not beat a majority-class baseline at this sample size, likely a data-volume limitation rather than evidence the signal doesn't exist. Next step is deciding upon scaling.

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

Run in order:

| Step | Script | Output |
|---|---|---|
| 1. Select base images | `src/select_images.py` | `data/manifests/base_images.json` |
| 2. Apply corruptions | `src/apply_corruptions.py` | `data/corrupted/`, `data/manifests/corrupted_manifest.json` |
| 3. Vanilla CLIP baseline | `src/run_vanilla_clip.py` | `results/vanilla_clip_results.json` |
| 4. Run TPT | `src/run_tpt.py` | `results/tpt_results.json` |
| 5. Run TDA | `src/run_tda.py` | `results/tda_results.json` |
| 6. Merge results | `src/build_win_labels.py` | `results/win_labels.json` |
| 7. Test gate signal | `src/fit_gate_pilot.py` | `results/entropy_vs_winner.png` |

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