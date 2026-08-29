# Meta-Learned Test-Time Adaptation: Strategy Selection for VLMs

Pilot project exploring whether a lightweight, meta-learned gate can predict, per image, which test-time adaptation (TTA) strategy (TPT vs. TDA) will work better for a CLIP-based vision-language model under domain shift, without manual per-domain tuning.

## Status: scaling in progress (n=150 → n=300, results trending but still inconclusive)

A small end-to-end pilot (150 images) has been run and validated.

**TL;DR of pilot results (n=150):**
- Vanilla CLIP baseline: 57.3%
- TPT: 56.7% (roughly a wash — 12 images fixed, 13 regressed)
- TDA: 61.3% (net positive — 9 fixed, 3 regressed)
- Oracle (best of TPT/TDA per image): 66.7% — a 5.3-point gap over TDA alone, meaning a perfect gate could meaningfully outperform either fixed strategy
- 23/150 images saw TPT and TDA disagree on correctness — the core signal a gate would need
- A logistic-regression gate using pre-adaptation entropy (and confidence, corruption type) did not beat a majority-class baseline at this sample size — likely a data-volume limitation given the small number of disagreement cases, not evidence the signal doesn't exist

**Scaled to n=300:** vanilla 56.3%, TPT 58.0%, TDA 61.0%, oracle 64.7% (3.7pt gap over TDA) — core accuracy pattern held up, confirming the 150-image result wasn't a small-sample fluke. Disagreement cases grew to 31/300. A 3-feature gate (entropy + confidence + corruption type) improved over the entropy-only version and moved closer to (but still below) majority baseline.

The oracle gap confirms real headroom exists for a gate to capture. Currently scaling up image count and shift-type diversity to get a larger, more conclusive test of the entropy signal.

**Read so far:** more data is helping incrementally but hasn't yet produced a gate that clearly beats naive baselines. Continuing to scale (targeting ~500 images) before deciding whether the signal is real-but-data-limited or genuinely weak.

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