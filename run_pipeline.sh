#!/bin/bash
set -e  # stop immediately if any step fails

echo "=== 1/8: Selecting base images ==="
time python src/select_images.py

echo -e "\n=== 2/8: Applying corruptions ==="
time python src/apply_corruptions.py

echo -e "\n=== 3/8: Vanilla CLIP baseline ==="
time python src/run_vanilla_clip.py

echo -e "\n=== 4/8: Running TPT ==="
time python src/run_tpt.py

echo -e "\n=== 5/8: Running TDA ==="
time python src/run_tda.py

echo -e "\n=== 6/8: Merging results into win-labels ==="
time python src/build_win_labels.py

echo -e "\n=== 7/8: Testing gate signal (entropy only) ==="
time python src/fit_gate_pilot.py

echo -e "\n=== 7b/8: Testing gate signal (multi-feature) ==="
time python src/fit_gate_pilot_v2.py

echo -e "\n=== 8/8: Computing oracle baseline ==="
time python src/compute_oracle.py

echo -e "\n=== Pipeline complete ==="