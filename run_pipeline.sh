#!/bin/bash
set -e  # stop immediately if any step fails

echo "=== 1/9: Selecting base images ==="
time python src/select_images.py

echo -e "\n=== 2/9: Applying corruptions ==="
time python src/apply_corruptions.py

echo -e "\n=== 3/9: Vanilla CLIP baseline ==="
time python src/run_vanilla_clip.py

echo -e "\n=== 4/9: Running TPT ==="
time python src/run_tpt.py

echo -e "\n=== 5/9: Running TDA ==="
time python src/run_tda.py

echo -e "\n=== 6/9: Merging results into win-labels ==="
time python src/build_win_labels.py

echo -e "\n=== 7/9: Testing gate signal (entropy only) ==="
time python src/fit_gate_pilot.py

echo -e "\n=== 7b/9: Testing gate signal (full search + stability check) ==="
time python src/fit_gate_final.py

echo -e "\n=== 8/9: Computing oracle baseline ==="
time python src/compute_oracle.py

echo -e "\n=== Pipeline complete ==="