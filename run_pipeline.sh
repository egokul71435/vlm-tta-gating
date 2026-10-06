#!/bin/bash
set -e  # stop immediately if any step fails

echo "=== 1/10: Selecting base images ==="
time python src/select_images.py

echo -e "\n=== 2/10: Applying corruptions ==="
time python src/apply_corruptions.py

echo -e "\n=== 3/10: Vanilla CLIP baseline ==="
time python src/run_vanilla_clip.py

echo -e "\n=== 4/10: Running TPT ==="
time python src/run_tpt.py

echo -e "\n=== 5/10: Running TDA ==="
time python src/run_tda.py

echo -e "\n=== 6/10: Merging results into win-labels ==="
time python src/build_win_labels.py

echo -e "\n=== 7/10: Testing gate signal (entropy only) ==="
time python src/fit_gate_pilot.py

echo -e "\n=== 7b/10: Testing gate signal (full search + stability check) ==="
time python src/fit_gate_final.py

echo -e "\n=== 8/10: Computing oracle baseline ==="
time python src/compute_oracle.py

echo -e "\n=== 9/10: RL gate + PCA-embedding gate ==="
time python src/train_rl_gate.py

echo -e "\n=== 10/10: Cost-aware routing curve ==="
time python src/cost_aware_routing.py

echo -e "\n=== Pipeline complete ==="