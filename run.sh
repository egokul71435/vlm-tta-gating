#!/usr/bin/env bash
# Run the pipeline for one registered dataset, from the repo root.
#
#   ./run.sh <dataset>                  all stages: clip tda tpt labels route
#   ./run.sh <dataset> clip             only the zero-shot check
#   ./run.sh <dataset> labels route     re-analyze existing results (seconds, no model)
#   N=100 ./run.sh <dataset> clip tda   smoke test on the first 100 images
#
# Each stage's output is also saved to results/<dataset>/logs/<stage>.log.
# For variants (--tag, --n-views 1, legacy TPT flags, ...) call the Python
# scripts directly.

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"

DATASET=${1:?"usage: ./run.sh <dataset> [clip|tda|tpt|labels|route ...]"}
shift
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
  STAGES=(clip tda tpt labels route)
fi

S=src_main/scripts
LIMIT=()
if [ -n "${N:-}" ]; then
  LIMIT=(--n "$N")
fi
LOGS="${RESULTS_ROOT:-results}/$DATASET/logs"
mkdir -p "$LOGS"

for stage in "${STAGES[@]}"; do
  case "$stage" in
    clip)   cmd=(python "$S/run_vanilla_clip.py" --dataset "$DATASET" ${LIMIT[@]+"${LIMIT[@]}"}) ;;
    tda)    cmd=(python "$S/run_tda.py" --dataset "$DATASET" ${LIMIT[@]+"${LIMIT[@]}"}) ;;
    tpt)    cmd=(python "$S/run_tpt.py" --dataset "$DATASET" ${LIMIT[@]+"${LIMIT[@]}"}) ;;
    labels) cmd=(python "$S/build_win_labels.py" --dataset "$DATASET") ;;
    route)  cmd=(python "$S/cost_aware_routing.py" --dataset "$DATASET") ;;
    *) echo "Unknown stage '$stage' (expected clip, tda, tpt, labels or route)" >&2; exit 1 ;;
  esac
  echo "=== [$DATASET] $stage: ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "$LOGS/$stage.log"
done
echo "=== [$DATASET] done: ${STAGES[*]}"