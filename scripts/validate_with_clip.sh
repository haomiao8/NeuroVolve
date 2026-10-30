#!/bin/bash
# Score every optimization run in OUTPUT_DIR with a CLIP validation encoder (mean over the given ROIs).
# Writes RUN_DIR/encoder_predicted_response/{ROIS}.npz for each random_{seed} run directory.
#   bash scripts/validate_with_clip.sh CHECKPOINT OUTPUT_DIR ROI [ROI ...]
#   e.g. bash scripts/validate_with_clip.sh checkpoints/clip_encoder/subj1/00100.chkpt results/s1_ffa FFA1 FFA2
set -euo pipefail
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 CHECKPOINT OUTPUT_DIR ROI [ROI ...]" >&2
    exit 2
fi
checkpoint="$1"
output_dir="$2"
shift 2

for run_dir in "$output_dir"/*/*/random_*; do
    python -m neurovolve.clip_encoder.predict \
        --checkpoint "$checkpoint" \
        --rois "$@" \
        --run_dir "$run_dir"
done
