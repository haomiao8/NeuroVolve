#!/bin/bash
# Generate images that drive the encoder outputs selected by MASK (a .npy array, one entry per
# encoder output), with the settings used for the paper (300 Adam steps, lr 0.01, one diffusion
# seed; decodes the seed, the iteration reaching the maximum and the final iteration).
#   bash scripts/run_optimization.sh MODEL_DIR MASK SEED_DIR OUTPUT_DIR
#   e.g. bash scripts/run_optimization.sh checkpoints/encoder_voxel/subj1 ffa_mask.npy seed_images results/s1_ffa
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 MODEL_DIR MASK SEED_DIR OUTPUT_DIR" >&2
    exit 2
fi

python -m neurovolve.optimization.optimize \
    --model_dir "$1" \
    --mask "$2" \
    --mode maximize \
    --image_path "$3" \
    --output_dir "$4" \
    --num_iterations 300 \
    --lr 0.01 \
    --elbow_percentages 100 \
    --num_seeds 1
