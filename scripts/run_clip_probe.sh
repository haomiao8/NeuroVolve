#!/bin/bash
# Zero-shot CLIP category probe of the top images per subject and region.
#   bash scripts/run_clip_probe.sh IMAGE_ROOT [OUTPUT_DIR]
# IMAGE_ROOT/{neurovolve,predicted,measured}/s{S}_{ffa,eba,ppa,mfswords}/*.png
set -euo pipefail
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 IMAGE_ROOT [OUTPUT_DIR]" >&2
    exit 2
fi
image_root="$1"
output_dir="${2:-results/clip_probe}"

python -m neurovolve.clip_probe.probe --image_root "$image_root" --output_dir "$output_dir"
