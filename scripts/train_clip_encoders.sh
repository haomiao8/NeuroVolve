#!/bin/bash
# Extract CLIP ViT-H/14 features of the stimuli and train the CLIP validation encoder for one or more
# subjects (one ROI-averaged output per ROI).
#   NSD_ROOT=/path/to/nsd ROIS="V1v V1d FFA1 FFA2 PPA L_amygdala R_amygdala" bash scripts/train_clip_encoders.sh SUBJECT [SUBJECT ...]
set -euo pipefail
: "${NSD_ROOT:?set NSD_ROOT to the preprocessed data (see README)}"
: "${ROIS:?set ROIS to the ROIs to model, e.g. ROIS=\"V1v V1d FFA1 FFA2\"}"
if [ "$#" -lt 1 ]; then
    echo "Usage: ROIS=\"...\" $0 SUBJECT [SUBJECT ...]" >&2
    exit 2
fi
CKPT_DIR="${CKPT_DIR:-checkpoints}"
FEATURE_DIR="${FEATURE_DIR:-features/clip}"
read -r -a rois <<< "$ROIS"

for subject in "$@"; do
    if [ ! -f "$FEATURE_DIR/subj${subject}_CLIP_embedding_normalized.npy" ]; then
        python -m neurovolve.clip_encoder.extract_features \
            --subject "$subject" --nsd_root "$NSD_ROOT" --output_dir "$FEATURE_DIR"
    fi
    python -m neurovolve.clip_encoder.train \
        --subject "$subject" \
        --nsd_root "$NSD_ROOT" \
        --rois "${rois[@]}" \
        --feature_dir "$FEATURE_DIR" \
        --output_dir "$CKPT_DIR/clip_encoder/subj${subject}"
done
