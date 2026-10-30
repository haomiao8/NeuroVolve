#!/bin/bash
# Train the Q-Former encoder for one or more subjects.
#   NSD_ROOT=/path/to/nsd ROIS="V1v V1d FFA1 FFA2 PPA" bash scripts/train_encoders.sh SUBJECT [SUBJECT ...]
# LEVEL=voxel (default; voxel-wise, linear readout) or LEVEL=region (one ROI-averaged output per ROI,
# nonlinear readout), e.g. LEVEL=region ROIS="L_amygdala R_amygdala".
# The Q-Former embeddings of each subject's stimuli are extracted from the images once, into
# $EMBEDDING_DIR (default features/qformer), and reused by every later run (and by both levels).
set -euo pipefail
: "${NSD_ROOT:?set NSD_ROOT to the preprocessed data (see README)}"
: "${ROIS:?set ROIS to the ROIs to model, e.g. ROIS=\"V1v V1d FFA1 FFA2\"}"
if [ "$#" -lt 1 ]; then
    echo "Usage: ROIS=\"...\" $0 SUBJECT [SUBJECT ...]" >&2
    exit 2
fi
LEVEL="${LEVEL:-voxel}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
EMBEDDING_DIR="${EMBEDDING_DIR:-features/qformer}"
read -r -a rois <<< "$ROIS"

for subject in "$@"; do
    embeddings="$EMBEDDING_DIR/subj${subject}_qformer_embeddings.pt"
    if [ ! -f "$embeddings" ]; then
        python -m neurovolve.blip_encoder.extract_embeddings --subject "$subject" --nsd_root "$NSD_ROOT" --output_dir "$EMBEDDING_DIR"
    fi
    python -m neurovolve.blip_encoder.train \
        --subject "$subject" \
        --nsd_root "$NSD_ROOT" \
        --level "$LEVEL" \
        --rois "${rois[@]}" \
        --output_dir "$CKPT_DIR/encoder_${LEVEL}/subj${subject}" \
        --embeddings "$embeddings"
done
