"""Predict ROI responses of images with the CLIP validation encoder.

Two modes:

  --run_dir RUN_DIR   a random_{seed} directory written by neurovolve.optimization.optimize: predicts the
                      seed image (iteration 0) and every RUN_DIR/iter_*.png, and writes
                      RUN_DIR/encoder_predicted_response/{ROIS}.npz
                      with arrays `iters`, `pred` and `paths`.
  --images ...        image files and/or directories; writes a CSV (--output) with one row per image.

The prediction of an image is the mean over --rois of the predicted ROI responses. ROI names are read
from the checkpoint; for checkpoints that do not store them, give the output order with --checkpoint_rois.

Usage:
    python -m neurovolve.clip_encoder.predict --rois FFA1 FFA2 \
        --checkpoint checkpoints/clip_encoder/subj1/00100.chkpt \
        --run_dir results/s1_ffa/maximize/1004/random_123456
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import torch

from neurovolve.clip_encoder.model import CLIPFeatureExtractor, CLIPROIEncoder, predict_images

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def run_dir_images(run_dir):
    """(iterations, paths): the seed image as iteration 0, then every decoded iteration of the run."""
    parent = os.path.dirname(run_dir)
    image_id = os.path.basename(parent)
    seed_images = [p for p in glob.glob(os.path.join(parent, f"{image_id}.*")) if p.lower().endswith(IMAGE_EXTENSIONS)]
    if not seed_images:
        raise FileNotFoundError(f"No seed image {image_id}.png in {parent}")
    iter_images = {int(re.search(r"iter_(\d+)_activation", os.path.basename(p)).group(1)): p
                   for p in glob.glob(os.path.join(run_dir, "iter_*_activation_*.png"))}
    iter_images.pop(0, None)
    iterations = [0] + sorted(iter_images)
    return iterations, [seed_images[0]] + [iter_images[i] for i in iterations[1:]]


def expand_images(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths += sorted(os.path.join(item, f) for f in os.listdir(item) if f.lower().endswith(IMAGE_EXTENSIONS))
        else:
            paths.append(item)
    return paths


def main(args):
    encoder = CLIPROIEncoder.load(args.checkpoint, device=args.device)
    rois = args.checkpoint_rois or encoder.rois
    if not rois:
        raise ValueError("This checkpoint does not store its ROI names; pass them in output order with --checkpoint_rois")
    if len(rois) != encoder.linear.out_features:
        raise ValueError(f"{len(rois)} ROI names for a checkpoint with {encoder.linear.out_features} outputs")
    missing = [r for r in args.rois if r not in rois]
    if missing:
        raise ValueError(f"ROIs {missing} are not predicted by this encoder; available: {rois}")
    roi_idx = [rois.index(r) for r in args.rois]
    extractor = CLIPFeatureExtractor(device=args.device)

    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        iterations, paths = run_dir_images(run_dir)
        pred = predict_images(paths, encoder, extractor, args.batch_size)[:, roi_idx].mean(axis=1)
        out_dir = os.path.join(run_dir, "encoder_predicted_response")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{'_'.join(args.rois)}.npz")
        np.savez(out, iters=np.array(iterations), pred=pred, paths=np.array(paths))
        for it, value, path in zip(iterations, pred, paths):
            print(f"iter {it:>3}  pred {value:+.4f}  {os.path.basename(path)}")
    else:
        paths = expand_images(args.images)
        pred = predict_images(paths, encoder, extractor, args.batch_size)[:, roi_idx].mean(axis=1)
        out = args.output or "clip_encoder_predictions.csv"
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", f"pred_{'_'.join(args.rois)}"])
            writer.writerows(zip(paths, pred))
    print("saved", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rois", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True, help="CLIP encoder checkpoint, e.g. .../00100.chkpt")
    parser.add_argument("--checkpoint_rois", nargs="+", default=None,
                        help="ROI of each checkpoint output, in order (only for checkpoints that do not store them)")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run_dir")
    inputs.add_argument("--images", nargs="+")
    parser.add_argument("--output", default=None, help="CSV path for --images mode")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
