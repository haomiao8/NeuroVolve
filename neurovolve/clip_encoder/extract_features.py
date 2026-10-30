"""Extract L2-normalized CLIP ViT-H/14 embeddings of a subject's 10000 NSD stimuli.

Writes OUTPUT_DIR/subj{S}_CLIP_embedding_normalized.npy, shape (10000, 1024), indexed by NSD image index.

Usage:
    python -m neurovolve.clip_encoder.extract_features --subject 1 --nsd_root $NSD_ROOT --output_dir features/clip
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from neurovolve.clip_encoder.model import CLIPFeatureExtractor
from neurovolve.dataloader import load_stimuli


def feature_path(feature_dir, subject):
    return os.path.join(feature_dir, f"subj{subject}_CLIP_embedding_normalized.npy")


def main(args):
    extractor = CLIPFeatureExtractor(device=args.device)
    stimuli = load_stimuli(args.nsd_root, args.subject)  # (10000, 3, 227, 227) uint8
    features = np.concatenate([
        extractor(list(stimuli[i:i + args.batch_size])).cpu().numpy()
        for i in tqdm(range(0, len(stimuli), args.batch_size), desc=f"CLIP features S{args.subject}")
    ])
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(feature_path(args.output_dir, args.subject), features)
    print(f"Saved {features.shape} to {feature_path(args.output_dir, args.subject)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--nsd_root", type=str, default=os.environ.get("NSD_ROOT"), required=os.environ.get("NSD_ROOT") is None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
