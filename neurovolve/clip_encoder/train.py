"""Train the CLIP validation encoder (linear ROI-wise model) for one subject.

Inputs are the precomputed normalized CLIP embeddings (neurovolve.clip_encoder.extract_features);
targets are the voxel-averaged responses of each ROI given with --rois (one output per ROI). The
linear layer is trained on all of the subject's non-shared images (train + val; there is no early
stopping) for 100 epochs with AdamW and an exponentially decaying learning rate; the
last three epochs are saved, and the final checkpoint (00100.chkpt) is the one used for validation.
Each checkpoint stores the ROI order of its outputs. Held-out correlations per ROI are written every
20 epochs to {epoch}_eval.h5py.

Usage:
    python -m neurovolve.clip_encoder.train --subject 1 --nsd_root $NSD_ROOT \
        --rois V1v V1d V2v V2d hV4 FFA1 FFA2 EBA PPA L_amygdala R_amygdala \
        --feature_dir features/clip --output_dir checkpoints/clip_encoder/subj1
"""
import argparse
import os
from time import time

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr

from neurovolve.clip_encoder.extract_features import feature_path
from neurovolve.clip_encoder.model import CLIPROIEncoder
from neurovolve.dataloader import NeuralDataset, make_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, features, targets, rois, save_path):
    model.eval()
    pred = model(features.to(device)).cpu().numpy()
    targets = targets.float().numpy()
    with h5py.File(save_path, "w") as f:
        for i, roi in enumerate(rois):
            r, _ = pearsonr(pred[:, i], targets[:, i])
            print(f"  {roi}: r = {r:.4f}")
            group = f.require_group(roi)
            group.create_dataset("prediction", data=pred[:, i:i + 1])
            group.create_dataset("target", data=targets[:, i:i + 1])
            group.create_dataset("accuracy", data=np.array([r]))
    model.train()


def train(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    rois = args.rois
    os.makedirs(args.output_dir, exist_ok=True)
    features = np.load(feature_path(args.feature_dir, args.subject))
    # fixed schedule without early stopping: train on all subject-specific images, evaluate on the test split
    train_set = NeuralDataset(args.nsd_root, args.subject, "trainval", rois, features, level="region")
    val_set = NeuralDataset(args.nsd_root, args.subject, "test", rois, features, level="region")
    print(f"S{args.subject}: {len(train_set)} train / {len(val_set)} held-out images, {len(rois)} ROIs")

    loader = make_dataloader(train_set, args.batch_size)
    model = CLIPROIEncoder(output_dim=len(rois)).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, args.num_epochs + 1):
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (0.5 ** (epoch / args.num_epochs))
        total_loss, start = 0.0, time()
        for it, (x, y) in enumerate(loader, start=1):
            optimizer.zero_grad()
            loss = criterion(model(x.to(device).float()), y.to(device).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch}: loss {total_loss / it:.4f} ({time() - start:.1f}s)")

        if epoch > args.num_epochs - 3:
            torch.save({"network": model.state_dict(), "rois": rois}, os.path.join(args.output_dir, f"{epoch:05d}.chkpt"))
        if epoch % 20 == 0:
            evaluate(model, val_set.features, val_set.targets, rois, os.path.join(args.output_dir, f"{epoch}_eval.h5py"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--nsd_root", type=str, default=os.environ.get("NSD_ROOT"), required=os.environ.get("NSD_ROOT") is None)
    parser.add_argument("--rois", type=str, nargs="+", required=True,
                        help="ROIs to model, one output each; each needs a response file (see neurovolve/dataloader.py)")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=None)
    train(parser.parse_args())
