"""Train the Q-Former (BLIP-based) encoding model for one subject.

Two variants (see neurovolve/blip_encoder/model.py):

    --level voxel    voxel-wise responses of the given ROIs, linear readout (h = 512)
    --level region   voxel-averaged response of each given ROI, nonlinear readout (h = 256)

The encoder is trained on the BLIP-Diffusion Q-Former embeddings of the stimuli (the Q-Former itself
is frozen). They can be given in two ways:

    --embeddings FILE     precomputed embeddings (n_stimuli, 16, 768), e.g. from
                          `python -m neurovolve.blip_encoder.extract_embeddings` -- fast, recommended
    (no --embeddings)     computed here from the stimulus images (S{S}_stimuli_227.h5py) with a float16
                          BLIP-Diffusion pipeline -- slower, needs the pipeline on the GPU; add
                          --save_embeddings FILE to keep them for the next run

A random --val_size (default 500) of the subject's own images is
held out for validation, and the MLP and readout are trained with MSE on the rest of them, with the
order of the 16 Q-Former tokens randomly permuted for every training sample (--no-shuffle_tokens to
disable). After every epoch the model is scored on the validation images (mean over ROIs of the median
per-output correlation); training stops when it has not improved for --patience epochs, and the best
epoch is evaluated once on the NSD shared1000 images (test), which are never used for training or
model selection.

Outputs in --output_dir:
    model.chkpt, model_config.json   the best epoch's encoder (the config records which outputs belong to which ROI)
    eval.h5py                        test predictions / targets / per-output r for each ROI
    training_log.json                validation score per epoch, best epoch, test score

Usage:
    python -m neurovolve.blip_encoder.extract_embeddings --subject 1 --nsd_root $NSD_ROOT --output_dir features/qformer
    python -m neurovolve.blip_encoder.train --subject 1 --nsd_root $NSD_ROOT --level voxel \
        --rois V1v V1d V2v V2d hV4 FFA1 FFA2 EBA PPA --output_dir checkpoints/encoder/subj1 \
        --embeddings features/qformer/subj1_qformer_embeddings.pt
"""
import argparse
import json
import os
from time import time

import h5py
import numpy as np
import torch
from scipy.stats import pearsonr

from neurovolve.blip_encoder.extract_embeddings import extract_qformer_embeddings, load_qformer_embeddings, save_qformer_embeddings
from neurovolve.blip_encoder.model import VARIANTS, BlipEncoder
from neurovolve.dataloader import NeuralDataset, make_dataloader


def correlation(pred, target):
    """Pearson r per column, skipping columns where it is undefined."""
    r_list = []
    for i in range(pred.shape[-1]):
        r, p = pearsonr(pred[:, i], target[:, i])
        if np.isnan(p):
            continue
        r_list.append(r)
    return np.array(r_list)


def get_embeddings(args):
    """Q-Former embeddings of all of the subject's stimuli: loaded from --embeddings, or computed from the images."""
    if args.embeddings is not None:
        print(f"Loading precomputed Q-Former embeddings from {args.embeddings}")
        return load_qformer_embeddings(args.embeddings)
    print("No --embeddings given: computing the Q-Former embeddings from the stimulus images")
    embeddings = extract_qformer_embeddings(args.nsd_root, args.subject)
    if args.save_embeddings is not None:
        save_qformer_embeddings(embeddings, args.save_embeddings)
    return embeddings


def shuffle_tokens(embeddings):
    """Randomly permute the 16 Q-Former tokens of every sample (training augmentation).

    The query tokens are not image patches, so their order carries no spatial meaning; training on
    random orders makes the encoder (nearly) invariant to it, which keeps the optimization from
    exploiting order-specific directions.
    """
    batch, num_tokens, dim = embeddings.shape
    order = torch.argsort(torch.rand(batch, num_tokens), dim=1)
    return embeddings.gather(1, order.unsqueeze(-1).expand(-1, -1, dim))


@torch.no_grad()
def predict(model, embeddings, batch_size=64):
    model.eval()
    return torch.cat([model(embeddings[i:i + batch_size]).cpu()
                      for i in range(0, len(embeddings), batch_size)]).numpy()


def evaluate(model, dataset, save_path=None, verbose=True):
    """Per-output correlations on a dataset, grouped by ROI; returns the mean over ROIs of the median r."""
    pred = predict(model, dataset.features)
    targets = dataset.targets.float().numpy()
    h5 = h5py.File(save_path, "w") if save_path else None
    medians = []
    for roi, (start, end) in model.output_layout.items():
        corr = correlation(pred[:, start:end], targets[:, start:end])
        medians.append(float(np.median(corr)))
        if verbose:
            print(f"  {roi}: median r = {medians[-1]:.4f}")
        if h5 is not None:
            group = h5.require_group(roi)
            group.create_dataset("prediction", data=pred[:, start:end])
            group.create_dataset("target", data=targets[:, start:end])
            group.create_dataset("accuracy", data=corr)
    if h5 is not None:
        h5.close()
    return float(np.mean(medians))


def train(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    embeddings = get_embeddings(args)
    split = dict(rois=args.rois, features=embeddings, level=args.level,
                 val_size=args.val_size, split_seed=args.split_seed)
    train_set = NeuralDataset(args.nsd_root, args.subject, "train", **split)
    val_set = NeuralDataset(args.nsd_root, args.subject, "val", **split)
    test_set = NeuralDataset(args.nsd_root, args.subject, "test", **split)
    print(f"S{args.subject}: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test images, "
          f"{train_set.num_outputs} outputs for {len(args.rois)} ROIs ({args.level}-wise)")

    variant = dict(VARIANTS[args.level])
    if args.hidden_dim is not None:
        variant["hidden_dim"] = args.hidden_dim
    model = BlipEncoder(output_dim=train_set.num_outputs, dropout=0.1, **variant)
    model.output_layout = train_set.output_layout

    loader = make_dataloader(train_set, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    best_path = os.path.join(args.output_dir, "model.chkpt")
    log = {"val_score": {}, "best_epoch": None}
    best_score, patience_counter = -float("inf"), 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss, start_time = 0.0, time()
        for it, (emb, target) in enumerate(loader, start=1):
            if args.shuffle_tokens:
                emb = shuffle_tokens(emb)
            optimizer.zero_grad()
            loss = criterion(model(emb), target.to(model.device, dtype=torch.float32))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val_score = evaluate(model, val_set, verbose=False)
        log["val_score"][epoch] = val_score
        print(f"epoch {epoch}: loss {total_loss / it:.4f}, val {val_score:.4f} ({time() - start_time:.1f}s)")
        if args.save_epochs and epoch in args.save_epochs:
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save(os.path.join(epoch_dir, "model.chkpt"), epoch=epoch, val_score=val_score)
        if val_score > best_score:
            best_score, log["best_epoch"], patience_counter = val_score, epoch, 0
            model.save(best_path, epoch=epoch, val_score=val_score)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    print(f"Best epoch {log['best_epoch']} (val {best_score:.4f}); evaluating it on the test images")
    model = BlipEncoder.load(args.output_dir, device=model.device)
    log["test_score"] = evaluate(model, test_set, os.path.join(args.output_dir, "eval.h5py"))
    print(f"Test: mean over ROIs of median r = {log['test_score']:.4f}")
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--nsd_root", type=str, default=os.environ.get("NSD_ROOT"), required=os.environ.get("NSD_ROOT") is None)
    parser.add_argument("--rois", type=str, nargs="+", required=True,
                        help="ROIs to model; each needs a response file (see neurovolve/dataloader.py)")
    parser.add_argument("--level", type=str, default="voxel", choices=list(VARIANTS),
                        help="voxel: voxel-wise outputs, linear readout; region: one ROI-averaged output per ROI, "
                             "nonlinear readout")
    parser.add_argument("--output_dir", type=str, required=True)
    image_input = parser.add_mutually_exclusive_group()
    image_input.add_argument("--embeddings", type=str, default=None,
                             help="Precomputed Q-Former embeddings of the subject's stimuli (.pt or .npy, shape "
                                  "(n_stimuli, 16, 768), row i = NSD image index i), e.g. from "
                                  "neurovolve.blip_encoder.extract_embeddings. Without it they are computed from the images")
    image_input.add_argument("--save_embeddings", type=str, default=None,
                             help="When computing the embeddings from the images, also save them to this file "
                                  "(pass it as --embeddings next time)")
    parser.add_argument("--num_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop when the validation score has not improved for this many epochs")
    parser.add_argument("--val_size", type=int, default=500,
                        help="Validation images (random, --split_seed); the subject's other images are for training")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Override the variant's hidden size (voxel: 512, region: 256)")
    parser.add_argument("--save_epochs", type=int, nargs="+", default=None,
                        help="Also save the model after these epochs, in OUTPUT_DIR/epoch_{N}/")
    parser.add_argument("--shuffle_tokens", action=argparse.BooleanOptionalAction, default=True,
                        help="Randomly permute the 16 Q-Former tokens of each training sample (default: on)")
    parser.add_argument("--seed", type=int, default=None)
    train(parser.parse_args())
