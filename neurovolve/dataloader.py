"""Data loading shared by both encoders (the Q-Former encoder and the CLIP validation encoder).

Expected layout under NSD_ROOT (see README):

    NSD_ROOT/responses/subj0{S}/avg_roi_response/avg_beta_order.npy   (n_images,) NSD image index of each row
    NSD_ROOT/responses/subj0{S}/avg_roi_response/avg_beta_{ROI}.npy   (n_images, n_voxels) repeat-averaged betas
    NSD_ROOT/stimuli/S{S}_stimuli_227.h5py                            dataset "stimuli": (10000, 3, 227, 227) uint8

Splits (see split_indices):

    "test"      images with index < 1000 (the NSD shared1000 images), never used for training or model selection
    "val"       a random subset of val_size (default 500) of the subject's other images (fixed split_seed), for early stopping
    "train"     the remaining subject-specific images
    "trainval"  train + val (for models trained without early stopping)

Edit this file to train on data stored differently.

NeuralDataset pairs precomputed image features (Q-Former embeddings or CLIP embeddings, indexed by
NSD image index) with the responses of the chosen ROIs:

    level="voxel"    every voxel of the ROIs, concatenated in the given order
    level="region"   the voxel-averaged response of each ROI (one value per ROI)
"""
import os

import h5py
import numpy as np
import torch

N_SHARED = 1000


def response_dir(nsd_root, subject):
    return os.path.join(nsd_root, "responses", f"subj0{subject}", "avg_roi_response")


def stimuli_path(nsd_root, subject):
    return os.path.join(nsd_root, "stimuli", f"S{subject}_stimuli_227.h5py")


def split_indices(nsd_root, subject, split, val_size=500, split_seed=0):
    """Return (row indices into the response arrays, NSD image indices) of a split (see above)."""
    image_order = np.load(os.path.join(response_dir(nsd_root, subject), "avg_beta_order.npy"))
    assert len(np.unique(image_order)) == len(image_order), "avg_beta_order.npy must not contain repeats"
    test_rows = np.nonzero(image_order < N_SHARED)[0]
    trainval_rows = np.nonzero(image_order >= N_SHARED)[0]
    assert val_size < len(trainval_rows), f"val_size {val_size} >= {len(trainval_rows)} subject-specific images"
    val_rows = np.sort(np.random.default_rng(split_seed).choice(trainval_rows, size=val_size, replace=False))
    if split == "test":
        rows = test_rows
    elif split == "trainval":
        rows = trainval_rows
    elif split == "train":
        rows = np.setdiff1d(trainval_rows, val_rows)
    elif split == "val":
        rows = val_rows
    else:
        raise ValueError(f"Unknown split: {split}")
    return rows, image_order[rows]


def load_roi_responses(nsd_root, subject, roi, rows):
    """Voxel responses of one ROI: (len(rows), n_voxels_in_roi)."""
    return np.load(os.path.join(response_dir(nsd_root, subject), f"avg_beta_{roi}.npy"))[rows]


def load_stimuli(nsd_root, subject):
    """All of a subject's stimuli, (n_stimuli, 3, 227, 227) uint8, indexed by NSD image index."""
    with h5py.File(stimuli_path(nsd_root, subject), "r") as f:
        return f["stimuli"][:]


class NeuralDataset(torch.utils.data.Dataset):
    """(image features, responses) pairs for one subject and split.

    Args:
        nsd_root, subject: where to read the responses (see the layout above).
        split: "train", "val", "test" or "trainval" (see split_indices).
        rois: ROI names; each needs an avg_beta_{ROI}.npy file.
        features: tensor (n_stimuli, ...) of image features, indexed by NSD image index.
        level: "voxel" (all voxels of the ROIs) or "region" (one ROI-averaged response per ROI).
        val_size, split_seed: size and seed of the random validation split (the rest is training).

    Attributes:
        output_layout: ROI -> [start, end) columns of the targets.
    """

    def __init__(self, nsd_root, subject, split, rois, features, level="voxel", val_size=500, split_seed=0):
        assert level in ("voxel", "region"), f"Unknown level: {level}"
        rows, images = split_indices(nsd_root, subject, split, val_size, split_seed)
        blocks = [load_roi_responses(nsd_root, subject, roi, rows) for roi in rois]
        if level == "region":
            blocks = [block.mean(axis=1, keepdims=True) for block in blocks]

        self.output_layout, start = {}, 0
        for roi, block in zip(rois, blocks):
            self.output_layout[roi] = [start, start + block.shape[1]]
            start += block.shape[1]
        assert len(features) > images.max(), \
            f"features have {len(features)} rows but must be indexed by NSD image index (up to {images.max()})"
        # features and targets are stored in float16, as in the original training runs
        self.features = torch.as_tensor(features[images]).to(torch.float16)
        self.targets = torch.from_numpy(np.hstack(blocks)).to(torch.float16)

    @property
    def num_outputs(self):
        return self.targets.shape[1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def make_dataloader(dataset, batch_size=64):
    # images are visited in a fixed order (no shuffling), as in the original training runs
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
