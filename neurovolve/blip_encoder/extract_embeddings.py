"""Extract the BLIP-Diffusion Q-Former embeddings of a subject's 10000 NSD stimuli.

The Q-Former encoder is trained on these embeddings, not on the images themselves. Extracting them
needs the BLIP-Diffusion pipeline and a GPU; doing it once and passing the file to training with
--embeddings makes every later training run start immediately (and lets the voxel-wise and
region-wise models of a subject share one file).

Writes OUTPUT_DIR/subj{S}_qformer_embeddings.pt: a float16 tensor of shape (10000, 16, 768), row i
being the embedding of image i of S{S}_stimuli_227.h5py (i.e. indexed by NSD image index, like the
indices in avg_beta_order.npy).

Usage:
    python -m neurovolve.blip_encoder.extract_embeddings --subject 1 --nsd_root $NSD_ROOT --output_dir features/qformer
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from neurovolve.blip_diffusion import get_blip_embed, init_pipe
from neurovolve.dataloader import load_stimuli

NUM_TOKENS, EMBED_DIM = 16, 768


def embedding_path(embedding_dir, subject):
    return os.path.join(embedding_dir, f"subj{subject}_qformer_embeddings.pt")


def extract_qformer_embeddings(nsd_root, subject):
    """Q-Former embeddings (n_stimuli, 16, 768) float16 of all of a subject's stimuli, computed from the images."""
    pipe = init_pipe(torch_dtype=torch.float16)
    stimuli = load_stimuli(nsd_root, subject)
    embeddings = torch.stack([
        get_blip_embed(pipe, torch.from_numpy(img).float(), category="").cpu()
        for img in tqdm(stimuli, desc=f"Q-Former embeddings S{subject}")
    ])
    del pipe
    torch.cuda.empty_cache()
    return embeddings


def load_qformer_embeddings(path):
    """Precomputed Q-Former embeddings from a .pt (torch tensor) or .npy file, checked for shape (n, 16, 768)."""
    if path.endswith(".npy"):
        embeddings = torch.from_numpy(np.load(path))
    else:
        embeddings = torch.load(path, map_location="cpu")
    assert embeddings.dim() == 3 and tuple(embeddings.shape[1:]) == (NUM_TOKENS, EMBED_DIM), \
        f"{path}: expected Q-Former embeddings of shape (n_stimuli, {NUM_TOKENS}, {EMBED_DIM}), got {tuple(embeddings.shape)}"
    return embeddings.to(torch.float16)


def save_qformer_embeddings(embeddings, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(embeddings, path)
    print(f"Saved Q-Former embeddings {tuple(embeddings.shape)} to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--nsd_root", type=str, default=os.environ.get("NSD_ROOT"), required=os.environ.get("NSD_ROOT") is None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    save_qformer_embeddings(extract_qformer_embeddings(args.nsd_root, args.subject), embedding_path(args.output_dir, args.subject))
