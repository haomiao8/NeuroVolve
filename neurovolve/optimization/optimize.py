"""NeuroVolve: optimize a Q-Former embedding to drive a target brain region, then decode it to images.

Starting from the Q-Former embedding of a seed image, the embedding is updated with Adam to
maximize (or minimize) the predicted response of the encoder outputs selected by a mask (voxels, or
ROI-averaged responses for a region-level encoder). Every intermediate embedding is kept; selected
iterations are decoded with BLIP-Diffusion (50 steps, guidance 7.5, empty prompt).

The mask is a .npy array with one entry per encoder output, for example:

    import json, numpy as np
    config = json.load(open("checkpoints/encoder_voxel/subj1/model_config.json"))
    mask = np.zeros(config["output_dim"], dtype=bool)
    start, end = config["output_layout"]["FFA1"]    # outputs of each training ROI, or any indices you like
    mask[start:end] = True
    np.save("ffa1_mask.npy", mask)

Output layout, for every seed image {image_id}.png in --image_path:

    OUTPUT_DIR/{mode}/{image_id}/{image_id}.png                 copy of the seed image
    OUTPUT_DIR/{mode}/{image_id}/results.pt                     embeddings + predicted activation per iteration
    OUTPUT_DIR/{mode}/{image_id}/random_{seed}/initial_image.png
    OUTPUT_DIR/{mode}/{image_id}/random_{seed}/iter_{i:05d}_activation_{a:.4f}.png
    OUTPUT_DIR/{mode}/{image_id}/random_{seed}/optimization_progress.png

Usage:
    python -m neurovolve.optimization.optimize \
        --model_dir checkpoints/encoder_voxel/subj1 --mask ffa1_mask.npy \
        --image_path seed_images --output_dir results/s1_ffa
"""
import argparse
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from neurovolve.blip_diffusion import generate_image_from_embedding, get_blip_embed, init_pipe, load_image
from neurovolve.blip_encoder.model import BlipEncoder

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_mask(path, output_dim):
    """
    Load a mask over the encoder outputs from a .npy file of shape (output_dim,).

    Boolean / 0-1 masks select the target outputs; the objective is their mean predicted response.
    Real-valued masks weight the outputs: the objective is sum(w * prediction) / sum(|w|), so negative
    weights penalize a region (e.g. +1 on region A and -1 on region B drives A while suppressing B).
    """
    mask = torch.as_tensor(np.load(path), dtype=torch.float32)
    if mask.shape != (output_dim,):
        raise ValueError(f"Mask has shape {tuple(mask.shape)}, but the encoder has {output_dim} outputs")
    if not mask.abs().sum() > 0:
        raise ValueError("Mask selects no outputs")
    return mask


def objective(encoder, embedding, mask):
    """Mask-weighted mean predicted response: sum(w * prediction) / sum(|w|)."""
    return (encoder(embedding) * mask).sum(dim=1).mean() / mask.abs().sum()


def optimize(encoder, mask, initial_embedding, mode="maximize", num_iterations=300, lr=0.01):
    """
    Adam on the embedding to maximize (or minimize) the mask-weighted predicted response.

    Returns:
        embeddings: (num_iterations + 1, 1, 16, 768) float16, the embedding before each step and the final one
        activations: (num_iterations + 1,) objective value of each of them
    """
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    mask = mask.to(encoder.device)
    embedding = initial_embedding.clone().detach().to(encoder.device, dtype=torch.float32).requires_grad_(True)
    optimizer = torch.optim.Adam([embedding], lr=lr)

    embeddings, activations = [embedding.detach().half().cpu()], []
    for _ in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        activation = objective(encoder, embedding, mask)
        loss = -activation if mode == "maximize" else activation
        loss.backward()
        optimizer.step()
        activations.append(activation.item())
        embeddings.append(embedding.detach().half().cpu())
    with torch.no_grad():
        activations.append(objective(encoder, embedding, mask).item())
    return torch.stack(embeddings), torch.tensor(activations)


def find_elbow_points(activations, elbow_percentages, mode="maximize"):
    """First iteration reaching each percentage of the activation range, plus first and last iteration."""
    activations = np.asarray(activations)
    low, high = activations.min(), activations.max()
    indices = []
    for percentage in elbow_percentages:
        if mode == "maximize":
            reached = np.where(activations >= low + percentage / 100.0 * (high - low))[0]
        else:
            reached = np.where(activations <= high - percentage / 100.0 * (high - low))[0]
        indices.append(int(reached[0]) if len(reached) else len(activations) - 1)
    return sorted(set([0] + indices + [len(activations) - 1]))


def image_name(i, activations):
    return "initial_image.png" if i == 0 else f"iter_{i:05d}_activation_{activations[i].item():.4f}.png"


def generate_images(pipe, embeddings, activations, iterations, output_dir, seed):
    """Decode the embeddings at `iterations` with one diffusion seed into OUTPUT_DIR/random_{seed}/."""
    seed_dir = os.path.join(output_dir, f"random_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    for i in tqdm(iterations, desc=f"Generating images (seed {seed})"):
        with torch.no_grad():
            image = generate_image_from_embedding(pipe, embeddings[i], random_seed=seed)
        image.save(os.path.join(seed_dir, image_name(i, activations)))
    return seed_dir


def plot_progress(activations, iterations, seed_dir, seed_image):
    """Predicted response over iterations, with the decoded images of `iterations` underneath."""
    values = activations.numpy()
    fig = plt.figure(figsize=(max(8, 2.5 * len(iterations)), 7))
    grid = fig.add_gridspec(2, len(iterations), height_ratios=[2, 1], hspace=0.3)
    ax = fig.add_subplot(grid[0, :])
    ax.plot(values, color="black", linewidth=2)
    ax.scatter(iterations, values[iterations], color="red", edgecolor="darkred", s=80, zorder=5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Predicted response")
    ax.grid(alpha=0.3)
    for k, i in enumerate(iterations):
        path = seed_image if i == 0 else os.path.join(seed_dir, image_name(i, activations))
        thumb = fig.add_subplot(grid[1, k])
        thumb.axis("off")
        thumb.imshow(Image.open(path))
        thumb.set_title(f"iter {i}", fontsize=9)
    fig.savefig(os.path.join(seed_dir, "optimization_progress.png"), bbox_inches="tight")
    plt.close(fig)


def main(args):
    encoder = BlipEncoder.load(args.model_dir)
    mask = load_mask(args.mask, encoder.output_dim)
    print(f"Mask: {int((mask != 0).sum())} of {encoder.output_dim} encoder outputs")
    pipe = init_pipe()

    files = sorted((f for f in os.listdir(args.image_path) if f.lower().endswith(IMAGE_EXTENSIONS)),
                   key=lambda f: (0, int(os.path.splitext(f)[0])) if os.path.splitext(f)[0].isdigit() else (1, f))
    for file_name in files:
        image_id = os.path.splitext(file_name)[0]
        print(f"Optimizing {file_name}")
        result_dir = os.path.join(args.output_dir, args.mode, image_id)
        os.makedirs(result_dir, exist_ok=True)
        seed_image = os.path.join(result_dir, file_name)
        shutil.copy(os.path.join(args.image_path, file_name), seed_image)

        initial = get_blip_embed(pipe, load_image(seed_image), category=[""]).unsqueeze(0)
        embeddings, activations = optimize(encoder, mask, initial, args.mode, args.num_iterations, args.lr)
        if args.image_save_interval:
            iterations = sorted(set([0] + list(range(args.image_save_interval, len(embeddings),
                                                     args.image_save_interval))))
        else:
            iterations = find_elbow_points(activations, args.elbow_percentages, args.mode)
        torch.save({"embeddings": embeddings, "activations_history": activations, "decoded_iterations": iterations,
                    "mode": args.mode, "mask": mask, "model_dir": args.model_dir},
                   os.path.join(result_dir, "results.pt"))

        seeds = args.generation_seeds or [random.randint(1, 1000000) for _ in range(args.num_seeds)]
        for seed in seeds:
            seed_dir = generate_images(pipe, embeddings, activations, iterations, result_dir, seed)
            plot_progress(activations, iterations, seed_dir, seed_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with the trained encoder (model.chkpt and model_config.json)")
    parser.add_argument("--mask", type=str, required=True,
                        help="Path to a .npy array of shape (n_outputs,): which encoder outputs to drive "
                             "(boolean), or a weight per output (see load_mask)")
    parser.add_argument("--mode", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--image_path", type=str, required=True, help="Directory of seed images")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--elbow_percentages", nargs="+", type=float, default=[100],
                        help="Decode the iterations first reaching these percentages of the activation change "
                             "(plus the first and last iteration)")
    parser.add_argument("--image_save_interval", type=int, default=None,
                        help="Instead, decode every N iterations")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of random diffusion seeds per image")
    parser.add_argument("--generation_seeds", nargs="+", type=int, default=None,
                        help="Explicit diffusion seeds (overrides --num_seeds)")
    main(parser.parse_args())
