"""Zero-shot CLIP (OpenCLIP CoCa ViT-L/14) category probe of region-preferred images.

Each image is scored against text prompts for the categories face, body, scene and text;
a category's score is its best-matching prompt, the softmax over
categories gives probabilities and the argmax the label. A region's accuracy is the fraction of its
images labelled with the category the region is selective for (FFA -> face, EBA -> body,
PPA -> scene, mfs-words -> text).

Expected input layout (e.g. the top-100 images per subject and region of each image set):

    IMAGE_ROOT/{set}/s{S}_{region}/*.png      region in {ffa, eba, ppa, mfswords}

Outputs in --output_dir:
    clip_probe_per_image.csv      one row per image: set, subject, region, rank, file, label, probabilities
    clip_probe_accuracy.csv       one row per set x subject x region: n, n_correct, accuracy
    clip_probe_accuracy_wide.csv  accuracy table, rows = set x subject (+ mean), columns = regions

Usage:
    python -m neurovolve.clip_probe.probe --image_root top100_images --output_dir results/clip_probe \
        --sets neurovolve predicted measured
"""
import argparse
import csv
import os

import numpy as np
import open_clip
import torch
from PIL import Image

# region -> category that region is selective for
REGION_TARGET = {"ffa": "face", "eba": "body", "ppa": "scene", "mfswords": "text"}

CATEGORY_PROMPTS = {
    "face": ["A face facing the camera", "A photo of a face", "A photo of a human face",
             "A photo of faces", "A photo of a person's face", "A person looking at the camera",
             "People looking at the camera", "A portrait of a person", "A portrait photo"],
    "body": ["A photo of a torso", "A photo of torsos", "A photo of limbs",
             "A photo of bodies", "A photo of a person", "A photo of people"],
    "scene": ["A photo of a bedroom", "A photo of an office", "A photo of a hallway",
              "A photo of a doorway", "A photo of interior design", "A photo of a building",
              "A photo of a house", "A photo of nature", "A photo of landscape",
              "A landscape photo", "A photo of trees", "A photo of grass"],
    "text": ["A photo of words", "A photo of glyphs", "A photo of a glyph",
             "A photo of text", "A photo of numbers", "A photo of a letter",
             "A photo of letters", "A photo of writing", "A photo of text on an object"],
}


class CategoryProbe:
    def __init__(self, categories, device):
        self.device = device
        self.categories = categories
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k")
        self.model = self.model.to(device).eval()
        context_length = getattr(self.model, "context_length", None) or \
            getattr(getattr(self.model, "text", None), "context_length", 77)

        prompts, prompt_category = [], []
        for c, category in enumerate(categories):
            prompts += CATEGORY_PROMPTS[category]
            prompt_category += [c] * len(CATEGORY_PROMPTS[category])
        with torch.no_grad():
            tokens = open_clip.tokenize(prompts, context_length=context_length).to(device)
            self.text_features = self.model.encode_text(tokens, normalize=True)
        prompt_category = torch.tensor(prompt_category, device=device)
        self.category_masks = [prompt_category == c for c in range(len(categories))]

    @torch.no_grad()
    def classify(self, paths):
        """(labels, probs[n, n_categories]) for a list of image paths."""
        images = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in paths]).to(self.device)
        image_features = self.model.encode_image(images, normalize=True)
        logits = 100.0 * image_features @ self.text_features.T
        # a category's score is its best-matching prompt
        scores = torch.stack([logits[:, m].max(dim=1).values for m in self.category_masks], dim=1)
        probs = scores.softmax(dim=-1)
        return [self.categories[i] for i in probs.argmax(dim=1).tolist()], probs.cpu().numpy()


def main(args):
    categories = list(CATEGORY_PROMPTS)
    probe = CategoryProbe(categories, "cuda" if torch.cuda.is_available() else "cpu")
    regions = args.regions

    per_image, summary = [], []
    for set_name in args.sets:
        for subject in args.subjects:
            for region in regions:
                target = REGION_TARGET[region]
                image_dir = os.path.join(args.image_root, set_name, f"s{subject}_{region}")
                files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
                if args.expected_images and len(files) != args.expected_images:
                    print(f"WARNING: {image_dir} has {len(files)} images (expected {args.expected_images})")
                labels, probs = [], []
                for i in range(0, len(files), args.batch_size):
                    lab, pr = probe.classify([os.path.join(image_dir, f) for f in files[i:i + args.batch_size]])
                    labels += lab
                    probs.append(pr)
                probs = np.concatenate(probs)

                n_correct = sum(label == target for label in labels)
                summary.append(dict(set=set_name, subject=subject, region=region, target=target,
                                    n_images=len(files), n_correct=n_correct,
                                    accuracy=round(n_correct / len(files), 4)))
                for rank, (f, label, p) in enumerate(zip(files, labels, probs), start=1):
                    per_image.append(dict(set=set_name, subject=subject, region=region, target=target,
                                          rank=rank, file=f, label=label, correct=int(label == target),
                                          **{f"p_{c}": round(float(v), 4) for c, v in zip(categories, p)}))
                print(f"{set_name:10s} s{subject} {region:8s} -> {target:5s}: "
                      f"{n_correct}/{len(files)} = {n_correct / len(files):.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "clip_probe_per_image.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    with open(os.path.join(args.output_dir, "clip_probe_accuracy.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    accuracy = {(r["set"], r["subject"], r["region"]): r["accuracy"] for r in summary}
    with open(os.path.join(args.output_dir, "clip_probe_accuracy_wide.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["set", "subject"] + [f"{r} ({REGION_TARGET[r]})" for r in regions])
        for set_name in args.sets:
            for subject in args.subjects:
                writer.writerow([set_name, f"s{subject}"] + [accuracy[(set_name, subject, r)] for r in regions])
            writer.writerow([set_name, "mean"] + [
                round(float(np.mean([accuracy[(set_name, s, r)] for s in args.subjects])), 4) for r in regions])
    print(f"Saved CSVs to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sets", nargs="+", default=["neurovolve", "predicted", "measured"])
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--regions", nargs="+", default=list(REGION_TARGET), choices=list(REGION_TARGET))
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--expected_images", type=int, default=100,
                        help="Warn if a folder does not hold this many images (0 = no check)")
    main(parser.parse_args())
