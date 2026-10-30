"""CLIP validation encoder: L2-normalized CLIP ViT-H/14 image embedding -> linear -> ROI-averaged responses.

This encoder is independent of the Q-Former encoder used for optimization and is used to
validate generated images in a different feature space.
"""
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor

CLIP_MODEL = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CLIP_DIM = 1024


class CLIPFeatureExtractor(nn.Module):
    """L2-normalized OpenCLIP ViT-H/14 (LAION-2B) image embeddings."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL)
        self.model = self.model.to(self.device).eval()
        # 224px resize + center crop with the OpenAI CLIP mean/std
        self.processor = CLIPImageProcessor()

    @torch.no_grad()
    def forward(self, images):
        """images: list of PIL images or (3, H, W) uint8 arrays -> (n, 1024) unit-norm embeddings."""
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        return F.normalize(self.model.encode_image(pixel_values).float(), p=2, dim=1)


class CLIPROIEncoder(nn.Module):
    """Linear readout from normalized CLIP embeddings to one response per ROI."""

    def __init__(self, output_dim, input_dim=CLIP_DIM):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, clip_features):
        return self.linear(clip_features.to(dtype=torch.float32))

    @classmethod
    def load(cls, checkpoint_path, device="cpu"):
        """Load a checkpoint; `model.rois` lists the ROI of each output (empty if not stored)."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["network"]
        model = cls(output_dim=state_dict["linear.weight"].shape[0], input_dim=state_dict["linear.weight"].shape[1])
        model.load_state_dict(state_dict)
        model.rois = list(checkpoint.get("rois", []))
        return model.to(device).eval()


def load_images(paths):
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in paths]


@torch.no_grad()
def predict_images(image_paths, encoder, extractor, batch_size=100):
    """Predicted ROI responses (n_images, n_rois) of image files."""
    outputs = []
    for i in range(0, len(image_paths), batch_size):
        features = extractor(load_images(image_paths[i:i + batch_size]))
        outputs.append(encoder(features.to(next(encoder.parameters()).device)).cpu().numpy())
    return np.concatenate(outputs)
