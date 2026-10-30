"""Q-Former encoding model (the BLIP-based optimization encoder).

Shared structure, with hidden size h:

    Q-Former embedding (16 tokens x 768)
      -> L2-normalize each token -> flatten (12288)
      -> MLP: Linear(12288, 4h) - ReLU - Dropout - Linear(4h, 2h) - ReLU - Dropout - Linear(2h, h) - ReLU - Dropout
      -> readout

Two variants were used in the experiments:

    voxel-wise   h = 512, linear readout       Linear(h, n_voxels)
    region-wise  h = 256, nonlinear readout    Linear(h, 2h) - ReLU - Dropout - Linear(2h, h) - ReLU - Dropout
                                               - Linear(h, n_regions)
                 (voxel-averaged left and right amygdala responses)
"""
import inspect
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = {
    "voxel": {"hidden_dim": 512, "readout": "linear"},
    "region": {"hidden_dim": 256, "readout": "mlp"},
}


class BlipEncoder(nn.Module):
    def __init__(self, output_dim, hidden_dim=512, readout="linear", embed_dim=768, num_tokens=16,
                 dropout=0.1, device="cuda"):
        super().__init__()
        assert readout in ("linear", "mlp"), f"Unknown readout: {readout}"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.readout = readout
        self.dropout = dropout
        # output name -> [start, end) index range in the output, set by training or loading
        self.output_layout = {}

        h = hidden_dim
        self.processor = nn.Sequential(
            nn.Linear(num_tokens * embed_dim, h * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h * 4, h * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h * 2, h),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if readout == "linear":
            self.roi_predictor = nn.Sequential(nn.Linear(h, output_dim))
        else:
            self.roi_predictor = nn.Sequential(
                nn.Linear(h, h * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h * 2, h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h, output_dim),
            )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        self.to(self.device)

    def forward(self, embeddings):
        """
        Args:
            embeddings: Q-Former embeddings, (batch, 16, 768) or (16, 768).
        Returns:
            Predicted responses, (batch, output_dim).
        """
        embeddings = embeddings.to(self.device, dtype=torch.float32)
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        assert embeddings.shape[1:] == (self.num_tokens, self.embed_dim), \
            f"Expected (B, {self.num_tokens}, {self.embed_dim}) embeddings, got {tuple(embeddings.shape)}"
        x = F.normalize(embeddings, p=2, dim=-1)
        features = self.processor(x.flatten(start_dim=1))
        return self.roi_predictor(features)

    def config(self):
        return {
            "embed_dim": self.embed_dim,
            "num_tokens": self.num_tokens,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "readout": self.readout,
            "dropout": self.dropout,
            "normalization": "l2_norm",
            "output_layout": self.output_layout,
        }

    def save(self, checkpoint_path, **extra):
        torch.save({"network": self.state_dict(), **extra}, checkpoint_path)
        with open(checkpoint_path.replace(".chkpt", "_config.json"), "w") as f:
            json.dump(self.config(), f, indent=2)

    @classmethod
    def load(cls, model_dir, device="cuda"):
        """Load `model.chkpt` (or `best_model.chkpt`) and its `*_config.json` from `model_dir`."""
        checkpoint_path = os.path.join(model_dir, "model.chkpt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(model_dir, "best_model.chkpt")
        with open(checkpoint_path.replace(".chkpt", "_config.json")) as f:
            config = json.load(f)
        assert config.get("normalization", "l2_norm") == "l2_norm", f"Unsupported encoder config: {config}"
        if "readout" not in config:
            # older configs: "regression": true -> linear readout, false -> nonlinear readout
            config["readout"] = "linear" if config.get("regression", True) else "mlp"
        valid_keys = set(inspect.signature(cls.__init__).parameters) - {"self", "device"}
        model = cls(**{k: v for k, v in config.items() if k in valid_keys}, device=device)
        state_dict = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
        if isinstance(state_dict, dict) and "network" in state_dict:
            state_dict = state_dict["network"]
        model.load_state_dict(state_dict)
        model.output_layout = config.get("output_layout", {})
        model.eval()
        return model
