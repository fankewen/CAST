"""Load the official DINOv2 ViT-L/14 from explicit local source and weights."""

import sys
from pathlib import Path

import torch


def load_dinov2_vitl14(repo, weights, device="cpu"):
    repo = Path(repo).expanduser().resolve()
    weights = Path(weights).expanduser().resolve()
    if not (repo / "dinov2" / "hub" / "backbones.py").is_file():
        raise FileNotFoundError(f"DINOv2 source repository not found: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv2 pretrained weights not found: {weights}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov2.hub.backbones import dinov2_vitl14

    # Never download or silently initialize random weights in a training worker.
    model = dinov2_vitl14(pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)
