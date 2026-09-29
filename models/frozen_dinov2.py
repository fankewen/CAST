"""Frozen online DINOv2 image encoder for video-to-motion training."""

import os
from pathlib import Path

import torch
import torch.nn as nn

from utils.config_validation import require_bool, require_positive_int
from models.dinov2_infer import load_dinov2_vitl14


class FrozenDinoV2Encoder(nn.Module):
    """Run DINOv2 inside the model while keeping all backbone weights frozen.

    The dataset must provide input_size (224 by default) frames directly.
    This encoder never resizes images; input_size only validates their size.
    Input images are already mask-bbox cropped, resized, and ImageNet-normalized
    by the dataset. Frames are encoded in micro-batches to bound activation
    memory. ``torch.no_grad`` is deliberate: ``inference_mode`` tensors cannot
    always be saved by downstream trainable Linear layers for backward.
    """

    def __init__(
        self,
        repo: str,
        weights: str,
        micro_batch_size: int = 8,
        include_cls_token: bool = True,
        compile_backbone: bool = False,
        compile_cache_dir: str | None = None,
        input_size: int = 224,
    ):
        super().__init__()
        self.micro_batch_size = require_positive_int(
            micro_batch_size, "DINO micro_batch_size"
        )
        self.input_size = require_positive_int(input_size, "DINOv2 input_size")
        if self.input_size % 14:
            raise ValueError("DINOv2 input_size must be divisible by 14")
        self.include_cls_token = bool(include_cls_token)
        self.compile_backbone = require_bool(compile_backbone, "DINO compile_backbone")
        self._backbone_compiled = False
        if self.compile_backbone:
            # Keep compiler artifacts away from the training data/checkpoints.
            if compile_cache_dir is not None:
                cache = Path(compile_cache_dir).expanduser()
                cache.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
                os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
            os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")
        self.backbone = load_dinov2_vitl14(repo, weights, device="cpu")
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Parent model.train() must never enable stochastic training behavior
        # in the frozen visual backbone.
        self.backbone.eval()
        return self

    def forward(self, image_rgb: torch.Tensor, frame_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if image_rgb.ndim != 5 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"image_rgb must have shape [B,T,3,H,W], got {tuple(image_rgb.shape)}"
            )
        if image_rgb.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"DINOv2 expects {self.input_size}x{self.input_size} input frames, "
                f"got {tuple(image_rgb.shape[-2:])}; set data.image_size "
                f"to {self.input_size}. The encoder does not resize images."
            )
        if self.compile_backbone and image_rgb.is_cuda and not self._backbone_compiled:
            # Compile the callable, preserving module/state_dict names. CPU stays eager.
            self.backbone.forward_features = torch.compile(
                self.backbone.forward_features, fullgraph=True, dynamic=None,
                options={"emulate_precision_casts": True, "force_same_precision": True,
                         "mixed_mm_choice": "aten", "compile_threads": 2},
            )
            self._backbone_compiled = True
        batch, frames = image_rgb.shape[:2]
        flat = image_rgb.flatten(0, 1)
        valid_indices = None
        if frame_valid_mask is not None:
            if frame_valid_mask.shape != (batch, frames):
                raise ValueError("frame_valid_mask must have shape [B,T]")
            valid_indices = frame_valid_mask.to(device=flat.device, dtype=torch.bool).flatten().nonzero().flatten()
            flat = flat.index_select(0, valid_indices)
        if len(flat) == 0:
            patch_size = self.backbone.patch_size
            if isinstance(patch_size, (tuple, list)):
                ph, pw = patch_size
            else:
                ph = pw = patch_size
            token_count = (self.input_size // ph) * (self.input_size // pw) + int(self.include_cls_token)
            return image_rgb.new_zeros(batch, frames, token_count, self.backbone.embed_dim)
        encoded = []
        with torch.no_grad():
            for start in range(0, len(flat), self.micro_batch_size):
                micro = flat[start:start + self.micro_batch_size]
                if self._backbone_compiled and len(micro) > 1:
                    # Only frame count varies; keep spatial/token dimensions static.
                    torch._dynamo.mark_dynamic(micro, 0)
                features = self.backbone.forward_features(micro)
                patch = features["x_norm_patchtokens"]
                if self.include_cls_token:
                    cls = features["x_norm_clstoken"].unsqueeze(1)
                    patch = torch.cat((cls, patch), dim=1)
                encoded.append(patch)
        tokens = torch.cat(encoded, dim=0)
        if valid_indices is not None:
            full = tokens.new_zeros(batch * frames, tokens.shape[1], tokens.shape[2])
            tokens = full.index_copy(0, valid_indices, tokens)
        return tokens.reshape(batch, frames, tokens.shape[1], tokens.shape[2])
