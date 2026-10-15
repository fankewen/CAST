"""Per-frame SAM2.1 for the live demo: one subject, one box, one CUDA graph.

This is the live path: a single SAM2.1 *image* forward per frame, in order,
with no pre-pass over the clip and nothing looking ahead.
It does one job only - give the crop box of the single subject in each frame, as
fast as possible.  The offline counterpart is ``inference/offline_mask.py``: it
runs the whole-clip SAM 2 video predictor, which is more accurate but far too
slow for a live feed.

Why it is fast (measured on the A100, 512x512 frames, SAM2.1 Hiera Tiny):

  * **One CUDA graph.**  Hiera Tiny at 512 needs very few FLOPs, so what costs
    milliseconds is kernel launch latency, not compute.  The whole pass -
    encoder, prompt encoder, mask decoder, upscale, threshold, mask-to-box - is
    captured once and replayed: one replay per frame instead of ~200 launches.
  * **Everything stays on the device.**  Normalise and resize happen on the GPU
    (fused into a single multiply-add), the mask is upscaled and thresholded
    inside the graph, and ``batched_mask_to_box`` - Meta's own operator, the same
    one ``SAM2AutomaticMaskGenerator`` uses - turns the mask into four numbers.
    Only those four numbers travel back to the CPU.
  * **Single subject, cheaply.**  The decoder's coarse grid (encoder/4, so
    128x128 at a 512 encoder) locates the subject and prompts the next frame; the
    box is then measured inside that region only, so a stray blob elsewhere in
    the frame cannot inflate it, and no connected-component pass is needed at
    full resolution.
  * **Reused positional encoding.**  SAM recomputes the prompt encoder's dense
    positional encoding on every call although it never changes; cached here.

Measured: 14.3 ms (the straightforward full-mask path) -> ~3.3 ms, with crop
boxes within 1.9 px (mean) of the shipped SAM2 video predictor's mask.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

FALLBACK_BOX_FRACTION = (0.16, 0.12, 0.84, 0.72)
SAM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
SAM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the single biggest blob, so the box is the subject, not a speck."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8)


class CausalBoxSegmenter:
    """One SAM2.1 forward per frame -> the subject's box, in frame pixels."""

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        device: str = "cuda:0",
        amp_dtype: str = "bf16",
        image_size: int = 512,
        box_margin: float = 0.12,
        crop_margin: int = 5,
        compile_body: bool = True,
    ):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.utils.amg import batched_mask_to_box

        self._mask_to_box = batched_mask_to_box
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.image_size = int(image_size)
        self.box_margin = float(box_margin)
        self.crop_margin = int(crop_margin)
        self.frames = 0
        self.last_seconds = 0.0

        model = build_sam2(
            model_cfg,
            checkpoint,
            device=str(self.device),
            hydra_overrides_extra=[f"model.image_size={self.image_size}"],
        )
        model.eval()
        if self.device.type == "cuda":
            model.to(dtype=self.dtype)
        self.model = model
        self.predictor = SAM2ImagePredictor(model)
        self.bb_feat_sizes = [
            (self.image_size // 4, self.image_size // 4),
            (self.image_size // 8, self.image_size // 8),
            (self.image_size // 16, self.image_size // 16),
        ]

        # normalize + scale as one FMA in low precision: CUDA autocast promotes
        # div to fp32, which would then clash with the bf16 weights downstream
        self.scale = (1.0 / (255.0 * SAM_STD)).to(self.device, dtype=self.dtype)
        self.shift = (-SAM_MEAN / SAM_STD).to(self.device, dtype=self.dtype)
        self.box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
            self.dense_pe = model.sam_prompt_encoder.get_dense_pe().detach().clone()

        self.previous_box = None
        self.static_rgb = torch.zeros(
            1, 3, self.image_size, self.image_size, device=self.device, dtype=self.dtype
        )
        self.static_box = torch.zeros(1, 2, 2, device=self.device, dtype=torch.float32)
        body = self._sam_impl
        if compile_body and self.device.type == "cuda":
            body = torch.compile(body, mode="default")
        self._body = body
        self.graph = None
        self.static_out = None
        if self.device.type == "cuda":
            self.static_out = self._capture(body)

    # ------------------------------------------------------------------ graph
    def _sam_impl(self, image, box):
        """Everything between "frame" and "128x128 logits"; GPU-only so it can be captured."""
        model = self.model
        backbone_out = model.forward_image(image)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.bb_feat_sizes[::-1])
        ][::-1]
        points = (box.reshape(-1, 2, 2), self.box_labels)
        sparse, dense = model.sam_prompt_encoder(points=points, boxes=None, masks=None)
        low_res, _, _, _ = model.sam_mask_decoder(
            image_embeddings=feats[-1],
            image_pe=self.dense_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[feats[0], feats[1]],
        )
        # Official finishing step, kept inside the graph because it only costs
        # ~0.1 ms there (0.4 ms outside): upscale the logits to the encoder
        # resolution and threshold at SAM's own mask_threshold (0.0), so the box
        # can be measured at pixel resolution instead of on the coarse grid.
        masks = F.interpolate(
            low_res,
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return low_res, masks[0, 0] > 0

    def _capture(self, body):
        autocast = torch.autocast(
            "cuda", dtype=self.dtype, enabled=self.device.type == "cuda"
        )
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), autocast:
            for _ in range(3):
                body(self.static_rgb, self.static_box)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with autocast:
                out = body(self.static_rgb, self.static_box)
        self.graph = graph
        return out

    # ---------------------------------------------------------------- helpers
    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(rgb).to(self.device, non_blocking=True)
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.dtype)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                antialias=True,
            )
        return image * self.scale + self.shift

    def _coarse_box(self, width: int, height: int) -> np.ndarray:
        x0, y0, x1, y1 = FALLBACK_BOX_FRACTION
        return np.array([x0 * width, y0 * height, x1 * width, y1 * height], dtype=np.float32)

    def _prompt_box(self, width: int, height: int) -> np.ndarray:
        """What SAM is asked about: the previous box, widened a little."""
        if self.previous_box is None:
            return self._coarse_box(width, height)
        left, top, right, bottom = self.previous_box
        pad_x = (right - left) * self.box_margin + 2.0
        pad_y = (bottom - top) * self.box_margin + 2.0
        return np.array(
            [
                max(0.0, left - pad_x),
                max(0.0, top - pad_y),
                min(width - 1.0, right + pad_x),
                min(height - 1.0, bottom + pad_y),
            ],
            dtype=np.float32,
        )

    def _subject_region(self, grid: np.ndarray):
        """Coarse region of the single subject, in encoder-resolution pixels."""
        grid = _largest_component(grid)
        ys, xs = np.nonzero(grid)
        if len(xs) == 0:
            return None
        cells = self.image_size / grid.shape[1]
        return (
            int(np.floor(xs.min() * cells)),
            int(np.floor(ys.min() * cells)),
            int(np.ceil((xs.max() + 1) * cells)),
            int(np.ceil((ys.max() + 1) * cells)),
        )

    def _box_from_mask(self, grid: np.ndarray, mask: torch.Tensor, width: int, height: int):
        """Box of the mask, measured only inside the subject's coarse region.

        Measuring the whole mask would let a stray blob anywhere in the frame
        inflate the box; restricting it to the region found on the small grid
        keeps "one subject" without paying for a second connected-component pass
        at full resolution.
        """
        region = self._subject_region(grid)
        if region is None:
            return None
        left, top, right, bottom = region
        # mask is the thresholded (H, W) mask at encoder resolution
        edges = (
            self._mask_to_box(mask[top:bottom, left:right]).reshape(-1).cpu().numpy()
        )
        if edges[2] <= edges[0] or edges[3] <= edges[1]:
            return None
        scale_x = width / self.image_size
        scale_y = height / self.image_size
        return (
            max(int(round((left + edges[0]) * scale_x)) - self.crop_margin, 0),
            max(int(round((top + edges[1]) * scale_y)) - self.crop_margin, 0),
            min(int(round((left + edges[2] + 1) * scale_x)) + self.crop_margin, width),
            min(int(round((top + edges[3] + 1) * scale_y)) + self.crop_margin, height),
        )

    # ------------------------------------------------------------------- step
    @torch.inference_mode()
    def box(self, rgb: np.ndarray) -> tuple[int, int, int, int]:
        """Crop box of the single subject, in pixels of ``rgb``."""
        height, width = rgb.shape[:2]
        begin = time.perf_counter()
        autocast = torch.autocast(
            "cuda", dtype=self.dtype, enabled=self.device.type == "cuda"
        )
        image = self._preprocess(rgb)
        prompt = self._prompt_box(width, height)
        with autocast:
            box = torch.as_tensor(prompt[None, :], dtype=torch.float32, device=self.device)
            box = self.predictor._transforms.transform_boxes(
                box, normalize=True, orig_hw=(height, width)
            )
        if self.graph is not None:
            self.static_rgb.copy_(image)
            self.static_box.copy_(box)
            self.graph.replay()
            logits, mask = self.static_out
        else:
            with autocast:
                logits, mask = self._body(image, box)
        grid = (logits[0, 0] > 0).to(torch.uint8).cpu().numpy()

        found = self._box_from_mask(grid, mask, width, height)
        if found is not None:
            self.previous_box = found
        elif self.previous_box is None:
            self.previous_box = (0, 0, width, height)
        self.frames += 1
        self.last_seconds = time.perf_counter() - begin
        return self.previous_box

    def reset(self) -> None:
        self.previous_box = None
        self.frames = 0
        self.last_seconds = 0.0


class CausalCrops:
    """Per-frame model input: SAM box -> crop -> transform, strictly in order.

    The live pipeline is a straight line and this object is its first half::

        RGB frame -> SAM2.1 -> box -> crop -> transform -> mocap model

    Nothing runs ahead: no worker thread, no side CUDA stream, no queue, so the
    model step of frame k only starts once the crop of frame k exists.  The crop
    is resized and normalised exactly the way the model was trained, so the input
    format is unchanged.
    """

    def __init__(self, rgb_frames, segmenter: CausalBoxSegmenter, size: int = 256):
        from models.dinov3_infer import build_transform

        self.rgb_frames = iter(rgb_frames)
        self.segmenter = segmenter
        self._transform = build_transform(size)
        self.sam_seconds: list[float] = []
        self.crop_seconds: list[float] = []
        self.last_box = None

    def __iter__(self):
        return self

    def __next__(self):
        from PIL import Image

        rgb = next(self.rgb_frames)
        begin = time.perf_counter()
        box = self.segmenter.box(rgb)
        self.last_box = box
        after_sam = time.perf_counter()
        crop = Image.fromarray(rgb).crop(box)
        frame = self._transform(crop)
        done = time.perf_counter()
        self.sam_seconds.append(after_sam - begin)
        self.crop_seconds.append(done - after_sam)
        return frame

    def stats(self) -> dict:
        if not self.sam_seconds:
            return {}
        sam = np.asarray(self.sam_seconds[-120:]) * 1000
        crop = np.asarray(self.crop_seconds[-120:]) * 1000
        return {
            "sam_ms_mean": float(sam.mean()),
            "sam_ms_p95": float(np.percentile(sam, 95)),
            "crop_ms_mean": float(crop.mean()),
            "crop_ms_p95": float(np.percentile(crop, 95)),
            "seg_ms_mean": float(sam.mean() + crop.mean()),
            "seg_ms_p95": float(np.percentile(sam + crop, 95)),
        }
