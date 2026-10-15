"""Single-process deployment for one fixed-skeleton causal CAST stream.

All skeleton-only work is performed while the engine is constructed. The
first image binds the reference visual feature once; every later ``step`` call
accepts only one new normalized RGB frame.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Iterable


# cpp_extension resolves CUDA_HOME while torch is imported. Some machines keep
# nvcc outside /usr/local, so discover it before importing any model modules.
if not os.environ.get("CUDA_HOME"):
    for _cuda_root in (
        os.environ.get("CAST_CUDA_HOME"),
        "/usr/local/cuda",
    ):
        if _cuda_root and (Path(_cuda_root) / "bin" / "nvcc").is_file():
            os.environ["CUDA_HOME"] = _cuda_root
            break

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inference.video2motion_unlabeled import (
    iter_center_crops,
    iter_video_rgb,
    static_conditioning,
    video_metadata,
)
from models.attention_layout import set_attention_layout_cache
from utils.model_inputs import normalize_model_skeleton_inputs
from utils.config_utils import instantiate_from_config, load_yaml_config
from utils.config_validation import require_positive_int


def _amp_context(device: torch.device, dtype: str):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
    )


def _as_frame_sequence(frames: torch.Tensor | Iterable[torch.Tensor]):
    if not isinstance(frames, torch.Tensor):
        frames = torch.stack(list(frames), dim=0)
    if frames.ndim == 3:
        frames = frames[None, None]
    elif frames.ndim == 4:
        frames = frames[None]
    if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[2] != 3:
        raise ValueError(
            "frames must be [3,H,W], [T,3,H,W], or [1,T,3,H,W]; "
            f"got {tuple(frames.shape)}"
        )
    return frames


def _all_temporal_caches(state):
    yield from state["encoder"]["temporal"]["caches"]
    yield from state["decoder"]["decoder"]["caches"]


def reset_streaming_caches(state) -> None:
    """Empty every temporal window in place, keeping the compiled graph valid."""
    for cache in _all_temporal_caches(state):
        cache.reset_streaming()


def _snapshot_caches(state):
    result = []
    for cache in _all_temporal_caches(state):
        if cache.key is None:
            result.append(None)
            continue
        result.append(
            {
                "key": cache.key.clone(),
                "value": cache.value.clone(),
                "start_pos": cache.start_pos.clone(),
                "write_pos": cache._write_pos.clone(),
                "total": cache._total.clone(),
                "filled": cache._filled.clone(),
            }
        )
    return result


def _restore_caches(state, snapshots):
    caches = list(_all_temporal_caches(state))
    if len(caches) != len(snapshots):
        raise RuntimeError("streaming cache structure changed during warmup")
    for cache, saved in zip(caches, snapshots):
        if saved is None:
            continue
        cache.key.copy_(saved["key"])
        cache.value.copy_(saved["value"])
        cache.start_pos.copy_(saved["start_pos"])
        cache._write_pos.copy_(saved["write_pos"])
        cache._total.copy_(saved["total"])
        cache._filled.copy_(saved["filled"])
        cache._relative_delta = None


class _CompiledStreamingStep:
    """Stable-address compiled callable copied from the optimized runner."""

    _DYNAMIC_KEYS = ("image_embed", "frame_valid_mask")

    def __init__(self, model, state, frame, *, amp_dtype, use_cudagraphs=False):
        self.model = model
        self.state = state
        self.amp_dtype = amp_dtype
        self.frame = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in frame.items()
        }

        def bound_step():
            with _amp_context(self.frame["image_embed"].device, self.amp_dtype):
                return self.model.forward_streaming(self.frame, self.state)

        self._compiled = torch.compile(
            bound_step,
            fullgraph=False,
            dynamic=False,
            options={"triton.cudagraphs": bool(use_cudagraphs)},
        )

    @torch.no_grad()
    def step(self, frame):
        for key in self._DYNAMIC_KEYS:
            self.frame[key].copy_(frame[key])
        return self._compiled()


class RealtimeVideo2Motion:
    """The streaming deployment: one checkpoint, one skeleton, one causal stream.

    The pipeline is split into the three steps the caller actually has inputs
    for -- ``__init__`` loads the weights, ``prepare`` binds the fixed skeleton
    and ``compile_streaming`` compiles the step.  The CLI runs all three back to
    back; the live engine runs them from two panel buttons (weights first, rig
    when the armature is picked) and binds the reference frame when a clip
    starts.  ``step`` then consumes one RGB frame at a time.
    """

    def __init__(
        self,
        *,
        config: str,
        checkpoint: str,
        device: str = "cuda:0",
        cache_length: int = 81,
        amp_dtype: str = "bf16",
        compile_warmup: int = 3,
        compile_cache_dir: str = "/path/to/cache/cast_realtime_compile",
        use_cudagraphs: bool = False,
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("optimized deployment requires CUDA")
        if amp_dtype not in {"bf16", "fp16"}:
            raise ValueError("amp_dtype must be bf16 or fp16")
        self.amp_dtype = amp_dtype
        self.neural_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.cache_length = require_positive_int(cache_length, "cache_length")
        self.compile_warmup = require_positive_int(compile_warmup, "compile_warmup")
        self.use_cudagraphs = bool(use_cudagraphs)
        self.cfg = load_yaml_config(config)
        self.image_size = require_positive_int(
            self.cfg["data"].get("image_size"), "data.image_size"
        )
        self.model_joint_limit = require_positive_int(
            self.cfg["data"].get("max_joints"), "data.max_joints"
        )

        cache_root = Path(compile_cache_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "inductor"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
        os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        self.model = instantiate_from_config(self.cfg["model"])
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_trainable_state_dict(checkpoint_data["model_state"])
        self.model.to(self.device).eval().to(dtype=self.neural_dtype)
        if self.model.grace is not None:
            self.model.grace.float()
        self.model.image_backbone.compile_backbone = True
        self.model.image_backbone._backbone_compiled = False
        self.model.prepare_inference_optimizations()

        self.state = None
        self.joint_count: int | None = None
        self.runner: _CompiledStreamingStep | None = None

    def prepare(self, static: str) -> None:
        """Bind the fixed skeleton: topology and the caches."""
        fixed_joint_count = require_positive_int(
            np.load(static, allow_pickle=True).item().get("joints"), "static.joints"
        )
        if fixed_joint_count > self.model_joint_limit:
            raise ValueError(
                f"skeleton has {fixed_joint_count} joints; limit is {self.model_joint_limit}"
            )
        dummy_rgb = torch.zeros(1, 3, self.image_size, self.image_size)
        static_batch, self.joint_count = static_conditioning(
            static,
            dummy_rgb,
            self.device,
            max_joints=fixed_joint_count,
        )
        static_batch = normalize_model_skeleton_inputs(static_batch)
        static_batch.pop("image_rgb", None)
        static_batch.pop("frame_valid_mask", None)
        # This is the only skeleton initialization call. It happens before any
        # image is accepted and owns all static embeddings/topology metadata.
        with _amp_context(self.device, self.amp_dtype):
            self.state = self.model.prepare_streaming_static_state(
                static_batch, max_length=self.cache_length
            )

    def _compile(self, stable_frame, *, keep: bool = False) -> None:
        """Compile the streaming step and warm it up with one in-hand frame."""
        self.runner = _CompiledStreamingStep(
            self.model,
            self.state,
            stable_frame,
            amp_dtype=self.amp_dtype,
            use_cudagraphs=self.use_cudagraphs,
        )
        saved = _snapshot_caches(self.state) if keep else None
        for _ in range(self.compile_warmup):
            self.runner.step(stable_frame)
        torch.cuda.synchronize(self.device)
        if keep:
            _restore_caches(self.state, saved)
        else:
            reset_streaming_caches(self.state)

    @torch.inference_mode()
    def compile_streaming(self, reference_rgb=None) -> None:
        """Compile before any real frame exists, then hand over to ``step``.

        A placeholder reference is bound so the whole graph exists, the layout
        cache is frozen, and the warm-up frames are rewound afterwards, so the
        stream starts empty and the first real frame is just frame 0.
        """
        if self.state is None:
            raise RuntimeError("prepare(static) before compile_streaming")
        if reference_rgb is None:
            reference_rgb = torch.zeros(1, 3, self.image_size, self.image_size)
        embedding = self.encode_rgb(reference_rgb)
        with _amp_context(self.device, self.amp_dtype):
            self.model.bind_streaming_reference(self.state)
            # One eager frame materialises every layer cache; afterwards
            # ``cache.key is not None`` for the whole compiled lifetime.
            self.model.forward_streaming(self._embedding_frame(embedding), self.state)
        set_attention_layout_cache(self.state["attention_layout_cache"], readonly=True)
        self._compile(self._embedding_frame(embedding))

    @torch.inference_mode()
    def bind_reference(self, frame) -> None:
        """Point the compiled graph at a new clip's first frame, in place.

        The graph holds the reference tensors it was compiled with, so a new
        clip copies into them instead of replacing them.
        """
        state = self.state
        current = state["encoder"].get("ref_query")
        with _amp_context(self.device, self.amp_dtype):
            self.model.bind_streaming_reference(state)
        if current is None:
            return
        state["encoder"]["ref_query"] = current.copy_(state["encoder"]["ref_query"])

    def reset_streaming(self) -> None:
        """Drop the temporal window but keep the compiled graph.

        The cache tensors were created while the step was compiled, so they are
        inference tensors and may only be written from an inference-mode context.
        """
        with torch.inference_mode():
            reset_streaming_caches(self.state)

    @torch.inference_mode()
    def encode_rgb(self, frames):
        frames = _as_frame_sequence(frames).to(
            self.device, dtype=self.neural_dtype, non_blocking=True
        )
        with _amp_context(self.device, self.amp_dtype):
            return self.model.image_backbone(frames)

    @staticmethod
    def _embedding_frame(embedding, index=0):
        return {
            "image_embed": embedding[:, index : index + 1],
            "frame_valid_mask": torch.ones(
                embedding.shape[0], 1, dtype=torch.bool, device=embedding.device
            ),
        }

    @torch.inference_mode()
    def initialize(self, history):
        """Bind the reference image, prefill KV, and compile the stable step."""
        history = _as_frame_sequence(history)
        if history.shape[1] < 1:
            raise ValueError("initialize needs at least one history frame")
        history = history[:, -self.cache_length :]
        # DINO is compiled for the exact single-frame deployment shape.
        embeddings = torch.cat(
            [self.encode_rgb(history[:, index : index + 1])
             for index in range(history.shape[1])],
            dim=1,
        )
        with _amp_context(self.device, self.amp_dtype):
            self.model.bind_streaming_reference(self.state)

        rotations = []
        for index in range(embeddings.shape[1]):
            with _amp_context(self.device, self.amp_dtype):
                output, self.state = self.model.forward_streaming(
                    self._embedding_frame(embeddings, index), self.state
                )
            rotations.append(output["pred_rot6d"])

        set_attention_layout_cache(self.state["attention_layout_cache"], readonly=True)
        # The prefill is history the caller keeps, so the warm-up frames are
        # rewound afterwards instead of thrown away.
        self._compile(
            self._embedding_frame(embeddings, embeddings.shape[1] - 1), keep=True
        )
        return torch.cat(rotations, dim=1)

    @torch.inference_mode()
    def step_embedding(self, image_embedding):
        if self.runner is None:
            raise RuntimeError("call initialize(history) before step")
        if image_embedding.ndim == 3:
            image_embedding = image_embedding[:, None]
        frame = self._embedding_frame(
            image_embedding.to(self.device, dtype=self.neural_dtype, non_blocking=True)
        )
        output, self.state = self.runner.step(frame)
        return output["pred_rot6d"]

    @torch.inference_mode()
    def step(self, frame):
        """Consume one new normalized RGB image and return [1,1,J,6]."""
        return self.step_embedding(self.encode_rgb(frame))


@torch.inference_mode()
def run_video(args):
    engine = RealtimeVideo2Motion(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        cache_length=args.cache_length,
        amp_dtype=args.amp_dtype,
        compile_warmup=args.compile_warmup,
        compile_cache_dir=args.compile_cache_dir,
    )
    engine.prepare(args.static)
    metadata = video_metadata(args.input_video)
    frames = iter_center_crops(
        iter_video_rgb(args.input_video, int(metadata["source_frame_count"])),
        size=engine.image_size,
        crop_scale=args.center_crop_scale,
    )
    history = []
    for _ in range(args.prefill):
        try:
            history.append(next(frames))
        except StopIteration as exc:
            raise ValueError(f"video needs at least {args.prefill} frames") from exc
    rotations = [
        value.squeeze(0).cpu()
        for value in engine.initialize(torch.stack(history)).split(1, dim=1)
    ]
    print(
        f"Realtime engine ready: history={len(history)}, "
        f"cache={args.cache_length}, joints={engine.joint_count}",
        flush=True,
    )
    for frame in frames:
        rotations.append(engine.step(frame).squeeze(0).cpu())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, torch.cat(rotations).float().numpy())
    print(f"Saved causal rotations to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fixed-skeleton realtime CAST")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--static", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-length", type=int, default=81)
    parser.add_argument("--prefill", type=int, default=80)
    parser.add_argument("--center-crop-scale", type=float, default=0.62)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--compile-warmup", type=int, default=3)
    parser.add_argument(
        "--compile-cache-dir", default="/path/to/cache/cast_realtime_compile"
    )
    run_video(parser.parse_args())


if __name__ == "__main__":
    main()
