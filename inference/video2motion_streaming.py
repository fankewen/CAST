"""Explicit causal streaming inference and KV-cache correctness check.

The check performs an 80-frame prefill followed by one incremental frame,
then compares that output with the last frame of a regular causal forward.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.cache import _backend_cfg, _source_configs, _normalize_splits
from data.cache_reader import build_cache_dataset
from utils.batching import collate_single_anyspecies_padded
from utils.model_inputs import model_forward_inputs
from utils.config_utils import instantiate_from_config, load_yaml_config
from utils.config_validation import require_positive_int


TIME_KEYS = {"image_embed", "image_rgb", "frame_valid_mask", "position", "fk_position_target", "loss_frame_mask", "oracle_pose_position"}


def _amp_context(mode: str, device: torch.device):
    mode = str(mode).lower()
    if mode in {"off", "none", "fp32"} or device.type != "cuda":
        return contextlib.nullcontext()
    if mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mode == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unsupported amp dtype: {mode}")


def _run_with_amp(fn, mode: str, device: torch.device):
    with _amp_context(mode, device):
        return fn()


def _first_validation_dataset(data_cfg, name, window):
    for source_name, source_cfg in _source_configs(data_cfg).items():
        backend = _backend_cfg(source_cfg)
        raw = source_cfg.get("validation", {source_name: {}})
        if name not in raw:
            continue
        cfg = dict(raw[name] or {})
        splits = _normalize_splits(cfg.pop("source_splits", None), f"validation.{name}.source_splits")
        return build_cache_dataset({**backend, **cfg, "eval_stride": window}, "test", window, first_window_eval=False, complete_motion_eval=False, source_splits=splits)
    raise KeyError(f"validation source {name!r} not found")


def _frame_batch(batch, index):
    out = dict(batch)
    for key in TIME_KEYS:
        value = out.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] > index:
            out[key] = value[:, index:index + 1]
    return out


@torch.inference_mode()
def run(args):
    cfg = load_yaml_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = instantiate_from_config(cfg["model"]).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(checkpoint["model_state"])
    if args.weights_dtype == "bf16" and device.type == "cuda":
        model.to(dtype=torch.bfloat16)
        # Keep analytic geometry and the fused correction kernels in FP32.
        if model.grace is not None:
            model.grace.float()
    max_joints = require_positive_int(cfg["data"].get("max_joints"), "data.max_joints")
    dataset = _first_validation_dataset(cfg["data"], args.validation_source, args.window)
    sample = dataset[args.sample_index]
    batch = collate_single_anyspecies_padded(sample, max_joints=max_joints, dynamic_max_joints=bool(cfg["data"].get("dynamic_batch_max_joints", False)))
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    model_batch = model_forward_inputs(batch, cfg)
    if args.amp_dtype == "bf16" and isinstance(model_batch.get("image_embed"), torch.Tensor):
        model_batch["image_embed"] = model_batch["image_embed"].to(torch.bfloat16)
    elif args.amp_dtype == "fp16" and isinstance(model_batch.get("image_embed"), torch.Tensor):
        model_batch["image_embed"] = model_batch["image_embed"].to(torch.float16)
    total_frames = int(model_batch["image_embed"].shape[1]) if "image_embed" in model_batch else int(model_batch["image_rgb"].shape[1])
    if total_frames < args.prefill + 1:
        raise ValueError(f"sample has {total_frames} frames, need at least {args.prefill + 1}")

    streaming_step = model.forward_streaming
    if args.compile:
        # Compile only the steady-state/full-cache branch.  Compiling frames
        # 1..N specializes on every Python cache length and defeats Dynamo's
        # recompile limit.  A disposable state pays compilation once; the
        # real state below then retains exact stream semantics.
        warm_state = _run_with_amp(
            lambda: model.init_streaming_state(model_batch, max_length=args.max_cache_length),
            args.amp_dtype, device,
        )
        for frame_index in range(args.prefill):
            _, warm_state = _run_with_amp(
                lambda i=frame_index: model.forward_streaming(
                    _frame_batch(model_batch, i), warm_state
                ),
                args.amp_dtype, device,
            )
        streaming_step = torch.compile(
            model.forward_streaming,
            fullgraph=False,
            dynamic=False,
            mode=args.compile_mode,
        )
        warm_frame = _frame_batch(model_batch, args.prefill)
        for _ in range(args.compile_warmup):
            _, warm_state = _run_with_amp(
                lambda: streaming_step(warm_frame, warm_state),
                args.amp_dtype, device,
            )

    # Reference tensors remain static; init consumes frame 0 as the reference.
    state = _run_with_amp(
        lambda: model.init_streaming_state(model_batch, max_length=args.max_cache_length),
        args.amp_dtype, device,
    )
    stream_outputs = []
    for frame_index in range(args.prefill):
        frame = _frame_batch(model_batch, frame_index)
        out, state = _run_with_amp(
            lambda: model.forward_streaming(
                frame, state
            ),
            args.amp_dtype, device,
        )
        stream_outputs.append(out["pred_rot6d"])
    out, state = _run_with_amp(
        lambda: streaming_step(
            _frame_batch(model_batch, args.prefill),
            state,
        ),
        args.amp_dtype, device,
    )
    stream_outputs.append(out["pred_rot6d"])

    # Compare only the incremental frame with the regular causal model.
    regular = _run_with_amp(
        lambda: model(
            model_batch,
        )["pred_rot6d"][:, args.prefill:args.prefill + 1],
        args.amp_dtype, device,
    )
    diff = (stream_outputs[-1] - regular).abs()
    print({
        "frames_prefilled": args.prefill,
        "max_abs_diff_last_frame": float(diff.max().item()),
        "mean_abs_diff_last_frame": float(diff.mean().item()),
        "cache_lengths": [cache.length for cache in state["encoder"]["temporal"]["caches"]],
    }, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-source", default="zoo")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--window", type=int, default=81)
    parser.add_argument("--prefill", type=int, default=80)
    parser.add_argument(
        "--max-cache-length", type=int, default=81,
        help="Maximum causal KV-cache length; defaults to the 81-frame training window.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True,
        help="Compile the steady-state streaming callable (enabled by default).",
    )
    parser.add_argument(
        "--compile-mode", default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument("--compile-warmup", type=int, default=3)
    parser.add_argument(
        "--amp-dtype", choices=["off", "bf16", "fp16"], default="bf16",
        help="AMP dtype for DINO/neural paths; analytic correction remains FP32.",
    )
    parser.add_argument(
        "--weights-dtype", choices=["fp32", "bf16"], default="bf16",
        help="Neural-network parameter dtype; RotationCorrection remains FP32.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
