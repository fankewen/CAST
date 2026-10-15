"""Unified cache-based video-to-motion training entry point.

Every compatible cache path is configured under ``data.sources`` and uses the
same source builder, mixed sampler, model, losses, and optimized attention code.
"""

import argparse
import copy
import contextlib
import functools
import glob
import json
import math
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from utils.batch_prefetch import DeviceBatchPrefetcher
from utils.fk_depth import build_fk_layout, depth_fk, local_rotation_fk
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.cache import (
    CacheStepIterableDataset,
    MixedCacheStepIterableDataset,
    build_cache_source_bundle,
    required_sample_capabilities,
)
from utils.config_validation import (
    require_bool,
    require_int,
    require_positive_int,
)
from utils.config_utils import dump_yaml_config, instantiate_from_config, load_yaml_config
from utils.dist_utils import cleanup_distributed, is_main_process, reduce_sum_scalar, setup_distributed, worker_init_fn
from utils.rotation import (
    rot6d_to_rotmat_tensor,
    rotmat_to_rot6d_tensor,
    rotation_matrices_from_fused_sum,
)
from models.base_model.bone_direction_head import (
    canonical_fallback_axis,
    positions_to_bone_directions,
)
from models.graph_attention_layout import (
    DEFAULT_AUTO_GROUP_MAX_PAIR_RATIO,
    set_auto_group_max_pair_ratio,
)
from utils.transforms3d import (
    matrix_to_quaternion,
    quaternion_invert,
    quaternion_raw_multiply,
    standardize_quaternion,
)
from utils.motion_io import motion_output_dir
from visualization.motion_visualization import video_output
from utils.skeleton_observability import (
    DEFAULT_EDGE_EPSILON,
    resolve_edge_epsilon,
)
from visualization.render_skeleton_rgb_pose_compare import render_frame_sources

from utils.batching import (
    collate_single_anyspecies_padded,
    iter_worker_parallel_batches,
)
from utils.model_inputs import (
    ORACLE_UNUSED_VISUAL_KEYS,
    model_forward_inputs,
    uses_gt_pose_oracle,
)
from utils.motion_metrics import topocap_protocol_metrics


PROCESS_NAME = "video2motion"
DDP_STATIC_GRAPH = True
TOPOCAP_METRIC_KEYS = (
    "topocap_mpjpe_mm",
    "topocap_pa_mpjpe_mm",
    "topocap_mpjve_mm",
    "topocap_cd",
)
CHECKPOINT_SELECTION_PROTOCOL = "topocap_v1"


def _move_large_cpu_tensor_via_pinned_chunks(
    value, device, *, chunk_bytes
):
    """Enqueue a large CPU tensor through bounded pinned staging chunks.

    CUDA 12.8 on the A100-2 host rejects a single pinned allocation around
    768 MiB, while a per-rank cache image batch is about 1.9 GiB. Flattening
    the contiguous source makes the bound independent of tensor shape. The
    pinned chunks stay alive until every copy has been enqueued; PyTorch's
    caching host allocator then delays reuse until the CUDA copies complete.
    """
    chunk_bytes = require_positive_int(chunk_bytes, "pin-memory chunk bytes")
    if value.numel() == 0:
        return value.to(device, non_blocking=True)
    source = value.contiguous()
    source_flat = source.view(-1)
    output = torch.empty(source.shape, dtype=source.dtype, device=device)
    output_flat = output.view(-1)
    elements_per_chunk = max(chunk_bytes // source.element_size(), 1)
    pinned_chunks = []
    for start in range(0, source_flat.numel(), elements_per_chunk):
        end = min(start + elements_per_chunk, source_flat.numel())
        pinned_chunk = source_flat[start:end].pin_memory()
        output_flat[start:end].copy_(pinned_chunk, non_blocking=True)
        pinned_chunks.append(pinned_chunk)
    return output


def move_batch_to_device(
    batch,
    device,
    excluded_keys=frozenset(),
    *,
    pin_memory=False,
    pin_memory_chunk_bytes=256 * 1024 * 1024,
):
    """Move nested batches while preserving metadata and bounded pinning."""
    result = {}
    if "parent_a" in batch and batch["parent_a"].device.type == "cpu":
        result["_fk_layout"] = build_fk_layout(batch["parent_a"], device=device)
    for key, value in batch.items():
        if key in excluded_keys:
            result[key] = value
        elif isinstance(value, dict):
            result[key] = move_batch_to_device(
                value,
                device,
                excluded_keys=excluded_keys,
                pin_memory=pin_memory,
                pin_memory_chunk_bytes=pin_memory_chunk_bytes,
            )
        elif isinstance(value, torch.Tensor):
            value_bytes = value.numel() * value.element_size()
            if (
                pin_memory
                and device.type == "cuda"
                and value.device.type == "cpu"
                and not value.is_pinned()
                and value_bytes > pin_memory_chunk_bytes
            ):
                result[key] = _move_large_cpu_tensor_via_pinned_chunks(
                    value,
                    device,
                    chunk_bytes=pin_memory_chunk_bytes,
                )
            else:
                result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def validate_skeleton_normalize_config(cfg):
    """Validate skeleton-scale-normalized training loss configuration."""
    train_cfg = cfg.get("train", {})
    enabled = require_bool(
        train_cfg.get("skeleton_normalize", False),
        "train.skeleton_normalize",
    )
    if not enabled:
        return
    if not cfg.get("data", {}).get("sources"):
        raise ValueError(
            "train.skeleton_normalize requires data.sources cache datasets"
        )
    beta = float(train_cfg.get("skeleton_normalize_smooth_l1_beta", 1.0))
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(
            "train.skeleton_normalize_smooth_l1_beta must be finite and > 0"
        )


def _dataset_motion_groups(dataset, base_index=0):
    """Return complete motion-window index groups, including ConcatDataset."""
    if hasattr(dataset, "datasets"):
        groups = []
        offset = base_index
        for child in dataset.datasets:
            groups.extend(_dataset_motion_groups(child, offset))
            offset += len(child)
        return groups
    if hasattr(dataset, "items"):
        by_motion = {}
        for index, item in enumerate(dataset.items):
            key = (getattr(dataset, "skeleton_id", getattr(dataset, "rig_id", "")), item["seq"])
            by_motion.setdefault(key, []).append(base_index + index)
        return list(by_motion.values())
    return [[base_index + index] for index in range(len(dataset))]


def _dataset_motion_rels(dataset):
    """Return one stable label per complete motion without decoding RGB."""
    if hasattr(dataset, "datasets"):
        labels = []
        for child in dataset.datasets:
            labels.extend(_dataset_motion_rels(child))
        return labels
    if hasattr(dataset, "items"):
        labels = []
        seen = set()
        skeleton_id = getattr(dataset, "skeleton_id", getattr(dataset, "rig_id", ""))
        for item in dataset.items:
            rel = f"{skeleton_id}/{item['seq']}"
            if rel not in seen:
                seen.add(rel)
                labels.append(rel)
        return labels
    raise TypeError(
        "Validation visualization requires a dataset with motion metadata"
    )


def select_validation_visualization_motion(motion_rels, validation_index, seed):
    """Select one motion per validation, without repeats within a cycle."""
    if not motion_rels:
        raise ValueError("Validation visualization requires at least one motion")
    validation_index = require_int(
        validation_index, "validation visualization index"
    )
    if validation_index < 0:
        raise ValueError("validation visualization index must be >= 0")
    seed = require_int(seed, "eval.visualization.sample_seed")
    cycle, offset = divmod(validation_index, len(motion_rels))
    permutation = np.random.default_rng(seed + cycle).permutation(
        len(motion_rels)
    )
    return motion_rels[int(permutation[offset])]


class DistributedMotionEvalSampler(Sampler):
    """Partition whole validation motions without padding duplicates.

    PyTorch's ``DistributedSampler`` pads a non-divisible validation set by
    repeating samples. It can also put overlapping windows from one motion on
    different ranks, preventing local overlap fusion. Each rank therefore owns
    a disjoint set of complete motions. Evaluation has no per-batch collectives,
    so ranks may safely process different window counts.
    """

    def __init__(self, dataset, rank, world_size, max_samples=None):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        groups = _dataset_motion_groups(dataset)
        if max_samples is not None:
            max_samples = require_positive_int(
                max_samples, "eval.max_samples"
            )
            if max_samples < len(groups):
                # Cover the complete ordered validation universe rather than
                # taking only the first rig from a concatenated multi-rig set.
                selected = np.linspace(
                    0, len(groups) - 1, num=max_samples, dtype=np.int64
                )
                groups = [groups[int(index)] for index in selected]
        self.indices = [
            index
            for group in groups[self.rank::self.world_size]
            for index in group
        ]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def configure_reproducible_runtime(runtime_cfg):
    """Configure deterministic CUDA behavior for the accelerated path.

    This keeps the packed/depth VisualGate implementation while fixing CUDA
    kernel selection and reduction choices across repeated runs.
    """

    runtime_cfg = runtime_cfg or {}
    enabled = runtime_cfg.get("deterministic", False)
    if not isinstance(enabled, bool):
        raise TypeError("runtime.deterministic must be a boolean")
    if not enabled:
        return

    # This must be configured before the first CUDA matrix multiplication.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def partition_base_and_residual_parameters(model):
    """Return all trainable parameters as the single optimization group."""
    base_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return base_parameters, []


def apply_frame_masking(batch, cfg):
    """Sample mixed BERT-style and span masks for frame completion."""
    masking = cfg.get("train", {}).get("frame_masking", {})
    if not masking or not bool(masking.get("enabled", False)):
        return
    valid = batch.get("frame_valid_mask")
    if valid is None:
        raise KeyError("frame masking requires batch['frame_valid_mask']")
    valid = valid.bool()
    bsz, frames = valid.shape
    ratio = float(masking.get("mask_ratio", 0.25))
    min_frames = int(masking.get("min_frames", 1))
    max_frames = int(masking.get("max_frames", 0))
    if not 0.0 < ratio < 1.0:
        raise ValueError("train.frame_masking.mask_ratio must be in (0,1)")
    lengths = valid.sum(dim=1, dtype=torch.long)
    # Three modes: BERT random frames, one contiguous span, or several spans.
    # Probabilities are configurable and sampled per sequence without CPU sync.
    prob_values = [
        float(masking.get("bert_prob", 0.35)),
        float(masking.get("single_span_prob", 0.35)),
        float(masking.get("multi_span_prob", 0.30)),
    ]
    if any(value < 0 for value in prob_values) or sum(prob_values) <= 0:
        raise ValueError("frame mask mode probabilities must be non-negative")
    probs = torch.tensor(prob_values, device=valid.device, dtype=torch.float32)
    probs = probs / probs.sum()
    modes = torch.multinomial(probs.expand(bsz, -1), 1).squeeze(1)
    frame_index = torch.arange(frames, device=valid.device).unsqueeze(0)
    interior = valid.clone()
    interior[:, 0] = False
    last = (lengths - 1).clamp_min(0)
    interior.scatter_(1, last[:, None], False)

    # BERT-style random masking: sample independent scores and threshold at a
    # per-sequence ratio. Endpoints remain visible to anchor the completion.
    bernoulli_ratio = float(masking.get("bert_mask_ratio", ratio))
    bert = torch.rand(bsz, frames, device=valid.device).lt(bernoulli_ratio)
    bert &= interior

    # Single span: random length and start, leaving one observed frame at each
    # side whenever the sequence has at least three valid frames.
    widths = torch.round(lengths.float() * ratio).long().clamp_min(min_frames)
    if max_frames > 0:
        widths = widths.clamp_max(max_frames)
    widths = torch.minimum(widths, (lengths - 2).clamp_min(0))
    start_count = (lengths - widths - 1).clamp_min(1)
    starts = (torch.rand(bsz, device=valid.device) * start_count.float()).floor().long() + 1
    single = interior & frame_index.ge(starts[:, None]) & frame_index.lt((starts + widths)[:, None])

    # Multi-span: draw up to four short spans and union them. Overlap is
    # intentional; it gives varied gap patterns without per-sample Python.
    span_count = torch.randint(2, 5, (bsz,), device=valid.device)
    target = torch.round(lengths.float() * ratio).long().clamp_min(min_frames)
    if max_frames > 0:
        target = target.clamp_max(max_frames)
    span_width = torch.div(target + span_count - 1, span_count, rounding_mode="floor").clamp_min(1)
    multi = torch.zeros_like(valid)
    for span_id in range(4):
        active = span_count > span_id
        count = (lengths - span_width - 1).clamp_min(1)
        start = (torch.rand(bsz, device=valid.device) * count.float()).floor().long() + 1
        multi |= active[:, None] & interior & frame_index.ge(start[:, None]) & frame_index.lt((start + span_width)[:, None])

    hidden = torch.where(modes[:, None].eq(0), bert, torch.where(modes[:, None].eq(1), single, multi))
    # Very short/degenerate samples can produce an empty random mask; fall
    # back to one interior frame so every eligible sample contributes loss.
    empty = hidden.sum(dim=1).eq(0) & lengths.ge(3)
    fallback = ((lengths // 2).clamp_min(1)).clamp_max(frames - 2)
    hidden.scatter_(1, fallback[:, None], hidden.gather(1, fallback[:, None]) | empty[:, None])
    hidden &= valid
    batch["frame_observed_mask"] = valid & ~hidden
    batch["loss_frame_mask"] = hidden


def validate_gt_pose_oracle_config(cfg):
    """Fail fast unless the oracle is a pure GT-pose-to-rotation experiment."""
    if not uses_gt_pose_oracle(cfg):
        return
    model_params = cfg["model"]["params"]
    if require_bool(
        model_params.get("auxiliary_pose_head", False),
        "model.params.auxiliary_pose_head",
    ):
        raise ValueError("gt_pose_oracle must disable auxiliary_pose_head")
    if model_params.get("root_translation_cfg") is not None:
        raise ValueError("gt_pose_oracle must disable root_translation_cfg")
    decoder_params = model_params["decoder_cfg"]["params"]
    if decoder_params.get("external_query_dim") is not None:
        raise ValueError(
            "gt_pose_oracle requires the Decoder pose encoder; remove "
            "model.params.decoder_cfg.params.external_query_dim"
        )
    weights = cfg["train"].get("weight", {})
    forbidden_weights = {
        name: float(weights.get(name, 0.0))
        for name in ("pose_position", "pose_velocity")
        if float(weights.get(name, 0.0)) != 0.0
    }
    if forbidden_weights:
        raise ValueError(
            "gt_pose_oracle cannot optimize the privileged pose target with "
            f"pose-head losses: {forbidden_weights}"
        )


def _configured_prefixes(value, name):
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list of strings")
    prefixes = tuple(str(prefix) for prefix in value)
    if any(not prefix for prefix in prefixes):
        raise ValueError(f"{name} cannot contain an empty prefix")
    return prefixes


def _prefix_matches_name(prefix, name):
    """Match a configured prefix independently of trailing-dot spelling."""
    prefix = str(prefix).rstrip(".")
    name = str(name)
    return name == prefix or name.startswith(prefix + ".")


def apply_eval_to_frozen_leaf_modules(model):
    """Put actual frozen stateful/random leaf modules in eval mode.

    Unlike the explicit `frozen_eval_module_prefixes` config, this helper
    derives the set from the current `requires_grad` graph so it cannot
    drift from the architecture.  It is intentionally conservative: only leaf
    modules are touched, so a frozen parent cannot recursively freeze a
    trainable descendant.
    """
    core_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    for module in core_model.modules():
        if len(list(module.children())) != 0:
            continue
        parameters = list(module.parameters(recurse=False))
        if any(parameter.requires_grad for parameter in parameters):
            continue
        if parameters or isinstance(
            module,
            (
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
            ),
        ):
            module.eval()


def configure_trainable_parameter_prefixes(model, train_cfg):
    """Optionally restrict optimization to explicitly named model prefixes.

    Prefix matching is normalized so callers may use dotted or undotted names.
    """
    prefixes = _configured_prefixes(
        train_cfg.get("trainable_parameter_prefixes"),
        "train.trainable_parameter_prefixes",
    )
    if not prefixes:
        return prefixes
    matched = {prefix: False for prefix in prefixes}
    for name, parameter in model.named_parameters():
        selected = any(_prefix_matches_name(prefix, name) for prefix in prefixes)
        parameter.requires_grad_(selected and parameter.requires_grad)
        for prefix in prefixes:
            if _prefix_matches_name(prefix, name):
                matched[prefix] = True
    unmatched = [prefix for prefix, found in matched.items() if not found]
    if unmatched:
        raise ValueError(
            "trainable_parameter_prefixes did not match model parameters: "
            f"{unmatched}"
        )
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("No trainable parameters remain after prefix filtering")
    return prefixes


def apply_frozen_eval_module_prefixes(model, train_cfg):
    """Keep frozen pretrained modules deterministic during adapter training."""
    prefixes = _configured_prefixes(
        train_cfg.get("frozen_eval_module_prefixes"),
        "train.frozen_eval_module_prefixes",
    )
    if not prefixes:
        return
    core_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    modules = dict(core_model.named_modules())
    missing = [prefix for prefix in prefixes if prefix not in modules]
    if missing:
        raise ValueError(
            "frozen_eval_module_prefixes did not match exact module names: "
            f"{missing}"
        )
    for prefix in prefixes:
        modules[prefix].eval()


def validate_checkpoint_init_config(cfg):
    """Fail fast when an initialization request cannot be satisfied."""
    model_params = cfg["model"]["params"]
    train_cfg = cfg["train"]
    init_path = train_cfg.get("init_from_checkpoint")
    resume_path = train_cfg.get("resume_from_checkpoint")
    if init_path and resume_path:
        raise ValueError(
            "train.init_from_checkpoint and resume_from_checkpoint are "
            "mutually exclusive"
        )
    extension_prefixes = set()
    encoder_params = model_params.get("encoder_cfg", {}).get("params", {})
    if bool(encoder_params.get("use_mask_frame_token", False)):
        extension_prefixes.add("encoder.mask_frame_token")
    allowed_missing = set(
        _configured_prefixes(
            train_cfg.get("init_allow_missing_prefixes"),
            "train.init_allow_missing_prefixes",
        )
    )
    if allowed_missing and not extension_prefixes:
        raise ValueError(
            "init_allow_missing_prefixes requires a model extension"
        )
    invalid_missing = sorted(allowed_missing - extension_prefixes)
    if invalid_missing:
        raise ValueError(
            "init_allow_missing_prefixes contains non-extension prefixes: "
            f"{invalid_missing}; expected subset of {sorted(extension_prefixes)}"
        )
    if init_path and extension_prefixes and not allowed_missing:
        raise ValueError(
            "Checkpoint initialization with a model extension must "
            "explicitly allow its new parameter prefix"
        )


def validate_observable_swing_config(cfg):
    """Fail fast for representation/loss combinations not implemented safely."""

    model_params = cfg["model"]["params"]
    if model_params.get("grace_cfg") is None:
        return
    decoder_params = model_params["decoder_cfg"]["params"]
    representation = str(
        decoder_params.get("rotation_representation", "rot6d")
    ).lower()
    if representation != "rot6d":
        raise ValueError(
            "observable swing correction requires rotation_representation=rot6d"
        )
    loss_type = str(
        cfg["train"].get("rotation_loss_type", "geodesic")
    ).lower()
    if loss_type not in {"geodesic", "chordal"}:
        raise ValueError(
            "observable swing correction supports only geodesic/chordal "
            "rotation losses; final corrected quaternion output is not defined"
        )


def build_scheduler(optimizer, train_cfg):
    total = require_positive_int(
        train_cfg.get(
            "lr_decay_steps",
            train_cfg["total_optimizer_steps"],
        ),
        "train.lr_decay_steps",
    )
    max_train_steps = require_positive_int(
        train_cfg["total_optimizer_steps"],
        "train.total_optimizer_steps",
    )
    has_warmup_steps = "lr_warmup_steps" in train_cfg
    has_warmup_ratio = "lr_warmup_ratio" in train_cfg
    if has_warmup_steps and has_warmup_ratio:
        raise ValueError(
            "Configure only one of train.lr_warmup_steps and "
            "train.lr_warmup_ratio"
        )
    if has_warmup_ratio:
        warmup_ratio = float(train_cfg["lr_warmup_ratio"])
        if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
            raise ValueError(
                "train.lr_warmup_ratio must be finite and in [0,1), "
                f"got {warmup_ratio}"
            )
        # The user-facing ratio is defined against max_train_steps, not a
        # separately configured decay horizon. Round to the nearest complete
        # optimizer update so 1/30 maps 3000->100 and 6000->200.
        warmup = int(round(max_train_steps * warmup_ratio))
    else:
        warmup = require_int(
            train_cfg.get("lr_warmup_steps", 0),
            "train.lr_warmup_steps",
        )
    min_lr = float(train_cfg.get("min_lr", 0.0))
    base_lr = float(train_cfg["lr"])
    if warmup < 0 or warmup >= total:
        raise ValueError(
            f"resolved train warmup steps must be in [0,{total}), got "
            f"{warmup}"
        )
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError(f"train.lr must be finite and > 0, got {base_lr}")
    if not math.isfinite(min_lr) or not 0 <= min_lr <= base_lr:
        raise ValueError(
            f"train.min_lr must be finite and in [0,lr], got {min_lr}"
        )
    min_ratio = min_lr / base_lr

    def multiplier(scheduler_step):
        # LambdaLR calls lambda(0) during construction; that value is used by
        # optimizer update 1. Map scheduler indices explicitly to one-based
        # successful optimizer updates so warmup/decay boundaries are exact.
        update = scheduler_step + 1
        if warmup and update <= warmup:
            return update / warmup
        if warmup:
            progress = (update - warmup) / (total - warmup)
        else:
            # With no warmup, update 1 uses base LR and update `total` reaches
            # min LR exactly.
            progress = (update - 1) / max(total - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def resolve_amp_dtype(train_cfg):
    """Resolve the configured CUDA autocast dtype."""
    name = str(train_cfg.get("amp_dtype", "float16")).lower()
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in aliases:
        raise ValueError(
            "train.amp_dtype must be one of "
            "{'float16','fp16','bfloat16','bf16'}, "
            f"got {name!r}"
        )
    return aliases[name]


def _scale_safe_tensor_norm(tensor):
    """Compute a finite tensor's L2 norm without squaring its raw magnitude."""
    detached = tensor.detach()
    max_abs = float(detached.abs().max())
    if max_abs == 0.0:
        return 0.0
    scaled_square_sum = float(
        torch.sum(
            (detached / max_abs).square(),
            dtype=torch.float64,
        )
    )
    return max_abs * math.sqrt(scaled_square_sum)


def _gradient_failure_report(model, model_out, batch, parts, global_step):
    """Collect evidence for a fatal gradient-norm failure; never modifies grads."""
    parameter_stats = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = bool(torch.isfinite(gradient).all())
        parameter_stats.append(
            {
                "name": name,
                "finite": finite,
                "max_abs": (
                    float(gradient.detach().abs().max())
                    if finite
                    else None
                ),
                "l2_norm": (
                    _scale_safe_tensor_norm(gradient)
                    if finite
                    else None
                ),
                "numel": gradient.numel(),
            }
        )
    finite_norms = [
        item["l2_norm"]
        for item in parameter_stats
        if item["finite"] and item["l2_norm"] is not None
    ]
    total_norm = math.hypot(*finite_norms) if finite_norms else 0.0
    parameter_stats.sort(
        key=lambda item: (
            item["l2_norm"] is not None,
            item["l2_norm"] or 0.0,
        ),
        reverse=True,
    )

    report = {
        "global_step_attempted": int(global_step),
        "loss_parts": {
            key: float(value.detach()) for key, value in parts.items()
        },
        "gradient_total_l2_float64": total_norm,
        "top_gradient_parameters": parameter_stats[:40],
        "batch": {
            "rel": list(batch.get("rel", [])),
            "window_start": (
                batch["window_start"].detach().cpu().tolist()
                if isinstance(batch.get("window_start"), torch.Tensor)
                else batch.get("window_start")
            ),
            "valid_length": (
                batch["valid_length"].detach().cpu().tolist()
                if isinstance(batch.get("valid_length"), torch.Tensor)
                else batch.get("valid_length")
            ),
            "frame_indices": (
                batch["frame_indices"].detach().cpu().tolist()
                if isinstance(batch.get("frame_indices"), torch.Tensor)
                else None
            ),
        },
    }

    raw_rot6d = model_out.get("pred_rotation_raw")
    if raw_rot6d is not None and raw_rot6d.shape[-1] == 6:
        raw = raw_rot6d.detach().float()
        a1 = raw[..., :3]
        a2 = raw[..., 3:]
        a1_norm = torch.linalg.vector_norm(a1, dim=-1)
        b1 = a1 / a1_norm.clamp_min(1e-30).unsqueeze(-1)
        a2_orthogonal = a2 - (a2 * b1).sum(
            dim=-1, keepdim=True
        ) * b1
        a2_orthogonal_norm = torch.linalg.vector_norm(
            a2_orthogonal, dim=-1
        )
        valid = batch["joint_mask"].bool().unsqueeze(1)
        frame_valid = batch.get("frame_valid_mask")
        if frame_valid is not None:
            valid = valid & frame_valid.bool().unsqueeze(-1)
        a1_valid = a1_norm[valid]
        a2_valid = a2_orthogonal_norm[valid]
        report["raw_rot6d"] = {
            "max_abs": float(raw.abs().max()),
            "valid_count": int(a1_valid.numel()),
        }
        if a1_valid.numel():
            report["raw_rot6d"].update(
                {
                    "a1_norm_min": float(a1_valid.min()),
                    "a1_norm_mean": float(a1_valid.mean()),
                    "a1_norm_below_1e-4": int((a1_valid < 1e-4).sum()),
                    "a1_norm_below_1e-6": int((a1_valid < 1e-6).sum()),
                    "a2_orthogonal_norm_min": float(a2_valid.min()),
                    "a2_orthogonal_norm_mean": float(a2_valid.mean()),
                    "a2_orthogonal_norm_below_1e-4": int(
                        (a2_valid < 1e-4).sum()
                    ),
                    "a2_orthogonal_norm_below_1e-6": int(
                        (a2_valid < 1e-6).sum()
                    ),
                }
            )
    return report


def _detach_tree_to_cpu(value):
    """Detach every tensor in a nested training object for crash replay."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {
            key: _detach_tree_to_cpu(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_detach_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree_to_cpu(item) for item in value)
    return value


def _save_gradient_failure_bundle(
    *,
    base_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    cfg,
    batch,
    model_out,
    parts,
    report,
    completed_global_step,
    attempted_global_step,
):
    """Write the diagnostic files for a fatal backward failure.

    Files are intentionally separate: a partially written multi-gigabyte
    monolith would make otherwise valid batch/gradient evidence unusable.
    ``manifest.json`` is written last and marks a complete crash bundle.
    """
    bundle_dir = os.path.join(
        base_dir,
        f"gradient_failure_step_{attempted_global_step:06d}",
    )
    os.makedirs(bundle_dir, exist_ok=True)
    failure_model = model.module if hasattr(model, "module") else model

    files = {
        "report": "report.json",
        "training_state": "training_state.pt",
        "gradients": "gradients.pt",
        "batch": "batch.pt",
        "outputs": "outputs.pt",
        "rng": "rng_state.pt",
    }
    with open(
        os.path.join(bundle_dir, files["report"]),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    torch.save(
        {
            "optimizer_state": _detach_tree_to_cpu(optimizer.state_dict()),
            "scheduler_state": _detach_tree_to_cpu(scheduler.state_dict()),
            "scaler_state": _detach_tree_to_cpu(scaler.state_dict()),
            "completed_global_step": int(completed_global_step),
            "attempted_global_step": int(attempted_global_step),
            "config": copy.deepcopy(cfg),
        },
        os.path.join(bundle_dir, files["training_state"]),
    )
    torch.save(
        {
            name: parameter.grad.detach().cpu()
            for name, parameter in failure_model.named_parameters()
            if parameter.grad is not None
        },
        os.path.join(bundle_dir, files["gradients"]),
    )
    # This is the post-collation, exact batch used by forward, including RGB,
    # masks, camera fields, rotations, animation labels, and frame indices.
    torch.save(
        _detach_tree_to_cpu(batch),
        os.path.join(bundle_dir, files["batch"]),
    )
    torch.save(
        {
            "model_out": _detach_tree_to_cpu(model_out),
            "loss_parts": _detach_tree_to_cpu(parts),
        },
        os.path.join(bundle_dir, files["outputs"]),
    )
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
        },
        os.path.join(bundle_dir, files["rng"]),
    )

    manifest = {
        "complete": True,
        "completed_global_step": int(completed_global_step),
        "attempted_global_step": int(attempted_global_step),
        "files": files,
    }
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return bundle_dir


def _stable_geodesic_error(pred_matrix, target_matrix):
    """SO(3) geodesic angle using atan2 instead of unstable acos(trace)."""
    relative = pred_matrix @ target_matrix.transpose(-1, -2)
    skew_vector = 0.5 * torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = torch.linalg.vector_norm(skew_vector, dim=-1)
    cos_angle = 0.5 * (
        relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    )
    return torch.atan2(
        sin_angle.clamp(min=0.0, max=1.0),
        cos_angle.clamp(min=-1.0, max=1.0),
    )


def _temporal_continuous_quaternion(quaternion):
    """Choose q/-q signs so adjacent valid sequence entries share a hemisphere."""
    if quaternion.shape[1] < 2:
        return quaternion
    frames = [quaternion[:, 0]]
    for frame_index in range(1, quaternion.shape[1]):
        current = quaternion[:, frame_index]
        previous = frames[-1]
        flip = (current * previous).sum(dim=-1, keepdim=True) < 0
        frames.append(torch.where(flip, -current, current))
    return torch.stack(frames, dim=1)


def _quaternion_relative_sequence(quaternion):
    continuous = _temporal_continuous_quaternion(quaternion)
    relative = quaternion_raw_multiply(
        continuous[:, 1:], quaternion_invert(continuous[:, :-1])
    )
    return standardize_quaternion(relative)


def _rotation_error(
    pred_matrix,
    target_matrix,
    loss_type,
    pred_quaternion=None,
    target_quaternion=None,
):
    if loss_type == "chordal":
        return (pred_matrix - target_matrix).square().mean(dim=(-1, -2))
    if loss_type == "geodesic":
        angle = _stable_geodesic_error(pred_matrix, target_matrix)
        return F.smooth_l1_loss(
            angle, torch.zeros_like(angle), reduction="none"
        )
    if loss_type == "quaternion_smooth_l1":
        if pred_quaternion is None or target_quaternion is None:
            raise ValueError(
                "quaternion_smooth_l1 requires predicted and target quaternions"
            )
        # q and -q encode exactly the same SO(3) rotation.  Independently
        # forcing both quaternions into the w >= 0 hemisphere is still
        # discontinuous at rotations of pi: two rotations separated by a
        # tiny angle can have almost-opposite quaternion coordinates.  Align
        # the target to the prediction's closest representative before
        # applying the coordinate-space regression loss.
        target_sign = torch.where(
            (pred_quaternion.detach() * target_quaternion)
            .sum(dim=-1, keepdim=True)
            < 0,
            -torch.ones_like(target_quaternion[..., :1]),
            torch.ones_like(target_quaternion[..., :1]),
        )
        aligned_target = target_quaternion * target_sign
        return F.smooth_l1_loss(
            pred_quaternion, aligned_target, reduction="none"
        ).mean(dim=-1)
    raise ValueError(
        "train.rotation_loss_type must be one of "
        "{'chordal','geodesic','quaternion_smooth_l1'}, "
        f"got {loss_type!r}"
    )


def _adjacent_rotation_speed(rotation_matrix):
    """Return unsigned adjacent-frame SO(3) angles in radians."""
    if rotation_matrix.shape[1] < 2:
        return rotation_matrix.new_zeros(
            rotation_matrix.shape[:1] + (0,) + rotation_matrix.shape[2:3]
        )
    relative = (
        rotation_matrix[:, 1:]
        @ rotation_matrix[:, :-1].transpose(-1, -2)
    )
    identity = torch.eye(
        3,
        dtype=relative.dtype,
        device=relative.device,
    ).view(1, 1, 1, 3, 3)
    return _stable_geodesic_error(relative, identity.expand_as(relative))


def _frame_motion_importance(edge_speed, frame_mask, joint_mask):
    """Map detached GT adjacent-rotation magnitudes to frame weights."""
    batch_size = edge_speed.shape[0]
    frames = frame_mask.shape[1]
    joints = joint_mask.shape[1]
    importance = edge_speed.new_zeros((batch_size, frames, joints))
    if frames > 1:
        edge_valid = frame_mask[:, 1:] & frame_mask[:, :-1]
        valid_edge_speed = edge_speed * edge_valid.unsqueeze(-1).to(
            dtype=edge_speed.dtype
        )
        importance[:, 0] = valid_edge_speed[:, 0]
        importance[:, -1] = valid_edge_speed[:, -1]
        if frames > 2:
            importance[:, 1:-1] = torch.maximum(
                valid_edge_speed[:, :-1], valid_edge_speed[:, 1:]
            )
    valid = frame_mask.unsqueeze(-1) & joint_mask.unsqueeze(1)
    return importance.detach().clamp_min(0.0), valid


def _motion_weighted_mean(
    error, importance, valid, joint_importance=None, *, per_rig=False
):
    weight = importance * valid.to(dtype=importance.dtype)
    if joint_importance is not None:
        weight = weight * _expand_joint_importance(
            joint_importance, error, joint_axis=-1
        )
    return _weighted_loss_mean(error, weight, per_rig=per_rig)


def _expand_joint_importance(joint_importance, error, *, joint_axis):
    """Broadcast ``[B,J]`` importance onto an elementwise joint loss."""
    if joint_importance is None:
        return None
    if error.ndim < 2 or joint_importance.ndim != 2:
        raise ValueError("joint importance requires error [...,J,...] and [B,J]")
    joint_axis = joint_axis % error.ndim
    if joint_axis == 0 or error.shape[0] != joint_importance.shape[0]:
        raise ValueError("invalid batch/joint axis for joint importance")
    if error.shape[joint_axis] != joint_importance.shape[1]:
        raise ValueError(
            f"joint importance width mismatch: {error.shape}/{joint_importance.shape}"
        )
    shape = [1] * error.ndim
    shape[0] = joint_importance.shape[0]
    shape[joint_axis] = joint_importance.shape[1]
    return joint_importance.to(dtype=error.dtype).reshape(shape)


def _joint_weighted_loss_mean(
    error, valid, joint_importance, *, joint_axis=-1, per_rig=False
):
    """Reduce a joint-resolved loss with optional static mesh importance."""
    if error.shape != valid.shape:
        raise ValueError(f"loss/valid shape mismatch: {error.shape}/{valid.shape}")
    weight = valid.to(dtype=error.dtype)
    if joint_importance is not None:
        weight = weight * _expand_joint_importance(
            joint_importance, error, joint_axis=joint_axis
        )
    return _weighted_loss_mean(error, weight, per_rig=per_rig)


def _zero_rest_edge_virtual_camera_directions(
    batch,
    parents,
    joint_mask,
    *,
    axis,
    edge_epsilon,
    virtual_length,
):
    """Rotate one parent-local virtual axis into camera space."""

    axis = canonical_fallback_axis(axis)
    joints = parents.shape[1]
    zero_rest_edge = (
        joint_mask
        & parents.ge(0)
        & parents.lt(joints)
        & torch.linalg.vector_norm(
            batch["offset_a"].float(), dim=-1
        ).le(float(edge_epsilon))
    )
    if not bool(zero_rest_edge.any()):
        position = batch["position"]
        return position.new_zeros(position.shape), zero_rest_edge
    if "rot6d_a" not in batch:
        raise KeyError(
            "parent_gt_virtual_edge BoneDirection targets require batch['rot6d_a']"
        )
    virtual_length = float(virtual_length)
    if not math.isfinite(virtual_length) or virtual_length <= 0.0:
        raise ValueError(
            "train.zero_rest_edge_virtual_length must be finite and > 0"
        )

    position = batch["position"].float()
    local_rotation = rot6d_to_rotmat_tensor(batch["rot6d_a"].float())
    if local_rotation.shape[:3] != position.shape[:3]:
        raise ValueError(
            "rot6d_a/position shape mismatch for virtual zero-rest edges"
        )
    _, parent_global_rotation = _differentiable_local_rotation_fk_transforms(
        local_rotation, batch["offset_a"].float(), parents,
        layout=batch.get("_fk_layout"),
    )

    batch_size, frames, joints = position.shape[:3]
    safe_parent = parents.clamp(min=0, max=max(joints - 1, 0))
    parent_rotation = torch.gather(
        parent_global_rotation,
        2,
        safe_parent[:, None, :, None, None].expand(
            batch_size, frames, joints, 3, 3
        ),
    )
    local_axis = position.new_zeros(3)
    local_axis[{"x": 0, "y": 1, "z": 2}[axis]] = virtual_length
    direction = (
        parent_rotation @ local_axis.view(1, 1, 1, 3, 1)
    ).squeeze(-1)
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(float(edge_epsilon))
    return direction, zero_rest_edge


def _bone_direction_supervision(
    model_out,
    batch,
    reference,
    cfg=None,
    joint_importance=None,
    *,
    per_rig=False,
    edge_epsilon=None,
):
    """Return direction and confidence supervision without diagnostic metrics."""

    zero = reference.new_zeros(())
    if "pred_bone_direction" not in model_out:
        return zero
    if edge_epsilon is None:
        edge_epsilon = (
            resolve_edge_epsilon(cfg)
            if cfg is not None
            else DEFAULT_EDGE_EPSILON
        )
    required = ("position", "parent_a", "offset_a", "joint_mask")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"bone-direction supervision requires {missing}")
    pred_direction = model_out["pred_bone_direction"].float()
    pred_confidence = model_out["pred_bone_confidence"].float()
    parents = batch["parent_a"].long()
    joint_mask = batch["joint_mask"].bool()
    frame_mask = batch.get(
        "loss_frame_mask",
        batch.get(
            "frame_valid_mask",
            torch.ones(
                pred_direction.shape[:2],
                dtype=torch.bool,
                device=pred_direction.device,
            ),
        ),
    ).bool()
    train_cfg = cfg.get("train", {}) if cfg is not None else {}
    target_mode = str(
        train_cfg.get("zero_rest_edge_target_mode", "fixed_camera_axis")
    ).lower()
    if target_mode not in {"fixed_camera_axis", "parent_gt_virtual_edge"}:
        raise ValueError(
            "train.zero_rest_edge_target_mode must be 'fixed_camera_axis' "
            "or 'parent_gt_virtual_edge'"
        )
    gt_info = positions_to_bone_directions(
        batch["position"].float(),
        parents,
        batch["offset_a"].float(),
        joint_mask,
        frame_mask,
        edge_epsilon=edge_epsilon,
        zero_rest_edge_fallback_axis=(
            cfg.get("model", {})
            .get("params", {})
            .get("zero_rest_edge_fallback_axis", "x")
            if cfg is not None
            else "x"
        ),
        include_zero_rest_edges=target_mode == "parent_gt_virtual_edge",
    )
    gt_direction = gt_info["direction"]
    valid = gt_info["valid_mask"]
    if target_mode == "parent_gt_virtual_edge":
        virtual_direction, zero_rest_edge = (
            _zero_rest_edge_virtual_camera_directions(
                batch,
                parents,
                joint_mask,
                axis=cfg.get("model", {})
                .get("params", {})
                .get("zero_rest_edge_fallback_axis", "x"),
                edge_epsilon=edge_epsilon,
                virtual_length=train_cfg.get(
                    "zero_rest_edge_virtual_length", 1e-4
                ),
            )
        )
        gt_direction = torch.where(
            zero_rest_edge[:, None, :, None],
            virtual_direction,
            gt_direction,
        )
    repair_cfg = (
        cfg.get("train", {}).get("swing_loss_repair", {})
        if cfg is not None
        else {}
    )
    exclude_zero_rest_edges = bool(
        repair_cfg.get("exclude_zero_rest_edges", False)
    )
    nonzero_rest_edge = torch.linalg.vector_norm(
        batch["offset_a"].float(), dim=-1
    ).gt(edge_epsilon)
    if exclude_zero_rest_edges:
        valid = valid & nonzero_rest_edge.unsqueeze(1)
    if pred_direction.shape != gt_direction.shape:
        raise ValueError("predicted/GT bone-direction shape mismatch")
    if pred_confidence.shape != valid.shape:
        raise ValueError("bone-confidence shape mismatch")
    dot = (pred_direction * gt_direction).sum(dim=-1).clamp(-1.0, 1.0)
    cosine_error = 1.0 - dot
    safe_parent = parents.clamp(min=0, max=max(parents.shape[1] - 1, 0))
    owner_importance = None
    if joint_importance is not None:
        owner_importance = torch.gather(joint_importance, 1, safe_parent)
    direction_loss = _joint_weighted_loss_mean(
        cosine_error, valid, owner_importance, per_rig=per_rig
    )
    angle = torch.acos(dot)
    confidence_target = torch.exp(-angle.detach() / math.radians(20.0))
    confidence_error = F.smooth_l1_loss(
        pred_confidence, confidence_target, reduction="none"
    )
    confidence_calibration = _joint_weighted_loss_mean(
        confidence_error, valid, owner_importance, per_rig=per_rig
    )
    combined_loss = direction_loss + 0.1 * confidence_calibration

    return combined_loss


def _weighted_loss_mean(error, weight, *, per_rig):
    """Reduce an elementwise loss globally or with equal weight per rig."""
    if error.shape != weight.shape:
        raise ValueError(
            f"loss/weight shape mismatch: {error.shape}/{weight.shape}"
        )
    if error.ndim < 2:
        raise ValueError("weighted loss tensors must include batch and data axes")
    weight = weight.to(dtype=error.dtype)
    if not per_rig:
        return (error * weight).sum() / weight.sum().clamp(min=1e-8)
    dimensions = tuple(range(1, error.ndim))
    numerator = (error * weight).sum(dim=dimensions)
    denominator = weight.sum(dim=dimensions)
    valid_rig = denominator > 1e-8
    per_rig_loss = numerator / denominator.clamp(min=1e-8)
    return (
        per_rig_loss * valid_rig.to(per_rig_loss.dtype)
    ).sum() / valid_rig.sum().clamp(min=1).to(per_rig_loss.dtype)


def _configured_weighted_loss_mean(
    error,
    valid,
    *,
    importance=None,
    per_rig=False,
    joint_importance=None,
    joint_axis=-1,
):
    """Apply validity, optional motion, and optional static joint weights."""
    if error.shape != valid.shape:
        raise ValueError(
            f"loss/valid shape mismatch: {error.shape}/{valid.shape}"
        )
    weight = valid.to(dtype=error.dtype)
    if importance is not None:
        if importance.shape != error.shape:
            raise ValueError(
                "loss/importance shape mismatch: "
                f"{error.shape}/{importance.shape}"
            )
        weight = weight * importance.detach().to(dtype=error.dtype)
    if joint_importance is not None:
        weight = weight * _expand_joint_importance(
            joint_importance, error, joint_axis=joint_axis
        )
    return _weighted_loss_mean(error, weight, per_rig=per_rig)


def compute_losses(model_out, batch, cfg):
    final_pred_rot6d = model_out["pred_rot6d"]
    # A surface residual is an auxiliary correction head.  All pre-existing
    # rotation/FK objectives must continue to optimize the base prediction
    # exactly as they do in the baseline.  Supervising R_base @ Delta with the
    # old losses turns the two heads into a non-identifiable coupled decoder.
    observable_swing_applied = bool(
        model_out.get("observable_swing_applied", False)
    )
    base_rotation_matrix = (
        None
        if observable_swing_applied
        else model_out.get("pred_base_rotation_matrix")
    )
    if base_rotation_matrix is None:
        pred_rot6d = final_pred_rot6d
    else:
        pred_rot6d = model_out.get("pred_base_rot6d")
        if pred_rot6d is None:
            pred_rot6d = rotmat_to_rot6d_tensor(
                base_rotation_matrix.float()
            )
    gt_rot6d = batch["rot6d_a"]
    joint_mask = batch["joint_mask"].bool()
    joint_importance = None
    weights = cfg["train"]["weight"]
    frame_mask = batch.get(
        "loss_frame_mask",
        batch.get(
        "frame_valid_mask",
        torch.ones(pred_rot6d.shape[:2], dtype=torch.bool, device=pred_rot6d.device),
        ),
    ).bool()
    skeleton_normalize = require_bool(
        cfg.get("train", {}).get("skeleton_normalize", False),
        "train.skeleton_normalize",
    )
    skeleton_normalize_smooth_l1_beta = 1.0
    if skeleton_normalize:
        skeleton_normalize_smooth_l1_beta = float(
            cfg.get("train", {}).get(
                "skeleton_normalize_smooth_l1_beta", 1.0
            )
        )
        if (
            not math.isfinite(skeleton_normalize_smooth_l1_beta)
            or skeleton_normalize_smooth_l1_beta <= 0.0
        ):
            raise ValueError(
                "train.skeleton_normalize_smooth_l1_beta must be finite and > 0"
            )
    if skeleton_normalize:
        if "metric_scale" not in batch:
            raise KeyError(
                "train.skeleton_normalize requires batch['metric_scale']"
            )
        metric_scale = batch["metric_scale"].float()
        if metric_scale.numel() != pred_rot6d.shape[0]:
            raise ValueError("metric_scale must contain one value per rig")
        metric_scale = metric_scale.reshape(pred_rot6d.shape[0])
        if not torch.isfinite(metric_scale).all() or not bool(
            (metric_scale > 0).all()
        ):
            raise ValueError("metric_scale must be finite and positive")
    rotation_joint_mask = joint_mask
    temporal_rotation_mask = (
        frame_mask.unsqueeze(-1) & rotation_joint_mask.unsqueeze(1)
    )
    nonroot_mask = joint_mask.clone()
    nonroot_mask[:, 0] = False
    temporal_nonroot_mask = frame_mask.unsqueeze(-1) & nonroot_mask.unsqueeze(1)
    # Optimize decoded rotations, not arbitrary 6D coordinates. Squared
    # chordal distance on rotation matrices has finite gradients at identity
    # and pi, unlike an angle loss whose derivative is singular at boundaries.
    if base_rotation_matrix is not None:
        pred_mat = base_rotation_matrix.float()
        if pred_mat.shape != pred_rot6d.shape[:-1] + (3, 3):
            raise ValueError(
                "pred_base_rotation_matrix shape mismatch: "
                f"{pred_mat.shape}/{pred_rot6d.shape}"
            )
    elif model_out.get("pred_rotation_matrix") is not None:
        pred_mat = model_out["pred_rotation_matrix"].float()
        if pred_mat.shape != pred_rot6d.shape[:-1] + (3, 3):
            raise ValueError(
                "pred_rotation_matrix shape mismatch: "
                f"{pred_mat.shape}/{pred_rot6d.shape}"
            )
    else:
        pred_mat = rot6d_to_rotmat_tensor(pred_rot6d.float())
    gt_mat = rot6d_to_rotmat_tensor(gt_rot6d.float())
    loss_type = str(
        cfg["train"].get("rotation_loss_type", "geodesic")
    ).lower()
    # Surface residuals currently require rot6d, so a quaternion output can
    # only describe the non-residual prediction selected above.
    pred_quaternion = (
        None
        if base_rotation_matrix is not None
        else model_out.get("pred_quaternion")
    )
    gt_quaternion = None
    if loss_type == "quaternion_smooth_l1":
        if pred_quaternion is None:
            raise KeyError(
                "quaternion_smooth_l1 requires "
                "model_out['pred_quaternion'] from a quaternion head"
            )
        pred_quaternion = F.normalize(
            pred_quaternion.float(), dim=-1, eps=1e-8
        )
        gt_quaternion = matrix_to_quaternion(gt_mat)
    rotation_error = _rotation_error(
        pred_mat,
        gt_mat,
        loss_type,
        pred_quaternion,
        gt_quaternion,
    )
    gt_edge_speed = _adjacent_rotation_speed(gt_mat)
    motion_importance, motion_valid = _frame_motion_importance(
        gt_edge_speed,
        frame_mask,
        rotation_joint_mask,
    )
    if require_bool(
        cfg["train"].get("rotation_motion_weighted", False),
        "train.rotation_motion_weighted",
    ):
        rot_loss = _motion_weighted_mean(
            rotation_error,
            motion_importance,
            motion_valid,
            joint_importance,
            per_rig=skeleton_normalize,
        )
    else:
        rot_loss = _joint_weighted_loss_mean(
            rotation_error,
            temporal_rotation_mask,
            joint_importance,
            per_rig=skeleton_normalize,
        )
    pred_delta = pred_mat[:, 1:] @ pred_mat[:, :-1].transpose(-1, -2)
    gt_delta = gt_mat[:, 1:] @ gt_mat[:, :-1].transpose(-1, -2)
    pred_delta_quaternion = None
    gt_delta_quaternion = None
    if loss_type == "quaternion_smooth_l1":
        pred_delta_quaternion = _quaternion_relative_sequence(
            pred_quaternion
        )
        gt_delta_quaternion = _quaternion_relative_sequence(gt_quaternion)
    # Local angular-velocity supervision. Compare the relative rotation of
    # each adjacent frame pair directly with GT. This complements the
    # per-frame reset-relative rotation objective above with local temporal
    # information. It remains an auxiliary loss because a uniformly biased
    # velocity sequence produces cancelling gradients at interior absolute
    # pose outputs.
    velocity_error = _rotation_error(
        pred_delta,
        gt_delta,
        loss_type,
        pred_delta_quaternion,
        gt_delta_quaternion,
    )
    velocity_frame_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
    velocity_rotation_mask = (
        velocity_frame_mask.unsqueeze(-1)
        & rotation_joint_mask.unsqueeze(1)
    )
    rot_vel_loss = _joint_weighted_loss_mean(
        velocity_error,
        velocity_rotation_mask,
        joint_importance,
        per_rig=skeleton_normalize,
    )
    pose_weight = float(weights.get("pose_position", 0.0))
    pose_velocity_weight = float(weights.get("pose_velocity", 0.0))
    if pose_weight or pose_velocity_weight:
        if "pred_position" not in model_out or "position" not in batch:
            raise KeyError(
                "Pose position/velocity loss requires "
                "model_out['pred_position'] and batch['position']"
            )
        # The Encoder head already predicts in the normalized skeleton
        # coordinate system because ref_position/offset_a were normalized at
        # the model-input boundary. Only the raw dataset GT needs conversion.
        pred_pose = model_out["pred_position"].float()
        gt_pose = batch["position"].float()
        if skeleton_normalize:
            gt_pose = gt_pose * metric_scale.view(-1, 1, 1, 1)
        if pred_pose.shape != gt_pose.shape:
            raise ValueError(
                "Pose prediction/target shape mismatch: "
                f"{pred_pose.shape}/{gt_pose.shape}"
            )
        if pose_weight:
            pose_error = F.smooth_l1_loss(
                pred_pose,
                gt_pose,
                reduction="none",
                beta=skeleton_normalize_smooth_l1_beta,
            ).mean(dim=-1)
            pose_position_loss = _joint_weighted_loss_mean(
                pose_error,
                temporal_nonroot_mask,
                joint_importance,
                per_rig=skeleton_normalize,
            )
        else:
            pose_position_loss = rot_loss.new_zeros(())

        if pose_velocity_weight:
            pred_pose_delta = pred_pose[:, 1:] - pred_pose[:, :-1]
            gt_pose_delta = gt_pose[:, 1:] - gt_pose[:, :-1]
            pose_velocity_error = F.smooth_l1_loss(
                pred_pose_delta,
                gt_pose_delta,
                reduction="none",
                beta=skeleton_normalize_smooth_l1_beta,
            ).mean(dim=-1)
            pose_velocity_frame_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
            pose_velocity_mask = (
                pose_velocity_frame_mask.unsqueeze(-1)
                & nonroot_mask.unsqueeze(1)
            )
            pose_velocity_loss = _joint_weighted_loss_mean(
                pose_velocity_error,
                pose_velocity_mask,
                joint_importance,
                per_rig=skeleton_normalize,
            )
        else:
            pose_velocity_loss = rot_loss.new_zeros(())
    else:
        pose_position_loss = rot_loss.new_zeros(())
        pose_velocity_loss = rot_loss.new_zeros(())

    root_rotation_weight = float(weights.get("root_rotation", 0.0))
    root_rotation_mask = frame_mask & joint_mask[:, 0:1]
    root_rotation_error = _rotation_error(
        pred_mat[:, :, 0],
        gt_mat[:, :, 0],
        loss_type,
        (
            pred_quaternion[:, :, 0]
            if pred_quaternion is not None
            else None
        ),
        (
            gt_quaternion[:, :, 0]
            if gt_quaternion is not None
            else None
        ),
    )
    root_rotation_loss = _joint_weighted_loss_mean(
        root_rotation_error.unsqueeze(-1),
        root_rotation_mask.unsqueeze(-1),
        None if joint_importance is None else joint_importance[:, :1],
        per_rig=skeleton_normalize,
    )

    fk_weight = float(weights.get("fk_position", 0.0))
    pred_fk_position = None
    pred_global_rotation = None
    if fk_weight:
        required = ["offset_a", "parent_a"]
        if fk_weight:
            required.append("position")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "FK position loss requires batch keys: "
                f"{missing}"
            )
        fk_offset = batch["offset_a"].float()
        if skeleton_normalize:
            fk_offset = fk_offset * metric_scale.view(-1, 1, 1)
        pred_fk_position, pred_global_rotation = (
            _differentiable_local_rotation_fk_transforms(
                pred_mat,
                fk_offset,
                batch["parent_a"].long(),
                layout=batch.get("_fk_layout"),
            )
        )

    if fk_weight:
        assert pred_fk_position is not None
        # Use real dataset positions as the target. Do not synthesize the GT
        # side again from rotations and the current skeleton offsets.
        gt_position = batch.get(
            "fk_position_target", batch["position"]
        ).float()
        if skeleton_normalize:
            gt_position = gt_position * metric_scale.view(-1, 1, 1, 1)
        fk_difference = pred_fk_position - gt_position
        # With skeleton_normalize enabled, both FK offsets and the position
        # target are scaled once by metric_scale into the 2 m convention.
        # Disabled training stays in the dataset's original coordinate scale.
        # Validation separately uses TopoCap's rest-extent normalization.
        # Coordinate-wise SmoothL1 has a defined zero-error gradient.
        fk_error = F.smooth_l1_loss(
            fk_difference,
            torch.zeros_like(fk_difference),
            reduction="none",
            beta=skeleton_normalize_smooth_l1_beta,
        ).mean(dim=-1)
        fk_position_loss = _joint_weighted_loss_mean(
            fk_error,
            temporal_nonroot_mask,
            joint_importance,
            per_rig=skeleton_normalize,
        )
    else:
        fk_position_loss = rot_loss.new_zeros(())

    bone_direction_weight = float(weights.get("bone_direction", 0.0))
    bone_direction_loss = _bone_direction_supervision(
        model_out,
        batch,
        rot_loss,
        cfg,
        joint_importance,
        per_rig=skeleton_normalize,
    )
    if bone_direction_weight and "pred_bone_direction" not in model_out:
        raise KeyError(
            "configured direction losses require a bone-direction prediction"
        )
    weighted_losses = (
        (float(weights["rotation"]), rot_loss),
        (float(weights["rotation_velocity"]), rot_vel_loss),
        (pose_weight, pose_position_loss),
        (pose_velocity_weight, pose_velocity_loss),
        (bone_direction_weight, bone_direction_loss),
        (root_rotation_weight, root_rotation_loss),
        (fk_weight, fk_position_loss),
    )
    active_weighted_losses = [
        weight * loss for weight, loss in weighted_losses if weight != 0.0
    ]
    total = (
        functools.reduce(torch.add, active_weighted_losses)
        if active_weighted_losses
        else rot_loss * 0.0
    )
    parts = {
        "rotation": rot_loss,
        "rotation_velocity": rot_vel_loss,
        "pose_position": pose_position_loss,
        "pose_velocity": pose_velocity_loss,
        "root_rotation": root_rotation_loss,
        "fk_position": fk_position_loss,
    }
    if "bone_direction" in weights or "pred_bone_direction" in model_out:
        parts["bone_direction"] = bone_direction_loss
    return total, parts


def _differentiable_local_rotation_fk_transforms(
    local_rotation,
    rest_offset,
    parents,
    layout=None,
):
    """Return differentiable root-centered positions and global rotations."""
    return depth_fk(local_rotation, rest_offset, parents, layout=layout)


def _generic_joint_names(joint_count):
    """Return neutral per-joint labels for validation rendering.

    Rendering only needs stable, per-joint labels so `bone_color` can assign
    deterministic colors.  Cache readers keep the authoritative joint names in
    their static metadata, which is intentionally not re-read here; validation
    visualization must stay independent of any one cache layout.
    """
    return [f"joint_{index}" for index in range(joint_count)]


def _render_validation_motion(entry, output_path, cfg):
    pred_local = rotation_matrices_from_fused_sum(
        entry["rot_sum"], entry["count"]
    )
    gt_local = rotation_matrices_from_fused_sum(
        entry["gt_rot_sum"], entry["count"]
    )
    gt_position = local_rotation_fk(
        gt_local,
        entry["offsets"],
        entry["parents"],
    )
    pred_position = local_rotation_fk(
        pred_local,
        entry["offsets"],
        entry["parents"],
    )
    visual_cfg = cfg["eval"]["visualization"]
    use_predicted_root = require_bool(
        visual_cfg.get("use_predicted_root_if_available", True),
        "eval.visualization.use_predicted_root_if_available",
    )
    joint_names = entry.get("joint_names")
    if joint_names is None:
        joint_names = _generic_joint_names(len(entry["parents"]))
    render_frame_sources(
        entry["frame_sources"],
        gt_position,
        pred_position,
        entry["parents"],
        joint_names,
        output_path,
        fps=float(visual_cfg.get("fps", cfg["data"].get("fps", 30.0))),
        clip=str(entry["rel"]).split("/", 1)[-1],
        camera_root_position=entry["camera_root_position"],
        pred_camera_root_position=(
            entry.get("pred_camera_root_position")
            if use_predicted_root
            else None
        ),
        camera_intrinsics=entry["camera_intrinsics"],
        auxiliary_pose_position=entry.get("pose_head_position"),
    )


def resolve_eval_aggregation(cfg):
    """Resolve motion-weighted or equal-skeleton validation aggregation."""
    aggregation = str(
        cfg.get("eval", {}).get("aggregation", "per_motion")
    ).lower()
    if aggregation not in {"per_motion", "per_skeleton"}:
        raise ValueError(
            "eval.aggregation must be 'per_motion' or 'per_skeleton', "
            f"got {aggregation!r}"
        )
    return aggregation


def _resume_validation_state(checkpoint):
    """Discard old diagnostics and never compare different normalization protocols."""
    metrics = {
        key: value for key, value in checkpoint.get("metrics", {}).items()
        if any(key == name or key.endswith("_" + name) for name in TOPOCAP_METRIC_KEYS)
    }
    if checkpoint.get("selection_protocol") != CHECKPOINT_SELECTION_PROTOCOL:
        return metrics, float("inf"), float("inf")
    score = float(checkpoint.get("selection_score", float("inf")))
    return metrics, float(checkpoint.get("best_metric", score)), score


def _checkpoint_selection_score(metrics, cfg, primary_val_name, multiple_splits):
    """Rank checkpoints using one TopoCap metric, optionally across splits."""
    eval_cfg = cfg.get("eval", {})
    selection_metric = eval_cfg.get("selection_metric")
    selection_weights = eval_cfg.get("selection_weights")
    if selection_metric is not None:
        selection_metric = str(selection_metric)
        if selection_metric not in metrics:
            raise KeyError(
                "Configured eval.selection_metric is unavailable: "
                f"{selection_metric!r}; keys={sorted(metrics)}"
            )
        selection_score = float(metrics[selection_metric])
    elif selection_weights is not None:
        if not isinstance(selection_weights, dict) or not selection_weights:
            raise ValueError("eval.selection_weights must be a non-empty mapping")
        metric_suffix = str(
            eval_cfg.get("selection_weighted_metric", "topocap_mpjpe_mm")
        )
        weighted_values = []
        for name, raw_weight in selection_weights.items():
            metric_name = f"{name}_{metric_suffix}"
            if metric_name not in metrics:
                raise KeyError(
                    "Configured weighted selection metric is unavailable: "
                    f"{metric_name!r}; keys={sorted(metrics)}"
                )
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    "eval.selection_weights values must be finite and positive"
                )
            weighted_values.append((weight, float(metrics[metric_name])))
        total_weight = sum(weight for weight, _ in weighted_values)
        selection_score = sum(
            weight * value for weight, value in weighted_values
        ) / total_weight
    elif multiple_splits:
        selection_score = float(
            metrics[f"{primary_val_name}_topocap_mpjpe_mm"]
        )
    else:
        selection_score = metrics["topocap_mpjpe_mm"]
    return float(selection_score)


def _new_eval_group_state(metric_keys):
    return {"sums": {key: 0.0 for key in metric_keys}, "samples": 0}


def _merge_eval_group_state(target, source):
    for key, value in source["sums"].items():
        target["sums"][key] += float(value)
    target["samples"] += int(source["samples"])


def _gather_eval_group_states(local_states, metric_keys):
    """Gather scalar group states so every DDP rank returns identical metrics."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_states)
    else:
        gathered = [local_states]
    merged = {}
    for rank_states in gathered:
        for group_name, source in rank_states.items():
            target = merged.setdefault(
                group_name, _new_eval_group_state(metric_keys)
            )
            _merge_eval_group_state(target, source)
    return merged

def _finalize_eval_group_state(state):
    denominator = max(int(state["samples"]), 1)
    return {key: float(value) / denominator for key, value in state["sums"].items()}


def _aggregate_equal_skeleton_metrics(states):
    """Average each skeleton/species metric with equal group weight."""
    nonempty = {
        skeleton_id: state
        for skeleton_id, state in states.items()
        if int(state["samples"]) > 0
    }
    if not nonempty:
        raise RuntimeError("Per-skeleton validation did not evaluate any motions")
    per_skeleton = {
        skeleton_id: _finalize_eval_group_state(state)
        for skeleton_id, state in nonempty.items()
    }
    metric_names = sorted(
        {key for metrics in per_skeleton.values() for key in metrics}
    )
    aggregated = {}
    for key in metric_names:
        values = [
            metrics[key] for metrics in per_skeleton.values() if key in metrics
        ]
        aggregated[key] = float(sum(values) / len(values))
    return aggregated


def _evaluate_motion_sample(
    batch,
    out,
    b,
    rel,
    input_frames,
    visualization_output_dir=None,
    visualization_motion_rel=None,
):
    """Compute the four TopoCap metrics for one unpadded motion."""
    frame_count = int(batch["frame_valid_mask"][b].sum())
    joint_count = int(batch["J_valid"][b])
    if not 1 <= frame_count <= input_frames:
        raise ValueError(f"Invalid validation length {frame_count} for {rel}")
    pred_rot = out["pred_rot6d"][b, :frame_count, :joint_count]
    gt_rot = batch["rot6d_a"][b, :frame_count, :joint_count]
    pred_mat = rot6d_to_rotmat_tensor(pred_rot.float()).cpu().numpy()
    gt_mat = rot6d_to_rotmat_tensor(gt_rot.float()).cpu().numpy()
    parents = batch["parent_a"][b, :joint_count].cpu().numpy()
    offsets = batch["offset_a"][b, :joint_count].cpu().numpy()
    pred_position = local_rotation_fk(pred_mat, offsets, parents)
    if "position" in batch:
        gt_position = batch["position"][b, :frame_count, :joint_count].float().cpu().numpy()
    else:
        gt_position = local_rotation_fk(gt_mat, offsets, parents)
    motion_metrics = topocap_protocol_metrics(
        pred_position,
        gt_position,
        batch["ref_position"][b, :joint_count].float().cpu().numpy(),
    )
    visualization_entry = None
    if visualization_output_dir is not None and rel == visualization_motion_rel:
        pose_head_position = (
            out["pred_position"][b, :frame_count, :joint_count].float().cpu().numpy()
            if "pred_position" in out else None
        )
        member_names = batch["rgb_member_names"][b][:frame_count]
        if any(name is None for name in member_names):
            raise RuntimeError(f"Valid RGB source is missing for {rel}")
        visualization_entry = {
            "rel": rel,
            "rot_sum": pred_mat.astype(np.float64),
            "gt_rot_sum": gt_mat.astype(np.float64),
            "count": np.ones((frame_count,), dtype=np.float64),
            "parents": parents,
            "joint_names": (
                [str(name) for name in batch["joint_names"][b][:joint_count]]
                if "joint_names" in batch
                else _generic_joint_names(joint_count)
            ),
            "offsets": offsets,
            "frame_sources": [
                (batch["rgb_tar_path"][b], str(name))
                for name in member_names
            ],
            "camera_root_position": batch[
                "camera_root_position"
            ][b, :frame_count].cpu().numpy(),
            "camera_intrinsics": batch[
                "camera_intrinsics"
            ][b].cpu().numpy(),
            "pose_head_position": pose_head_position,
        }
    return {
        "motion_metrics": motion_metrics,
        "visualization_entry": visualization_entry,
    }

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    cfg,
    visualization_output_dir=None,
    visualization_motion_rel=None,
    visualization_filename_prefix="",
):
    model.eval()
    # Ranks intentionally own unequal motion counts, so validation calls the
    # already-synchronized local module and avoids DDP forward collectives.
    eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    sums = {key: 0.0 for key in TOPOCAP_METRIC_KEYS}
    aggregation = resolve_eval_aggregation(cfg)
    per_skeleton_states = {}
    samples = 0
    amp_dtype = resolve_amp_dtype(cfg.get("train", {}))
    device_excluded_keys = (
        ORACLE_UNUSED_VISUAL_KEYS if uses_gt_pose_oracle(cfg) else frozenset()
    )
    eval_pin_memory = require_bool(
        cfg["eval"].get("pin_memory", True), "eval.pin_memory"
    )
    eval_pin_memory_chunk_bytes = require_positive_int(
        cfg["eval"].get("pin_memory_chunk_mb", 256),
        "eval.pin_memory_chunk_mb",
    ) * 1024 * 1024

    def consume_result(result, rel, skeleton_id):
        nonlocal samples
        for key, value in result["motion_metrics"].items():
            sums[key] += float(value)
        samples += 1
        if aggregation == "per_skeleton":
            skeleton_id = str(skeleton_id)
            if not skeleton_id:
                raise ValueError(f"Validation sample {rel!r} has an empty species id")
            state = per_skeleton_states.setdefault(
                skeleton_id, _new_eval_group_state(tuple(sums))
            )
            for key, value in result["motion_metrics"].items():
                state["sums"][key] += float(value)
            state["samples"] += 1
        entry = result.get("visualization_entry")
        if entry is not None:
            run_dir = os.path.join(visualization_output_dir, visualization_filename_prefix.rstrip("_") or "eval")
            target = motion_output_dir(run_dir, skeleton_id, rel)
            with video_output(target) as (video_path, _):
                _render_validation_motion(entry, video_path, cfg)


    for batch in loader:
        batch = move_batch_to_device(
            batch,
            device,
            excluded_keys=device_excluded_keys,
            pin_memory=eval_pin_memory,
            pin_memory_chunk_bytes=eval_pin_memory_chunk_bytes,
        )
        input_frames = int(batch["frame_valid_mask"].shape[1])
        if input_frames < 1:
            raise ValueError("Validation input must contain at least one frame")
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            out = eval_model(model_forward_inputs(batch, cfg))

        for b, rel in enumerate(batch["rel"]):
            result = _evaluate_motion_sample(
                batch,
                out,
                b,
                rel,
                input_frames,
                visualization_output_dir,
                visualization_motion_rel,
            )
            consume_result(result, rel, batch["species"][b])
    if aggregation == "per_skeleton":
        gathered_states = _gather_eval_group_states(
            per_skeleton_states, tuple(sums)
        )
        metrics = _aggregate_equal_skeleton_metrics(gathered_states)
        model.train()
        return metrics
    samples = reduce_sum_scalar(samples, device)
    for key in sums:
        sums[key] = reduce_sum_scalar(sums[key], device) / max(samples, 1)
    model.train()
    return sums


@torch.no_grad()


def write_validation_video_manifest(
    output_dir, expected_count, step, selected_motion_rel=None
):
    filename_prefix = f"step_{int(step):06d}__"
    videos = sorted(
        glob.glob(os.path.join(output_dir, filename_prefix.rstrip("_"), "*", "*", "visualization.mp4"))
    )
    if len(videos) != expected_count:
        raise RuntimeError(
            f"Step {step} validation videos incomplete: "
            f"expected {expected_count}, found {len(videos)}"
        )
    manifest = {
        "global_step": int(step),
        "video_count": len(videos),
        "videos": [os.path.relpath(path, output_dir) for path in videos],
    }
    if selected_motion_rel is not None:
        manifest["selected_motion_rel"] = str(selected_motion_rel)
    os.makedirs(output_dir, exist_ok=True)
    manifest_name = f"manifest_step_{int(step):06d}.json"
    temporary = os.path.join(
        output_dir, f"{manifest_name}.tmp.{os.getpid()}"
    )
    with open(temporary, "w") as file:
        json.dump(manifest, file, indent=2)
    os.replace(temporary, os.path.join(output_dir, manifest_name))


def distributed_accumulation_context(
    model, *, distributed, should_step, static_graph
):
    """Return a safe DDP context for one accumulation micro-step.

    PyTorch static-graph DDP can assert in Reducer::expect_autograd_hooks_ when
    its first backward passes are wrapped in ``no_sync()``, especially when the
    forward graph also uses reentrant activation checkpointing. Synchronizing
    every micro-step is mathematically equivalent after dividing the loss by
    the accumulation count, and the OVGF run communicates only about one
    million trainable parameters.
    """
    if distributed and not should_step and not static_graph:
        return model.no_sync()
    return contextlib.nullcontext()


def train(cfg):
    gradient_diagnostics = require_bool(
        cfg["train"].get("gradient_diagnostics", False),
        "train.gradient_diagnostics",
    )
    prefetch_batches = require_bool(
        cfg["train"].get("prefetch_batches", True), "train.prefetch_batches"
    )
    prefetch_depth = require_positive_int(
        cfg["train"].get("prefetch_depth", 1), "train.prefetch_depth"
    )
    configure_reproducible_runtime(cfg.get("runtime", {}))
    distributed, rank, world_size, local_rank = setup_distributed()
    graph_auto_pair_ratio = float(
        cfg["data"].get(
            "graph_attention_auto_group_max_pair_ratio",
            DEFAULT_AUTO_GROUP_MAX_PAIR_RATIO,
        )
    )
    set_auto_group_max_pair_ratio(graph_auto_pair_ratio)
    if is_main_process():
        print(
            "Graph attention auto grouped threshold: "
            f"pair_ratio <= {graph_auto_pair_ratio:g}",
            flush=True,
        )
    validate_skeleton_normalize_config(cfg)
    validate_gt_pose_oracle_config(cfg)
    validate_checkpoint_init_config(cfg)
    validate_observable_swing_config(cfg)
    required_caps = required_sample_capabilities(cfg)
    runtime_seed = int(cfg["runtime"].get("seed", 42))
    set_seed(runtime_seed)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seq_len = require_positive_int(cfg["train"]["seq_len"], "train.seq_len")
    batch_size = require_positive_int(cfg["train"]["batch_size"], "train.batch_size")
    max_train_steps = require_positive_int(
        cfg["train"]["max_train_steps"], "train.max_train_steps"
    )
    eval_save_every_steps = require_positive_int(
        cfg["train"]["eval_save_every_steps"],
        "train.eval_save_every_steps",
    )
    grad_accum_steps = require_positive_int(
        cfg["train"].get("gradient_accumulation_steps", 1),
        "train.gradient_accumulation_steps",
    )

    base_dir = os.path.join(cfg["output"]["checkpoint_root"], cfg["experiment"]["exp"])
    os.makedirs(base_dir, exist_ok=True)
    if is_main_process():
        resume_config_path = cfg["train"].get("resume_from_checkpoint")
        if resume_config_path:
            checkpoint_stem = os.path.splitext(
                os.path.basename(str(resume_config_path))
            )[0]
            config_filename = f"config_resume_{checkpoint_stem}.yaml"
        else:
            config_filename = "config.yaml"
        dump_yaml_config(cfg, os.path.join(base_dir, config_filename))

    max_joints = require_positive_int(
        cfg["data"].get("max_joints"), "data.max_joints"
    )
    dynamic_batch_max_joints = require_bool(
        cfg["data"].get("dynamic_batch_max_joints", False),
        "data.dynamic_batch_max_joints",
    )
    model_max_joints = require_positive_int(
        cfg["model"]["params"]["encoder_cfg"]["params"].get("num_joints"),
        "model.Encoder.num_joints",
    )
    if model_max_joints != max_joints:
        raise ValueError(
            "data.max_joints and model.Encoder.num_joints must match, "
            f"got {max_joints}/{model_max_joints}"
        )
    collate_single = functools.partial(
        collate_single_anyspecies_padded,
        max_joints=max_joints,
        dynamic_max_joints=dynamic_batch_max_joints,
    )

    eval_window = require_positive_int(
        cfg["eval"].get("window", 81),
        "eval.window",
    )
    cache_bundle = build_cache_source_bundle(
        cfg["data"],
        train_window=seq_len,
        eval_window=eval_window,
        required_capabilities=required_caps,
    )
    if len(cache_bundle.train) == 1:
        (single_name, single_dataset), = cache_bundle.train.items()
        train_stream = CacheStepIterableDataset(
            single_dataset,
            rank=rank,
            world_size=world_size,
            seed=runtime_seed,
            sample_transform=cache_bundle.train_sample_transform,
        )
    else:
        train_stream = MixedCacheStepIterableDataset(
            cache_bundle.train,
            cache_bundle.train_weights,
            rank=rank,
            world_size=world_size,
            seed=runtime_seed,
            sample_transform=cache_bundle.train_sample_transform,
        )
    val_datasets = cache_bundle.validation
    primary_val_name = cache_bundle.primary_validation_name
    val_dataset = val_datasets[primary_val_name]

    # Do not use DistributedSampler for validation: it repeats samples to make
    # the length divisible by world_size and would bias the reported metrics.
    eval_max_samples = cfg.get("eval", {}).get("max_samples")
    val_samplers = {
        name: (
            DistributedMotionEvalSampler(
                dataset, rank, world_size, max_samples=eval_max_samples
            )
            if distributed or eval_max_samples is not None
            else None
        )
        for name, dataset in val_datasets.items()
    }
    train_workers = require_int(
        cfg["train"].get("num_workers", 4), "train.num_workers"
    )
    if train_workers < 0:
        raise ValueError("train.num_workers must be >= 0")
    train_pin_memory = require_bool(
        cfg["train"].get("pin_memory", True), "train.pin_memory"
    )
    train_pin_memory_chunk_bytes = require_positive_int(
        cfg["train"].get("pin_memory_chunk_mb", 256),
        "train.pin_memory_chunk_mb",
    ) * 1024 * 1024
    # Pin only the fully merged model batch.  Pinning at DataLoader level
    # would pin every singleton and every prefetched sample before the main
    # process allocates another batch-sized pinned output buffer.
    loader_kwargs = dict(
        num_workers=train_workers,
        pin_memory=False,
    )
    if train_workers > 0:
        loader_kwargs.update(
            prefetch_factor=require_positive_int(
                cfg["train"].get("prefetch_factor", 2),
                "train.prefetch_factor",
            ),
            persistent_workers=require_bool(
                cfg["train"].get("persistent_workers", False),
                "train.persistent_workers",
            ),
        )
    # Worker seeds must depend only on the experiment seed/rank.  Without an
    # explicit generator PyTorch draws the loader base seed from the global
    # CPU RNG after model construction, so an optional branch can change the
    # sampled temporal windows even when every configured seed is identical.
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(runtime_seed + rank)
    train_loader = DataLoader(
        train_stream,
        # Automatic batching would make one worker decode all 32x81 frames.
        # Load singleton samples in parallel and batch them in the main process.
        batch_size=None,
        collate_fn=collate_single,
        worker_init_fn=worker_init_fn,
        generator=train_loader_generator,
        **loader_kwargs,
    )
    val_workers = require_int(
        cfg["eval"].get("num_workers", 2), "eval.num_workers"
    )
    if val_workers < 0:
        raise ValueError("eval.num_workers must be >= 0")
    eval_pin_memory = require_bool(
        cfg["eval"].get("pin_memory", True), "eval.pin_memory"
    )
    eval_pin_memory_chunk_bytes = require_positive_int(
        cfg["eval"].get("pin_memory_chunk_mb", 256),
        "eval.pin_memory_chunk_mb",
    ) * 1024 * 1024
    val_loader_kwargs = dict(
        num_workers=val_workers,
        pin_memory=False,
    )
    if val_workers > 0:
        val_loader_kwargs.update(
            persistent_workers=require_bool(
                cfg["eval"].get("persistent_workers", True),
                "eval.persistent_workers",
            ),
            prefetch_factor=require_positive_int(
                cfg["eval"].get("prefetch_factor", 2),
                "eval.prefetch_factor",
            ),
        )
    eval_batch_size = require_positive_int(
        cfg["eval"].get("batch_size", 1), "eval.batch_size"
    )
    val_loaders = {
        name: DataLoader(
            dataset,
            batch_size=None,
            sampler=val_samplers[name],
            shuffle=False,
            collate_fn=collate_single,
            **val_loader_kwargs,
        )
        for name, dataset in val_datasets.items()
    }
    val_loader = val_loaders[primary_val_name]
    model = instantiate_from_config(cfg["model"]).to(device)
    resume_path = cfg["train"].get("resume_from_checkpoint")
    init_path = cfg["train"].get("init_from_checkpoint")
    resume_checkpoint = None
    if resume_path:
        resume_path = os.path.abspath(os.path.expanduser(str(resume_path)))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_path}"
            )
        resume_checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        model.load_trainable_state_dict(resume_checkpoint["model_state"])
    elif init_path:
        init_path = os.path.abspath(os.path.expanduser(str(init_path)))
        if not os.path.isfile(init_path):
            raise FileNotFoundError(
                f"Initialization checkpoint does not exist: {init_path}"
            )
        init_checkpoint = torch.load(
            init_path, map_location="cpu", weights_only=False
        )
        allowed_missing_prefixes = _configured_prefixes(
            cfg["train"].get("init_allow_missing_prefixes"),
            "train.init_allow_missing_prefixes",
        )
        model.load_trainable_state_dict(
            init_checkpoint["model_state"],
            allowed_missing_prefixes=allowed_missing_prefixes,
        )
        del init_checkpoint
        if is_main_process():
            print(
                f"Initialized model weights from {init_path}; "
                f"allowed_missing_prefixes={allowed_missing_prefixes}",
                flush=True,
            )
    trainable_prefixes = configure_trainable_parameter_prefixes(
        model, cfg["train"]
    )
    ddp_static_graph = DDP_STATIC_GRAPH
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], static_graph=ddp_static_graph
        )
        if grad_accum_steps > 1 and ddp_static_graph and is_main_process():
            print(
                "Static-graph DDP: synchronizing every gradient-accumulation "
                "micro-step to avoid no_sync reducer assertions",
                flush=True,
            )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if is_main_process():
        trainable_count = sum(
            parameter.numel() for parameter in trainable_parameters
        )
        print(
            f"Trainable parameters: {trainable_count:,}; "
            f"prefixes={trainable_prefixes or 'all'}",
            flush=True,
        )
    base_trainable_parameters, residual_trainable_parameters = (
        partition_base_and_residual_parameters(model)
    )
    optimizer = optim.AdamW(trainable_parameters, lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    scheduler_cfg = dict(cfg["train"])
    scheduler_cfg["total_optimizer_steps"] = max_train_steps
    scheduler = build_scheduler(optimizer, scheduler_cfg)
    amp_dtype = resolve_amp_dtype(cfg["train"])
    # BF16 has FP32-like exponent range and does not need loss scaling. Using
    # GradScaler with FP16 caused hundreds of discarded updates and eventually
    # terminated the rotation ablations when the scale underflowed to zero.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )
    global_step = 0
    last_metrics = {}
    best_metric = float("inf")
    selection_score = float("inf")
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        if scaler.is_enabled() and "scaler_state" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        global_step = require_int(
            resume_checkpoint["global_step"], "checkpoint.global_step"
        )
        if not 0 <= global_step < max_train_steps:
            raise ValueError(
                f"Resume global_step must be in [0,{max_train_steps}), "
                f"got {global_step}"
            )
        last_metrics, best_metric, selection_score = _resume_validation_state(
            resume_checkpoint
        )
        if (
            is_main_process()
            and resume_checkpoint.get("selection_protocol") != CHECKPOINT_SELECTION_PROTOCOL
        ):
            print(
                "Reset best-checkpoint score for TopoCap normalization; training state restored.",
                flush=True,
            )
        if is_main_process():
            print(
                f"Resumed model, optimizer, scheduler, and step from "
                f"{resume_path}: global_step={global_step}, "
                f"lr={optimizer.param_groups[0]['lr']:.8g}",
                flush=True,
            )
    writer = SummaryWriter(os.path.join(base_dir, "logs")) if is_main_process() else None
    visual_cfg = cfg["eval"].get("visualization", {})
    visualization_enabled = require_bool(
        visual_cfg.get("enabled", False), "eval.visualization.enabled"
    )
    visualization_every_eval = require_bool(
        visual_cfg.get("every_eval", True),
        "eval.visualization.every_eval",
    )
    expected_val_videos = 0
    visualization_motion_rels = []
    visualization_sample_seed = require_int(
        visual_cfg.get("sample_seed", cfg["runtime"].get("seed", 42)),
        "eval.visualization.sample_seed",
    )
    if visualization_enabled and visualization_every_eval:
        groups = _dataset_motion_groups(val_dataset)
        visualization_motion_rels = _dataset_motion_rels(val_dataset)
        if len(visualization_motion_rels) != len(groups):
            raise RuntimeError(
                "Validation motion labels/groups have inconsistent lengths"
            )
        if not visualization_motion_rels:
            raise ValueError(
                "Validation visualization requires at least one motion"
            )
        expected_val_videos = 1
    model.train()
    apply_frozen_eval_module_prefixes(model, cfg["train"])
    optimizer.zero_grad(set_to_none=True)

    progress = (
        tqdm(
            total=max_train_steps,
            initial=global_step,
            desc="video2motion",
            ncols=110,
        )
        if is_main_process()
        else None
    )
    train_iterator = iter_worker_parallel_batches(
        iter(train_loader),
        batch_size,
        drop_last=True,
        pin_memory=train_pin_memory,
        pin_memory_max_bytes=train_pin_memory_chunk_bytes,
    )
    validation_index = len(
        glob.glob(
            os.path.join(
                base_dir,
                "val_visualizations",
                "manifest_step_*.json",
            )
        )
    )
    device_excluded_keys = ORACLE_UNUSED_VISUAL_KEYS if uses_gt_pose_oracle(cfg) else frozenset()
    def transfer_batch(cpu_batch):
        return move_batch_to_device(
            cpu_batch, device, excluded_keys=device_excluded_keys,
            pin_memory=train_pin_memory,
            pin_memory_chunk_bytes=train_pin_memory_chunk_bytes,
        )
    prefetcher = (
        DeviceBatchPrefetcher(train_iterator, transfer_batch, device, depth=prefetch_depth)
        if prefetch_batches else None
    )
    device_batches = prefetcher if prefetcher is not None else map(transfer_batch, train_iterator)
    try:
        while global_step < max_train_steps:
            model.train()
            apply_frozen_eval_module_prefixes(model, cfg["train"])
            # This is the LR used by the optimizer update being computed below.
            # scheduler.step() prepares the following update and must not change
            # what gets logged against the current global_step.
            optimizer_step_lr = float(optimizer.param_groups[0]["lr"])
            for accumulation_index in range(grad_accum_steps):
                batch = next(device_batches)
                apply_frame_masking(batch, cfg)
                should_step = accumulation_index == grad_accum_steps - 1
                sync_context = distributed_accumulation_context(
                    model,
                    distributed=distributed,
                    should_step=should_step,
                    static_graph=ddp_static_graph,
                )
                with sync_context:
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=device.type == "cuda",
                    ):
                        out = model(model_forward_inputs(batch, cfg))
                        loss, parts = compute_losses(out, batch, cfg)
                        if gradient_diagnostics and not torch.isfinite(loss):
                            nonfinite_parts = [
                                key
                                for key, value in parts.items()
                                if not torch.isfinite(value)
                            ]
                            report = _gradient_failure_report(
                                model=model,
                                model_out=out,
                                batch=batch,
                                parts=parts,
                                global_step=global_step + 1,
                            )
                            report["failure_reason"] = "nonfinite_forward_loss"
                            report["nonfinite_loss_parts"] = nonfinite_parts
                            bundle_dir = None
                            if is_main_process():
                                bundle_dir = _save_gradient_failure_bundle(
                                    base_dir=base_dir,
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    scaler=scaler,
                                    cfg=cfg,
                                    batch=batch,
                                    model_out=out,
                                    parts=parts,
                                    report=report,
                                    completed_global_step=global_step,
                                    attempted_global_step=global_step + 1,
                                )
                            raise FloatingPointError(
                                "Non-finite forward loss before backward at "
                                f"global_step={global_step + 1}; "
                                f"parts={nonfinite_parts}; "
                                f"complete crash bundle={bundle_dir}"
                            )
                        backward_loss = loss / grad_accum_steps
                    scaler.scale(backward_loss).backward()

            # The iterable stream never exhausts, so there is no epoch reset or
            # first-batch refill between optimizer updates.
            scaler.unscale_(optimizer)
            if gradient_diagnostics:
                nonfinite_gradients = []
                for parameter_name, parameter in model.named_parameters():
                    gradient = parameter.grad
                    if gradient is None or torch.isfinite(gradient).all():
                        continue
                    nonfinite_gradients.append(
                        (
                            parameter_name,
                            int((~torch.isfinite(gradient)).sum().item()),
                            gradient.numel(),
                        )
                    )
                    if len(nonfinite_gradients) >= 20:
                        break
                if nonfinite_gradients:
                    report = _gradient_failure_report(
                        model=model,
                        model_out=out,
                        batch=batch,
                        parts=parts,
                        global_step=global_step + 1,
                    )
                    report["failure_reason"] = "nonfinite_parameter_gradient"
                    report["first_nonfinite_parameters"] = nonfinite_gradients
                    bundle_dir = None
                    if is_main_process():
                        bundle_dir = _save_gradient_failure_bundle(
                            base_dir=base_dir,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            cfg=cfg,
                            batch=batch,
                            model_out=out,
                            parts=parts,
                            report=report,
                            completed_global_step=global_step,
                            attempted_global_step=global_step + 1,
                        )
                    raise FloatingPointError(
                        "Non-finite gradients before clipping/optimizer.step at "
                        f"global_step={global_step + 1}; "
                        f"first_parameters={nonfinite_gradients}; "
                        f"complete crash bundle={bundle_dir}"
                    )
            try:
                grad_clip_norm = float(
                    cfg["train"].get("grad_clip_norm", 1.0)
                )
                if not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
                    raise ValueError("train.grad_clip_norm must be finite and > 0")
                residual_grad_clip_norm = float(
                    cfg["train"].get("residual_grad_clip_norm", grad_clip_norm)
                )
                if (
                    not math.isfinite(residual_grad_clip_norm)
                    or residual_grad_clip_norm <= 0.0
                ):
                    raise ValueError(
                        "train.residual_grad_clip_norm must be finite and > 0"
                    )
                base_grad_norm = torch.nn.utils.clip_grad_norm_(
                    base_trainable_parameters,
                    grad_clip_norm,
                    error_if_nonfinite=gradient_diagnostics,
                )
                residual_grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        residual_trainable_parameters,
                        residual_grad_clip_norm,
                        error_if_nonfinite=gradient_diagnostics,
                    )
                    if residual_trainable_parameters
                    else base_grad_norm.new_zeros(())
                )
                if gradient_diagnostics:
                    grad_norm = torch.sqrt(
                        base_grad_norm.float().square()
                        + residual_grad_norm.float().square()
                    )
            except RuntimeError as error:
                if not gradient_diagnostics:
                    raise
                report = _gradient_failure_report(
                    model=model,
                    model_out=out,
                    batch=batch,
                    parts=parts,
                    global_step=global_step + 1,
                )
                report["failure_reason"] = "nonfinite_total_gradient_norm"
                bundle_dir = None
                if is_main_process():
                    bundle_dir = _save_gradient_failure_bundle(
                        base_dir=base_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        cfg=cfg,
                        batch=batch,
                        model_out=out,
                        parts=parts,
                        report=report,
                        completed_global_step=global_step,
                        attempted_global_step=global_step + 1,
                    )
                raise RuntimeError(
                    "Fatal gradient norm failure; parameters were not updated. "
                    f"Complete crash bundle: {bundle_dir}"
                ) from error
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after_step = scaler.get_scale()
            optimizer_step_succeeded = (
                not scaler.is_enabled()
                or (
                    math.isfinite(scale_before_step)
                    and math.isfinite(scale_after_step)
                    and scale_before_step > 0.0
                    and scale_after_step >= scale_before_step
                )
            )
            optimizer.zero_grad(set_to_none=True)
            if not optimizer_step_succeeded:
                if scaler.is_enabled() and scale_after_step <= 0.0:
                    raise FloatingPointError(
                        "AMP GradScaler reached zero after a non-finite backward "
                        "pass; aborting before any corrupted optimizer update"
                    )
                if is_main_process():
                    print(
                        f"AMP overflow: skipped optimizer update at "
                        f"global_step={global_step + 1}; LR schedule not advanced",
                        flush=True,
                    )
                continue
            scheduler.step()
            global_step += 1

            if is_main_process():
                progress.update(1)
                progress.set_postfix(
                    loss=f"{float(loss):.4f}",
                    lr=f"{optimizer_step_lr:.2e}",
                )
                if writer and global_step % 10 == 0:
                    writer.add_scalar("train/total", float(loss), global_step)
                    writer.add_scalar(
                        "train/lr", optimizer_step_lr, global_step
                    )
                    if gradient_diagnostics:
                        grad_norm_value = float(grad_norm)
                        base_grad_norm_value = float(base_grad_norm)
                        residual_grad_norm_value = float(residual_grad_norm)
                        writer.add_scalar(
                            "train/grad_norm_preclip",
                            grad_norm_value,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/grad_clip_scale",
                            min(
                                1.0,
                                grad_clip_norm / max(base_grad_norm_value, 1e-12),
                                residual_grad_clip_norm
                                / max(residual_grad_norm_value, 1e-12),
                            ),
                            global_step,
                        )
                        writer.add_scalar(
                            "train/base_grad_norm_preclip",
                            base_grad_norm_value,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/base_grad_clip_scale",
                            min(
                                1.0,
                                grad_clip_norm / max(base_grad_norm_value, 1e-12),
                            ),
                            global_step,
                        )
                        writer.add_scalar(
                            "train/residual_grad_norm_preclip",
                            residual_grad_norm_value,
                            global_step,
                        )
                        writer.add_scalar(
                            "train/residual_grad_clip_scale",
                            min(
                                1.0,
                                residual_grad_clip_norm
                                / max(residual_grad_norm_value, 1e-12),
                            ),
                            global_step,
                        )
                    for key, value in parts.items():
                        writer.add_scalar(
                            f"train/{key}", float(value), global_step
                        )
                        if key in cfg["train"]["weight"]:
                            writer.add_scalar(
                                f"train_weighted/{key}",
                                float(value)
                                * float(cfg["train"]["weight"][key]),
                                global_step,
                            )

            final_step = global_step == max_train_steps
            should_eval_save = (
                global_step % eval_save_every_steps == 0 or final_step
            )
            if not should_eval_save:
                continue

            metrics = last_metrics
            step_video_dir = None
            if should_eval_save:
                # Persist the completed optimizer update before running validation.
                # Evaluation includes metrics and video rendering and must never
                # make an otherwise valid training checkpoint unrecoverable.
                if is_main_process():
                    recovery_model = model.module if distributed else model
                    recovery_checkpoint = {
                        "model_state": recovery_model.trainable_state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "global_step": global_step,
                        "metrics": last_metrics,
                        "best_metric": best_metric,
                        "selection_score": selection_score,
                        "selection_protocol": CHECKPOINT_SELECTION_PROTOCOL,
                        "validation_index": validation_index,
                    }
                    torch.save(
                        recovery_checkpoint,
                        os.path.join(
                            base_dir,
                            f"{PROCESS_NAME}_ckpt_step{global_step:06d}.pt",
                        ),
                    )
                if distributed:
                    torch.distributed.barrier(device_ids=[local_rank])

                visualization_motion_rel = None
                if visualization_enabled and visualization_every_eval:
                    step_video_dir = os.path.join(
                        base_dir,
                        "val_visualizations",
                    )
                    visualization_motion_rel = (
                        select_validation_visualization_motion(
                            visualization_motion_rels,
                            validation_index,
                            visualization_sample_seed,
                        )
                    )
                validation_index += 1
                # Start validation workers only when validation is actually due.
                # Eagerly creating these iterators at training startup made train
                # and validation workers scan the same tar shards concurrently.
                if len(val_loaders) > 1:
                    metrics = {}
                    for split_name, split_loader in val_loaders.items():
                        split_iterator = iter_worker_parallel_batches(
                            split_loader,
                            eval_batch_size,
                            drop_last=False,
                            pin_memory=eval_pin_memory,
                            pin_memory_max_bytes=eval_pin_memory_chunk_bytes,
                        )
                        split_metrics = evaluate(
                            model,
                            split_iterator,
                            device,
                            cfg,
                        )
                        metrics.update(
                            {
                                f"{split_name}_{key}": value
                                for key, value in split_metrics.items()
                            }
                        )
                else:
                    val_iterator = iter_worker_parallel_batches(
                        val_loader,
                        eval_batch_size,
                        drop_last=False,
                        pin_memory=eval_pin_memory,
                        pin_memory_max_bytes=eval_pin_memory_chunk_bytes,
                    )
                    metrics = evaluate(
                        model,
                        val_iterator,
                        device,
                        cfg,
                        visualization_output_dir=step_video_dir,
                        visualization_motion_rel=visualization_motion_rel,
                        visualization_filename_prefix=f"step_{global_step:06d}__",
                    )
                selection_score = _checkpoint_selection_score(
                    metrics, cfg, primary_val_name, len(val_loaders) > 1
                )
                last_metrics = metrics

            if is_main_process():
                if should_eval_save:
                    print(
                        f"\nstep={global_step}/{max_train_steps} metrics={metrics}",
                        flush=True,
                    )
                    if writer:
                        for key, value in metrics.items():
                            writer.add_scalar(f"val/{key}", value, global_step)
                    if step_video_dir is not None:
                        write_validation_video_manifest(
                            step_video_dir,
                            expected_val_videos,
                            global_step,
                            selected_motion_rel=visualization_motion_rel,
                        )

                checkpoint_model = model.module if distributed else model
                checkpoint = {
                    "model_state": checkpoint_model.trainable_state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "global_step": global_step,
                    "metrics": metrics,
                    "selection_score": selection_score,
                    "selection_protocol": CHECKPOINT_SELECTION_PROTOCOL,
                    "best_metric": min(
                        best_metric,
                        selection_score,
                    ),
                    "validation_index": validation_index,
                }
                if selection_score < best_metric:
                    best_metric = selection_score
                    checkpoint["best_metric"] = best_metric
                    torch.save(
                        checkpoint,
                        os.path.join(base_dir, f"{PROCESS_NAME}_ckpt_best.pt"),
                    )
                torch.save(
                    checkpoint,
                    os.path.join(
                        base_dir,
                        f"{PROCESS_NAME}_ckpt_step{global_step:06d}.pt",
                    ),
                )
                step_paths = sorted(
                    glob.glob(
                        os.path.join(
                            base_dir, f"{PROCESS_NAME}_ckpt_step*.pt"
                        )
                    ),
                    key=os.path.getmtime,
                )
                keep = require_positive_int(
                    cfg["train"].get("max_ckpt", 100), "train.max_ckpt"
                )
                for path in step_paths[:-keep]:
                    os.remove(path)
            if distributed:
                torch.distributed.barrier()

    finally:
        if prefetcher is not None:
            prefetcher.close()

    if progress:
        progress.close()
    if writer:
        writer.close()
    cleanup_distributed()


def apply_cli_resume_override(cfg, resume_from_checkpoint):
    """CLI resume explicitly replaces and clears init_from_checkpoint.

    The static YAML may carry both, which config validation rejects.  The CLI
    override is an explicit operator choice, so it must not leave an init path
    behind that would make startup ambiguous.
    """
    if resume_from_checkpoint:
        cfg["train"]["resume_from_checkpoint"] = resume_from_checkpoint
        cfg["train"].pop("init_from_checkpoint", None)
    return cfg


def apply_cli_init_override(cfg, init_from_checkpoint):
    """Initialize model weights from a prior checkpoint via the CLI."""
    if init_from_checkpoint:
        cfg["train"]["init_from_checkpoint"] = init_from_checkpoint
        cfg["train"].pop("resume_from_checkpoint", None)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Deep-merge an additional YAML after --config. Repeat for ordered "
            "machine/local overrides; later files win."
        ),
    )
    parser.add_argument(
        "--experiment-exp",
        help="Override the experiment output name.",
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--init-from-checkpoint")
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        help="Deterministically limit validation motions for pilot runs.",
    )
    args = parser.parse_args()
    cfg = load_yaml_config(args.config, overrides=args.config_override)
    if args.experiment_exp:
        cfg["experiment"]["exp"] = args.experiment_exp
    if args.resume_from_checkpoint:
        cfg = apply_cli_resume_override(cfg, args.resume_from_checkpoint)
    elif args.init_from_checkpoint:
        cfg = apply_cli_init_override(cfg, args.init_from_checkpoint)
    if args.eval_max_samples is not None:
        cfg.setdefault("eval", {})["max_samples"] = require_positive_int(
            args.eval_max_samples, "--eval-max-samples"
        )
    train(cfg)


if __name__ == "__main__":
    main()
