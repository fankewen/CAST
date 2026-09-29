"""Run Video2Motion on an unlabeled RGB video.

This adapter intentionally requires only RGB, ``static.npy``, a model config,
and a checkpoint.  Animation targets such as ``world_pos.npy`` and
``rot6d.npy`` are never opened or synthesized.
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.cache_reader import (
    CACHE_CAMERA_COORDINATE_SYSTEM,
    _matrix_to_rotation_6d_columns,
    _quat_xyzw_to_matrix,
    crop_rgb_by_mask,
)
from data.cache_formats import skeleton_id_from_static
from utils.config_validation import require_int, require_positive_int
from models.dinov3_infer import build_transform
from utils.model_inputs import normalize_model_skeleton_inputs
from utils.config_utils import instantiate_from_config, load_yaml_config
from utils.rotation import project_rotmat_to_so3, rot6d_to_rotmat_tensor, rotmat_to_rot6d_tensor

# Test/inference default: 81-frame windows with stride 20.
DEFAULT_WINDOW_OVERLAP = 61


def validate_dino_batch_size(batch_size):
    """Enforce the bounded-memory contract for CLI and programmatic callers."""
    return require_positive_int(batch_size, "--dino-batch-size")


def resolve_window_overlap(value, window):
    """Validate the user-facing overlap contract.

    ``-1`` means first-window only. Values ``0..window-1`` process the full
    native-frame sequence with stride ``window - overlap``.
    """
    window = require_positive_int(window, "window")
    overlap = require_int(value, "--window-overlap")
    if overlap < -1 or overlap >= window:
        raise ValueError(
            f"--window-overlap must be -1 or in [0, {window - 1}], got {overlap}"
        )
    return overlap


def inference_window_starts(frame_count, window, overlap):
    """Return regular starts, stopping after the first window covering the tail."""
    frame_count = require_positive_int(frame_count, "frame_count")
    window = require_positive_int(window, "window")
    overlap = resolve_window_overlap(overlap, window)
    if overlap == -1 or frame_count <= window:
        return [0]
    stride = window - overlap
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + window >= frame_count:
            return starts
        start += stride


def video_metadata(path):
    """Read native video metadata without changing FPS or frame count."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required for --input-video") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Input video has invalid FPS metadata: {source_fps}")
    if source_frames <= 0:
        raise ValueError(f"Input video has invalid frame count: {source_frames}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Input video has invalid dimensions: {(width, height)}")
    return {
        "source_fps": source_fps,
        "source_frame_count": source_frames,
        "width": width,
        "height": height,
        "output_fps": source_fps,
        "output_frame_count": source_frames,
        "resampled": False,
    }


def iter_video_rgb(path, frame_count):
    """Decode exactly the requested native frames in source order."""
    import cv2

    frame_count = require_positive_int(frame_count, "frame_count")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {path}")
    produced = 0
    try:
        while produced < frame_count:
            ok, bgr = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            produced += 1
    finally:
        capture.release()
    if produced != frame_count:
        raise RuntimeError(
            f"Decoded native frame count mismatch: produced={produced}, expected={frame_count}"
        )

def center_crop_box(image_size, crop_scale):
    """Return a centered RGB crop without requiring segmentation metadata."""
    width, height = (int(image_size[0]), int(image_size[1]))
    crop_scale = float(crop_scale)
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {(width, height)}")
    if not np.isfinite(crop_scale) or not 0 < crop_scale <= 1:
        raise ValueError(f"--center-crop-scale must be in (0, 1], got {crop_scale}")
    crop_width = max(1, int(round(width * crop_scale)))
    crop_height = max(1, int(round(height * crop_scale)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def iter_center_crops(
    rgb_frames,
    size=256,
    crop_scale=0.62,
):
    """Yield deterministic center crops for RGB-only test videos.

    This is deliberately a preprocessing choice, not a recovered camera model.
    It is suitable for centered subjects and keeps unlabeled inference free of
    synthetic masks, GT joints, and invented camera intrinsics.
    """
    from PIL import Image

    transform = build_transform(size)
    for rgb in rgb_frames:
        rgb_image = Image.fromarray(rgb)
        crop_box = center_crop_box(rgb_image.size, crop_scale)
        transformed = transform(rgb_image.crop(crop_box))
        yield transformed


def iter_mask_bbox_crops(
    rgb_frames,
    mask_frames,
    size=256,
    bbox_margin=5,
):
    """Yield normalized RGB crops from synchronized foreground masks."""
    from PIL import Image

    transform = build_transform(size)
    rgb_iter = iter(rgb_frames)
    mask_iter = iter(mask_frames)
    while True:
        try:
            rgb = next(rgb_iter)
        except StopIteration:
            try:
                next(mask_iter)
            except StopIteration:
                return
            raise RuntimeError("Mask video has more native frames than RGB video")
        try:
            mask = next(mask_iter)
        except StopIteration as exc:
            raise RuntimeError("Mask video has fewer native frames than RGB video") from exc
        rgb_image = Image.fromarray(rgb)
        mask_image = Image.fromarray(mask)
        crop = crop_rgb_by_mask(rgb_image, mask_image, bbox_margin)
        transformed = transform(crop)
        yield transformed


def iter_feature_windows(features, frame_count, window, overlap):
    """Yield native-frame windows under the explicit overlap contract."""
    frame_count = require_positive_int(frame_count, "frame_count")
    window = require_positive_int(window, "window")
    starts = inference_window_starts(frame_count, window, overlap)
    buffer = deque()
    start_index = 0
    current_start = starts[0]
    observed = 0

    def stack(values):
        if isinstance(values[0], tuple):
            return tuple(
                torch.stack([value[index] for value in values], dim=0)
                for index in range(len(values[0]))
            )
        return torch.stack(values, dim=0)

    def blank_like(value):
        if isinstance(value, tuple):
            return tuple(torch.zeros_like(part) for part in value)
        return torch.zeros_like(value)

    for index, feature in enumerate(features):
        observed = index + 1
        if index < current_start:
            continue
        buffer.append(feature)
        valid = min(window, frame_count - current_start)
        if len(buffer) != valid:
            continue
        values = list(buffer)
        values.extend(blank_like(values[-1]) for _ in range(window - valid))
        yield current_start, valid, stack(values)
        start_index += 1
        if start_index >= len(starts):
            break
        next_start = starts[start_index]
        discard = next_start - current_start
        for _ in range(discard):
            buffer.popleft()
        current_start = next_start

    if observed != frame_count:
        raise RuntimeError(f"Feature count mismatch: {observed} != {frame_count}")
    if start_index != len(starts):
        raise RuntimeError(
            f"Window generation incomplete: yielded={start_index}, expected={len(starts)}"
        )

def graph_metadata(parents):
    joint_count = len(parents)
    graph_hop = np.full((joint_count, joint_count), 5, dtype=np.int64)
    graph_edge = np.full((joint_count, joint_count), 4, dtype=np.int64)
    np.fill_diagonal(graph_edge, 0)
    adjacency = [[] for _ in range(joint_count)]
    ancestor = np.zeros((joint_count, joint_count), dtype=np.bool_)
    for joint, parent in enumerate(parents):
        ancestor[joint, joint] = True
        current = int(parent)
        while current >= 0:
            ancestor[joint, current] = True
            current = int(parents[current])
        if parent >= 0:
            adjacency[joint].append(int(parent))
            adjacency[int(parent)].append(joint)
            graph_edge[int(parent), joint] = 1
            graph_edge[joint, int(parent)] = 2
    for source in range(joint_count):
        graph_hop[source, source] = 0
        queue = [source]
        while queue:
            current = queue.pop(0)
            if graph_hop[source, current] >= 5:
                continue
            for target in adjacency[current]:
                if graph_hop[source, target] > graph_hop[source, current] + 1:
                    graph_hop[source, target] = graph_hop[source, current] + 1
                    queue.append(target)
    return graph_hop, graph_edge, ancestor


def static_conditioning(
    static_path,
    image_window,
    device,
    max_joints: int,
    valid_length=None,
):
    static = np.load(static_path, allow_pickle=True).item()
    if static.get("coordinate_system") != CACHE_CAMERA_COORDINATE_SYSTEM or static.get("units") != "meters":
        raise ValueError("static.npy must use OpenCV camera coordinates in meters")
    metric_scale = np.asarray(static.get("metric_scale"), dtype=np.float32)
    if metric_scale.ndim != 0 or not np.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError("static.npy must contain a finite positive metric_scale")
    joint_count = int(static["joints"])
    if joint_count > max_joints:
        raise ValueError(f"Skeleton has {joint_count} joints, maximum is {max_joints}")
    parents = np.asarray(static["parents"], dtype=np.int64)
    if parents[0] != -1 or any(not (0 <= int(p) < j) for j, p in enumerate(parents[1:], 1)):
        raise ValueError("Skeleton must use root-0 parent-before-child order")

    local_t = np.asarray(static["rest_translations"], dtype=np.float32)
    local_r = _quat_xyzw_to_matrix(np.asarray(static["rest_rotations_quat"], dtype=np.float32))
    reset_position = np.zeros((joint_count, 3), dtype=np.float32)
    global_rotation = np.zeros((joint_count, 3, 3), dtype=np.float32)
    for joint, parent in enumerate(parents):
        if parent < 0:
            reset_position[joint] = local_t[joint]
            global_rotation[joint] = local_r[joint]
        else:
            reset_position[joint] = reset_position[parent] + global_rotation[parent] @ local_t[joint]
            global_rotation[joint] = global_rotation[parent] @ local_r[joint]
    reset_position -= reset_position[0:1]
    reset_rot6d = _matrix_to_rotation_6d_columns(local_r)
    graph_hop, graph_edge, ancestor = graph_metadata(parents)

    def pad(array, value=0):
        shape = (max_joints,) + array.shape[1:]
        result = np.full(shape, value, dtype=array.dtype)
        result[:joint_count] = array
        return result

    parent_pad = np.full((max_joints,), -1, dtype=np.int64)
    parent_pad[:joint_count] = parents
    square_pad = lambda x: np.pad(x, ((0, max_joints-joint_count), (0, max_joints-joint_count)))
    joint_mask = np.zeros((max_joints,), dtype=np.bool_)
    joint_mask[:joint_count] = True
    if valid_length is None:
        valid_length = image_window.shape[0]
    if not 1 <= int(valid_length) <= image_window.shape[0]:
        raise ValueError(
            f"valid_length must be in [1, {image_window.shape[0]}], got {valid_length}"
        )
    frame_valid_mask = torch.arange(image_window.shape[0]) < int(valid_length)
    batch = {
        "ref_position": torch.from_numpy(pad(reset_position))[None],
        "joint_mask": torch.from_numpy(joint_mask)[None],
        "frame_valid_mask": frame_valid_mask[None],
        "ancestor_mask": torch.from_numpy(square_pad(ancestor))[None],
        "graph_hop": torch.from_numpy(square_pad(graph_hop))[None],
        "graph_edge": torch.from_numpy(square_pad(graph_edge))[None],
        "static_rot_joint_mask": torch.zeros(1, max_joints, dtype=torch.bool),
        "metric_scale": torch.tensor([float(metric_scale)], dtype=torch.float32),
        "ref_rot6d_a": torch.from_numpy(pad(reset_rot6d))[None],
        "offset_a": torch.from_numpy(pad(local_t))[None],
        "parent_a": torch.from_numpy(parent_pad)[None],
        "image_rgb": image_window[None],
    }
    return {key: value.to(device) for key, value in batch.items()}, joint_count



@torch.no_grad()
def run(args):
    cfg = load_yaml_config(args.config)
    device = torch.device(args.device)
    window = require_positive_int(cfg["train"]["seq_len"], "train.seq_len")
    overlap = resolve_window_overlap(args.window_overlap, window)
    max_joints = require_positive_int(cfg["data"].get("max_joints"), "data.max_joints")
    metadata = video_metadata(args.input_video)
    source_frame_count = int(metadata["source_frame_count"])
    frame_count = min(source_frame_count, window) if overlap == -1 else source_frame_count
    metadata["output_frame_count"] = frame_count
    metadata["processed_frame_count"] = frame_count
    metadata["window_overlap"] = overlap
    metadata["window_starts"] = inference_window_starts(frame_count, window, overlap)
    rgb_stream = iter_video_rgb(args.input_video, frame_count)
    image_size = int(cfg["data"].get("image_size", 256))
    if args.input_mask_video is not None:
        mask_metadata = video_metadata(args.input_mask_video)
        if mask_metadata["source_frame_count"] < frame_count:
            raise ValueError(
                "Mask video has fewer native frames than required: "
                f"{mask_metadata['source_frame_count']} < {frame_count}"
            )
        mask_stream = iter_video_rgb(args.input_mask_video, frame_count)
        features = iter_mask_bbox_crops(
            rgb_stream,
            mask_stream,
            size=image_size,
            bbox_margin=int(cfg["data"].get("bbox_margin", 5)),
        )
        metadata["preprocessing"] = {
            "mode": "provided_mask_bbox",
            "mask_video": str(args.input_mask_video),
        }
    else:
        features = iter_center_crops(
            rgb_stream,
            size=image_size,
            crop_scale=args.center_crop_scale,
        )
        metadata["preprocessing"] = {
            "mode": "center_crop_rgb_only",
            "center_crop_scale": float(args.center_crop_scale),
            "camera_intrinsics_used": False,
        }
    model = instantiate_from_config(cfg["model"]).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(checkpoint["model_state"])

    rot_sum = None
    count = np.zeros((frame_count,), dtype=np.float64)
    joint_count = None
    for start, valid, window_data in iter_feature_windows(
        features, frame_count, window, overlap
    ):
        image_window = window_data
        batch, joint_count = static_conditioning(
            args.static,
            image_window,
            device,
            valid_length=valid,
            max_joints=max_joints,
        )
        if args.input_stride > 1:
            observed = torch.arange(
                image_window.shape[0], device=device, dtype=torch.long
            ).remainder(args.input_stride).eq(0)
            observed[0] = True
            observed &= batch["frame_valid_mask"][0].bool()
            batch["frame_observed_mask"] = observed[None]
        output = model(
            normalize_model_skeleton_inputs(batch),
            return_reconstructed_positions=False,
        )
        rot = output["pred_rot6d"][0, :valid, :joint_count].float().cpu().numpy()
        if rot_sum is None:
            rot_sum = np.zeros((frame_count, joint_count, 3, 3), dtype=np.float64)
        indices = np.arange(start, start + valid)
        rot_sum[indices] += rot6d_to_rotmat_tensor(torch.from_numpy(rot)).numpy()
        count[indices] += 1
    if np.any(count == 0):
        raise RuntimeError("Sliding windows did not cover every input frame")
    mean_rotmat = torch.from_numpy(rot_sum / count[:, None, None, None]).float()
    rot = rotmat_to_rot6d_tensor(project_rotmat_to_so3(mean_rotmat)).numpy()
    from utils.motion_visualization import motion_output_dir
    prefix = Path(args.input_video).stem
    static = np.load(args.static, allow_pickle=True).item()
    args.output_dir = motion_output_dir(args.output_dir, skeleton_id_from_static(static), prefix)
    np.save(os.path.join(args.output_dir, f"{prefix}_root_global_rot6d.npy"), rot[:, 0])
    np.save(os.path.join(args.output_dir, f"{prefix}_nonroot_local_rot6d.npy"), rot[:, 1:])
    timestamps = np.arange(frame_count, dtype=np.float64) / metadata["output_fps"]
    np.save(os.path.join(args.output_dir, f"{prefix}_timestamps.npy"), timestamps)
    with open(os.path.join(args.output_dir, f"{prefix}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"Saved unlabeled video prediction: {frame_count} frames, {joint_count} joints, "
        f"{metadata['output_fps']:.6g} FPS"
    )


def main():
    parser = argparse.ArgumentParser(description="Video2Motion inference from an unlabeled RGB video")
    parser.add_argument("--input-video", required=True)
    parser.add_argument(
        "--input-mask-video",
        default=None,
        help=(
            "Optional foreground mask synchronized with --input-video. If omitted, "
            "RGB-only inference uses a deterministic center crop."
        ),
    )
    parser.add_argument(
        "--center-crop-scale",
        type=float,
        default=0.62,
        help="Centered RGB crop fraction used only when --input-mask-video is omitted",
    )
    parser.add_argument("--static", required=True, help="Path to the rig's static.npy")
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=DEFAULT_WINDOW_OVERLAP,
        help=(
            "-1 processes/saves only the first window; 0..80 processes the full "
            "native-frame video with stride 81-overlap. Default: 61 "
            "(stride 20 for the default 81-frame window)."
        ),
    )
    parser.add_argument(
        "--input-stride",
        type=int,
        default=1,
        help="Observe one input frame every N frames; hidden frames use the learned mask token",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.input_stride < 1:
        raise ValueError(f"--input-stride must be positive, got {args.input_stride}")
    run(args)


if __name__ == "__main__":
    main()
