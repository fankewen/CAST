"""Offline mask video with Meta SAM 2.1 (whole-clip video predictor).

Offline inference has no latency budget, so this path uses the SAM 2 video
predictor: one automatically estimated first-frame box is propagated through the
complete clip, which keeps the mask temporally consistent.

No frame-rate conversion is performed; a lossless mask video is written.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch


def estimate_temporal_subject_box(
    small_rgb_frames: np.ndarray,
    original_size: tuple[int, int],
    *,
    fallback_box_fraction=(0.16, 0.12, 0.84, 0.72),
) -> tuple[int, int, int, int]:
    """Estimate a coarse prompt box from temporal change in a static-background clip."""
    import cv2

    frames = np.asarray(small_rgb_frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 2:
        raise ValueError(f"small_rgb_frames must be [T,H,W,3] with T>=2, got {frames.shape}")
    width, height = (int(original_size[0]), int(original_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"original_size must be positive, got {original_size}")
    median = np.median(frames.astype(np.float32), axis=0)
    difference = np.linalg.norm(frames.astype(np.float32) - median[None], axis=-1)
    motion = np.percentile(difference, 85, axis=0).astype(np.float32)
    normalized = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(float(otsu), float(np.percentile(normalized, 88)))
    mask = (normalized >= threshold).astype(np.uint8) * 255
    small_height, small_width = mask.shape
    # These generated clips place the subject in the central/upper field; the
    # lower strip is a reflected background and should not seed SAM.
    region = np.zeros_like(mask)
    region[
        int(round(0.05 * small_height)) : int(round(0.82 * small_height)),
        int(round(0.05 * small_width)) : int(round(0.95 * small_width)),
    ] = 255
    mask = cv2.bitwise_and(mask, region)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    selected = []
    image_center = np.asarray([0.5 * small_width, 0.42 * small_height])
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < max(12, 0.0005 * small_width * small_height):
            continue
        distance = np.linalg.norm((centroids[index] - image_center) / [small_width, small_height])
        if distance <= 0.48:
            selected.append((x, y, x + w, y + h))
    if selected:
        x0 = min(box[0] for box in selected)
        y0 = min(box[1] for box in selected)
        x1 = max(box[2] for box in selected)
        y1 = max(box[3] for box in selected)
        span_x = x1 - x0
        span_y = y1 - y0
        x0 -= int(round(0.18 * span_x))
        x1 += int(round(0.18 * span_x))
        y0 -= int(round(0.22 * span_y))
        y1 += int(round(0.22 * span_y))
        x0 = np.clip(x0, 0, small_width - 1)
        y0 = np.clip(y0, 0, small_height - 1)
        x1 = np.clip(x1, x0 + 1, small_width)
        y1 = np.clip(y1, y0 + 1, small_height)
        area_fraction = (x1 - x0) * (y1 - y0) / (small_width * small_height)
        if 0.01 <= area_fraction <= 0.65:
            scale_x = width / small_width
            scale_y = height / small_height
            return (
                int(round(x0 * scale_x)),
                int(round(y0 * scale_y)),
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
            )
    fx0, fy0, fx1, fy1 = fallback_box_fraction
    return (
        int(round(fx0 * width)), int(round(fy0 * height)),
        int(round(fx1 * width)), int(round(fy1 * height)),
    )



def estimate_temporal_positive_point(
    small_rgb_frames: np.ndarray,
    original_size: tuple[int, int],
    box: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Choose a moving, dark first-frame point inside the coarse subject box."""
    import cv2

    frames = np.asarray(small_rgb_frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 2:
        raise ValueError("small_rgb_frames must be [T,H,W,3] with T>=2")
    width, height = (int(original_size[0]), int(original_size[1]))
    small_height, small_width = frames.shape[1:3]
    x0, y0, x1, y1 = box
    sx0 = int(np.clip(round(x0 * small_width / width), 0, small_width - 1))
    sy0 = int(np.clip(round(y0 * small_height / height), 0, small_height - 1))
    sx1 = int(np.clip(round(x1 * small_width / width), sx0 + 1, small_width))
    sy1 = int(np.clip(round(y1 * small_height / height), sy0 + 1, small_height))
    median = np.median(frames.astype(np.float32), axis=0)
    difference = np.linalg.norm(frames.astype(np.float32) - median[None], axis=-1)
    motion = np.percentile(difference, 85, axis=0).astype(np.float32)
    motion = cv2.GaussianBlur(motion, (0, 0), 2.0)
    motion /= max(float(motion.max()), 1e-6)

    # Learn the background palette from the outer image border. This handles
    # bright clouds and dark blue sky without assuming that the subject is
    # dark: the positive point should be both moving and chromatically unlike
    # the border background.
    lab = cv2.cvtColor(frames[0], cv2.COLOR_RGB2LAB).astype(np.float32)
    border_width = max(4, int(round(min(small_width, small_height) * 0.09)))
    border = np.concatenate(
        (
            lab[:border_width].reshape(-1, 3),
            lab[-border_width:].reshape(-1, 3),
            lab[:, :border_width].reshape(-1, 3),
            lab[:, -border_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    sample_step = max(1, len(border) // 4000)
    sample = border[::sample_step].astype(np.float32)
    unique = np.unique(sample.astype(np.uint8), axis=0).astype(np.float32)
    cluster_count = min(4, len(unique))
    if cluster_count == 1:
        centers = unique
    else:
        cv2.setRNGSeed(42)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.2,
        )
        _, _, centers = cv2.kmeans(
            sample, cluster_count, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
    background_distance = np.min(
        np.linalg.norm(lab[:, :, None, :] - centers[None, None, :, :], axis=-1),
        axis=-1,
    )
    background_distance = cv2.GaussianBlur(background_distance, (0, 0), 1.2)
    distance_scale = max(float(np.percentile(background_distance, 99)), 1e-6)
    background_distance = np.clip(background_distance / distance_scale, 0.0, 1.0)
    yy, xx = np.mgrid[0:small_height, 0:small_width]
    center_prior = np.exp(
        -((xx - 0.5 * small_width) / (0.38 * small_width)) ** 2
        -((yy - 0.46 * small_height) / (0.38 * small_height)) ** 2
    )
    score = (
        motion
        * (0.05 + 0.95 * background_distance)
        * (0.25 + 0.75 * center_prior)
    )
    allowed = np.zeros_like(score, dtype=bool)
    allowed[sy0:sy1, sx0:sx1] = True
    score = np.where(allowed, score, -1.0)
    py, px = np.unravel_index(int(np.argmax(score)), score.shape)
    if score[py, px] <= 0:
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
    return (
        float((px + 0.5) * width / small_width),
        float((py + 0.5) * height / small_height),
    )

def extract_native_jpegs_and_prompt_box(
    video_path: str | Path,
    frame_dir: str | Path,
    sample_size: int = 256,
) -> tuple[dict, tuple[int, int, int, int]]:
    import cv2

    video_path = Path(video_path)
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0 or expected <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Input video has invalid native metadata")
    small_frames = []
    frame_count = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            output = frame_dir / f"{frame_count:05d}.jpg"
            if not cv2.imwrite(str(output), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {output}")
            scale = min(sample_size / width, sample_size / height)
            small = cv2.resize(
                bgr,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            small_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            frame_count += 1
    finally:
        capture.release()
    if frame_count != expected:
        raise RuntimeError(f"Decoded frame count mismatch: {frame_count} != {expected}")
    small_stack = np.stack(small_frames)
    box = estimate_temporal_subject_box(small_stack, (width, height))
    positive_point = estimate_temporal_positive_point(
        small_stack, (width, height), box
    )
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }, box, positive_point



def select_temporally_consistent_components(
    masks: list[np.ndarray],
    seed_point_xy: tuple[float, float],
) -> list[np.ndarray]:
    """Remove disconnected background regions while retaining the tracked subject."""
    import cv2

    if not masks:
        raise ValueError("No masks to postprocess")
    height, width = np.asarray(masks[0]).shape
    seed_x = int(np.clip(round(seed_point_xy[0]), 0, width - 1))
    seed_y = int(np.clip(round(seed_point_xy[1]), 0, height - 1))
    diagonal = float(np.hypot(width, height))
    cleaned: list[np.ndarray] = []
    previous = None
    previous_area = None
    previous_centroid = np.asarray([seed_x, seed_y], dtype=np.float64)
    dilation_size = max(9, int(round(min(width, height) * 0.04)) | 1)
    dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)
    close_kernel = np.ones((5, 5), np.uint8)

    for frame_index, raw_mask in enumerate(masks):
        binary = np.asarray(raw_mask, dtype=bool)
        if binary.shape != (height, width):
            raise ValueError("Masks do not share one image size")
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        candidates = [
            index for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= max(16, 0.0002 * width * height)
        ]
        if not candidates:
            raise RuntimeError(f"SAM mask has no valid component at frame {frame_index}")
        if frame_index == 0 and labels[seed_y, seed_x] in candidates:
            selected = int(labels[seed_y, seed_x])
        else:
            previous_dilated = (
                None if previous is None else cv2.dilate(
                    previous.astype(np.uint8), dilation_kernel, iterations=1
                ).astype(bool)
            )
            best_score = None
            selected = None
            for index in candidates:
                component = labels == index
                area = float(stats[index, cv2.CC_STAT_AREA])
                distance = np.linalg.norm(centroids[index] - previous_centroid) / diagonal
                intersection = (
                    0.0 if previous_dilated is None
                    else float(np.count_nonzero(component & previous_dilated))
                )
                candidate_overlap = intersection / area
                previous_overlap = (
                    0.0 if previous_area is None else intersection / previous_area
                )
                area_continuity = (
                    0.0 if previous_area is None
                    else min(area / previous_area, previous_area / area)
                )
                score = (
                    2.5 * previous_overlap
                    + 1.5 * candidate_overlap
                    + 1.5 * area_continuity
                    - 0.75 * distance
                    + 0.01 * np.log1p(area)
                )
                if previous_area is not None and area < 0.15 * previous_area:
                    score -= 3.0
                if best_score is None or score > best_score:
                    best_score = score
                    selected = index
            assert selected is not None
        component = (labels == selected).astype(np.uint8) * 255
        component = cv2.morphologyEx(
            component, cv2.MORPH_CLOSE, close_kernel, iterations=1
        ) > 0
        cleaned.append(component)
        previous = component
        previous_area = float(np.count_nonzero(component))
        ys, xs = np.nonzero(component)
        previous_centroid = np.asarray([xs.mean(), ys.mean()], dtype=np.float64)
    return cleaned

def write_lossless_mask_video(
    masks: list[np.ndarray],
    output_path: str | Path,
    fps: float,
) -> None:
    import cv2

    if not masks:
        raise ValueError("No masks to write")
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mkv":
        raise ValueError("Lossless SAM mask output must use .mkv")
    height, width = np.asarray(masks[0]).shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = output_path.with_suffix(f".staged.{output_path.suffix.lstrip('.')}")
    writer = cv2.VideoWriter(
        str(staged), cv2.VideoWriter_fourcc(*"FFV1"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the lossless FFV1 writer")
    try:
        for mask in masks:
            binary = np.asarray(mask, dtype=bool)
            if binary.shape != (height, width):
                raise ValueError("SAM masks do not share one image size")
            gray = binary.astype(np.uint8) * 255
            writer.write(np.repeat(gray[:, :, None], 3, axis=2))
    finally:
        writer.release()
    if not staged.is_file() or staged.stat().st_size == 0:
        raise RuntimeError("Lossless mask video is empty")
    staged.replace(output_path)


def generate_sam2_mask_video(
    *,
    input_video: str | Path,
    output_mask_video: str | Path,
    checkpoint: str | Path,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda:0",
    prompt_box: tuple[int, int, int, int] | None = None,
    keep_frames: bool = False,
) -> dict:
    """Track one subject mask through every native frame with SAM 2.1."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "Official facebookresearch/sam2 is required; install it with `pip install -e .`"
        ) from exc

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_mask_video = Path(output_mask_video)
    output_mask_video.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f"{Path(input_video).stem}_sam2_", dir=output_mask_video.parent)
    )
    frame_dir = temporary_root / "frames"
    try:
        metadata, automatic_box, automatic_positive_point = (
            extract_native_jpegs_and_prompt_box(input_video, frame_dir)
        )
        box = automatic_box if prompt_box is None else tuple(int(value) for value in prompt_box)
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= metadata["width"] and 0 <= y0 < y1 <= metadata["height"]):
            raise ValueError(f"Prompt box is outside the video: {box}")
        predictor = build_sam2_video_predictor(
            model_cfg, str(checkpoint), device=device, vos_optimized=False
        )
        inference_state = predictor.init_state(
            video_path=str(frame_dir), offload_video_to_cpu=True,
            offload_state_to_cpu=False, async_loading_frames=False,
        )
        masks_by_frame: dict[int, np.ndarray] = {}
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if str(device).startswith("cuda") else nullcontext()
        )
        box_width = x1 - x0
        box_height = y1 - y0
        prompt_points = np.asarray(
            [
                [automatic_positive_point[0], automatic_positive_point[1]],
                [x0 + 0.04 * box_width, (y0 + y1) * 0.5],
                [x1 - 0.04 * box_width, (y0 + y1) * 0.5],
                [(x0 + x1) * 0.5, y1 - 0.03 * box_height],
            ],
            dtype=np.float32,
        )
        prompt_labels = np.asarray([1, 0, 0, 0], dtype=np.int32)
        with torch.inference_mode(), autocast:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=prompt_points,
                labels=prompt_labels,
                box=np.asarray(box, dtype=np.float32),
            )
            for frame_index, object_ids, mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                matches = [i for i, object_id in enumerate(object_ids) if int(object_id) == 1]
                if len(matches) != 1:
                    raise RuntimeError(f"SAM 2 lost object id 1 at frame {frame_index}")
                mask = (mask_logits[matches[0]] > 0.0).detach().cpu().numpy().squeeze()
                masks_by_frame[int(frame_index)] = mask.astype(bool)
        missing = [i for i in range(metadata["frame_count"]) if i not in masks_by_frame]
        if missing:
            raise RuntimeError(f"SAM 2 did not return masks for frames {missing[:10]}")
        masks = [masks_by_frame[i] for i in range(metadata["frame_count"])]
        masks = select_temporally_consistent_components(
            masks, tuple(prompt_points[0])
        )
        area_fraction = np.asarray([mask.mean() for mask in masks], dtype=np.float64)
        if np.any(area_fraction <= 0.0005) or np.any(area_fraction >= 0.8):
            raise RuntimeError(
                "SAM 2 produced implausible mask areas: "
                f"min={area_fraction.min():.6f}, max={area_fraction.max():.6f}"
            )
        write_lossless_mask_video(masks, output_mask_video, metadata["fps"])
        report = {
            "schema": "sam2_video_mask",
            "input_video": str(Path(input_video)),
            "output_mask_video": str(output_mask_video),
            "model": "SAM 2.1 Hiera Large",
            "model_cfg": model_cfg,
            "checkpoint": str(checkpoint),
            "prompt": "automatic_temporal_motion_box_plus_background_points_on_frame_0",
            "prompt_box_xyxy": list(box),
            "prompt_points_xy": prompt_points.tolist(),
            "prompt_point_labels": prompt_labels.tolist(),
            "mask_postprocessing": (
                "positive_seeded_temporally_consistent_connected_component"
            ),
            "native_video": metadata,
            "mask_area_fraction": {
                "min": float(area_fraction.min()),
                "median": float(np.median(area_fraction)),
                "max": float(area_fraction.max()),
            },
        }
        output_mask_video.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    finally:
        if keep_frames:
            retained = output_mask_video.parent / f"{output_mask_video.stem}_frames"
            if retained.exists():
                shutil.rmtree(retained)
            shutil.move(str(frame_dir), str(retained))
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
        else:
            shutil.rmtree(temporary_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-mask-video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box", type=int, nargs=4)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    report = generate_sam2_mask_video(
        input_video=args.input_video,
        output_mask_video=args.output_mask_video,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
        prompt_box=None if args.box is None else tuple(args.box),
        keep_frames=args.keep_frames,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
