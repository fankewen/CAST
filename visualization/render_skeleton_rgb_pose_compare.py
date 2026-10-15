"""Render GT and predicted skeletons projected onto their source RGB frames."""

import io
from pathlib import Path
import re
import tarfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from PIL import Image

# Colors are keyed by the child joint that terminates each rendered bone.
# Left/right suffixes are removed before lookup, so e.g. Hip_L and Hip_R use
# exactly the same color. Keep colors stable across rigs and checkpoints.
BONE_COLOR_BY_NAME = {
    "root": "#ff595e",
    "waist": "#ff924c",
    "chest": "#ffca3a",
    "neck0": "#c5e063",
    "neck1": "#8ac926",
    "neck2": "#52b788",
    "neck3": "#20c997",
    "neck4": "#00b4d8",
    "neck5": "#0096c7",
    "head": "#4cc9f0",
    "chin": "#90e0ef",
    "hip": "#6a4c93",
    "knee": "#845ec2",
    "ankle": "#b39ddb",
    "foot": "#d0bfff",
    "toe": "#e0aaff",
    "shoulder": "#1982c4",
    "upperarm": "#277da1",
    "forearm": "#43aa8b",
    "hand": "#90be6d",
    "palm": "#b5e48c",
    "wing": "#f72585",
    "wingshoulder": "#b5179e",
    "wingupper": "#7209b7",
    "wingfore": "#560bad",
    "wingsupport": "#3a0ca3",
    "wingsupport0": "#4361ee",
    "wingsupport1": "#4895ef",
    "wingfinger0": "#f15bb5",
    "wingfinger1": "#fee440",
    "wingfinger2": "#00f5d4",
    "wingfinger3": "#00bbf9",
    "wingfinger4": "#9b5de5",
    "wingfinger5": "#f9844a",
    "wingfinger6": "#f94144",
    "tail0": "#ff6b6b",
    "tail1": "#ff8e72",
    "tail2": "#f9c74f",
    "tail3": "#90be6d",
    "tail4": "#43aa8b",
    "tail5": "#4d908e",
    "tail6": "#577590",
    "tail7": "#7b2cbf",
}

FALLBACK_BONE_COLORS = (
    "#ef476f", "#ffd166", "#06d6a0", "#118ab2", "#9b5de5",
    "#f15bb5", "#00bbf9", "#f8961e",
)


def canonical_bone_name(name):
    """Normalize a joint label while deliberately merging L/R counterparts."""
    return re.sub(r"(?:_(?:l|r|left|right)|(?:left|right))$", "", str(name).lower())


def bone_color(name):
    canonical = canonical_bone_name(name)
    configured = BONE_COLOR_BY_NAME.get(canonical)
    if configured is not None:
        return configured
    # Unknown rigs still receive a stable non-white color without relying on
    # Python's randomized hash().
    index = sum((position + 1) * ord(char) for position, char in enumerate(canonical))
    return FALLBACK_BONE_COLORS[index % len(FALLBACK_BONE_COLORS)]


def load_rgb_frame_sources(frame_sources):
    frames = []
    archives = {}
    try:
        for tar_path, member in frame_sources:
            archive = archives.get(tar_path)
            if archive is None:
                archive = tarfile.open(tar_path, "r")
                archives[tar_path] = archive
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"{member} not found in {tar_path}")
            with Image.open(io.BytesIO(extracted.read())) as image:
                frames.append(np.asarray(image.convert("RGB")))
    finally:
        for archive in archives.values():
            archive.close()
    return frames


def project_opencv_camera(position, intrinsics):
    """Project OpenCV camera xyz to source-image pixel coordinates."""
    position = np.asarray(position, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if position.ndim != 3 or position.shape[-1] != 3:
        raise ValueError(
            f"camera position must be [T,J,3], got {position.shape}"
        )
    if intrinsics.shape == (4,):
        intrinsics = np.broadcast_to(
            intrinsics[None], (position.shape[0], 4)
        )
    if intrinsics.shape != (position.shape[0], 4):
        raise ValueError(
            "camera intrinsics must be [4] or [T,4] containing fx/fy/cx/cy"
        )
    if not np.isfinite(position).all() or not np.isfinite(intrinsics).all():
        raise ValueError("camera position/intrinsics must be finite")
    depth = position[..., 2]
    safe_depth = np.where(np.abs(depth) > 1e-8, depth, np.nan)
    pixel = np.empty(position.shape[:-1] + (2,), dtype=np.float32)
    pixel[..., 0] = (
        intrinsics[:, None, 2]
        + intrinsics[:, None, 0] * position[..., 0] / safe_depth
    )
    pixel[..., 1] = (
        intrinsics[:, None, 3]
        + intrinsics[:, None, 1] * position[..., 1] / safe_depth
    )
    return pixel, depth


def _draw_camera_skeleton(ax, rgb, pixel, depth, parents, names, title):
    """Overlay a projected colored skeleton on its corresponding RGB."""
    ax.imshow(rgb)
    ax.set_title(title, fontsize=16, color="white", pad=8)
    ax.set_axis_off()
    valid = (
        np.isfinite(pixel).all(axis=-1)
        & np.isfinite(depth)
        & (depth > 1e-6)
    )
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        if valid[parent] and valid[joint]:
            ax.plot(
                pixel[[parent, joint], 0],
                pixel[[parent, joint], 1],
                color=bone_color(names[joint]),
                linewidth=3.0,
                solid_capstyle="round",
                path_effects=[],
            )
    if valid.any():
        ax.scatter(
            pixel[valid, 0],
            pixel[valid, 1],
            s=13,
            c=[bone_color(names[index]) for index in np.flatnonzero(valid)],
            edgecolors="#111820",
            linewidths=0.45,
            zorder=3,
        )
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)


def render_arrays(
    rgb_frames,
    gt_position,
    pred_position,
    parents,
    names,
    output_path,
    fps,
    *,
    camera_root_position,
    camera_intrinsics,
    clip="",
    pred_camera_root_position=None,
    auxiliary_pose_position=None,
    prediction_title="Decoder FK",
    mesh_rgb_frames=None,
    mesh_title="Driven mesh |RGB camera",
):
    """Render camera-projected skeleton comparisons as an H.264 video."""
    parents = np.asarray(parents, dtype=np.int64)
    names = np.asarray(names)
    joint_count = len(parents)
    gt_source = np.asarray(gt_position)[:, :joint_count]
    pred_source = np.asarray(pred_position)[:, :joint_count]
    auxiliary_source = (
        None
        if auxiliary_pose_position is None
        else np.asarray(auxiliary_pose_position)[:, :joint_count]
    )
    gt = np.asarray(gt_source, dtype=np.float32)
    pred = np.asarray(pred_source, dtype=np.float32)
    frame_count = min(len(rgb_frames), len(gt), len(pred))
    if auxiliary_source is not None:
        frame_count = min(frame_count, len(auxiliary_source))
    if mesh_rgb_frames is not None:
        frame_count = min(frame_count, len(mesh_rgb_frames))
    if frame_count <= 0:
        raise ValueError("RGB, GT, and prediction must contain at least one frame")
    gt, pred = gt[:frame_count], pred[:frame_count]
    auxiliary = (
        None
        if auxiliary_source is None
        else np.asarray(auxiliary_source[:frame_count], dtype=np.float32)
    )
    rgb_frames = rgb_frames[:frame_count]
    mesh_rgb_frames = (
        None if mesh_rgb_frames is None else mesh_rgb_frames[:frame_count]
    )
    gt_root = np.asarray(camera_root_position, dtype=np.float32)[
        :frame_count
    ]
    if gt_root.shape != (frame_count, 3):
        raise ValueError(
            f"camera_root_position must be [T,3], got {gt_root.shape}"
        )
    pred_root = (
        gt_root
        if pred_camera_root_position is None
        else np.asarray(pred_camera_root_position, dtype=np.float32)[
            :frame_count
        ]
    )
    if pred_root.shape != (frame_count, 3):
        raise ValueError(
            "pred_camera_root_position must be [T,3], got "
            f"{pred_root.shape}"
        )
    gt_camera = gt_source[:frame_count] + gt_root[:, None, :]
    pred_camera = pred_source[:frame_count] + pred_root[:, None, :]
    gt_pixel, gt_depth = project_opencv_camera(
        gt_camera, camera_intrinsics
    )
    pred_pixel, pred_depth = project_opencv_camera(
        pred_camera, camera_intrinsics
    )
    if auxiliary_source is not None:
        auxiliary_camera = (
            auxiliary_source[:frame_count] + gt_root[:, None, :]
        )
        auxiliary_pixel, auxiliary_depth = project_opencv_camera(
            auxiliary_camera, camera_intrinsics
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_count = 3 + int(auxiliary is not None) + int(mesh_rgb_frames is not None)
    fig = plt.figure(
        figsize=(6.4 * panel_count, 6.4), dpi=100, facecolor="#20242a"
    )
    writer = FFMpegWriter(
        fps=fps,
        bitrate=9000,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    with writer.saving(fig, str(output_path), dpi=100):
        for frame in range(frame_count):
            fig.clear()
            grid = fig.add_gridspec(
                1,
                panel_count,
                left=0.01,
                right=0.99,
                top=0.92,
                bottom=0.09,
                wspace=0.015,
            )
            rgb_ax = fig.add_subplot(grid[0, 0])
            gt_ax = fig.add_subplot(grid[0, 1])
            pred_ax = fig.add_subplot(grid[0, 2])
            auxiliary_ax = (
                fig.add_subplot(grid[0, 3])
                if auxiliary is not None
                else None
            )
            mesh_column = 3 + int(auxiliary is not None)
            mesh_ax = (
                fig.add_subplot(grid[0, mesh_column])
                if mesh_rgb_frames is not None
                else None
            )

            rgb_ax.imshow(rgb_frames[frame])
            rgb_ax.set_title("Input RGB", fontsize=16, color="white", pad=8)
            rgb_ax.set_axis_off()
            _draw_camera_skeleton(
                gt_ax,
                rgb_frames[frame],
                gt_pixel[frame],
                gt_depth[frame],
                parents,
                names,
                "GT skeleton |RGB camera",
            )
            _draw_camera_skeleton(
                pred_ax,
                rgb_frames[frame],
                pred_pixel[frame],
                pred_depth[frame],
                parents,
                names,
                f"{prediction_title} |RGB camera",
            )
            if auxiliary_ax is not None:
                _draw_camera_skeleton(
                    auxiliary_ax,
                    rgb_frames[frame],
                    auxiliary_pixel[frame],
                    auxiliary_depth[frame],
                    parents,
                    names,
                    "Encoder head |RGB camera",
                )
            if mesh_ax is not None:
                mesh_ax.imshow(mesh_rgb_frames[frame])
                mesh_ax.set_title(mesh_title, fontsize=16, color="white", pad=8)
                mesh_ax.set_axis_off()

            fk_mpjpe = np.linalg.norm(
                gt[frame] - pred[frame], axis=-1
            ).mean()
            pose_head_text = ""
            if auxiliary is not None:
                pose_head_mpjpe = np.linalg.norm(
                    gt[frame] - auxiliary[frame], axis=-1
                ).mean()
                pose_head_text = f"    Pose-head MPJPE {pose_head_mpjpe:.4f} m"
            fig.text(
                0.5,
                0.035,
                f"{clip}    Frame {frame + 1:02d}/{frame_count:02d}    "
                f"FK MPJPE {fk_mpjpe:.4f} m"
                + pose_head_text
                + (
                    "    camera-aligned; predicted root xyz"
                    if pred_camera_root_position is not None
                    else "    camera-aligned; shared GT root xyz"
                ),
                ha="center",
                color="white",
                fontsize=12,
            )
            writer.grab_frame()
    plt.close(fig)
    return frame_count


def render_frame_sources(
    frame_sources,
    gt_position,
    pred_position,
    parents,
    names,
    output_path,
    fps,
    *,
    camera_root_position,
    camera_intrinsics,
    clip="",
    pred_camera_root_position=None,
    auxiliary_pose_position=None,
    prediction_title="Decoder FK",
    mesh_rgb_frames=None,
    mesh_title="Driven mesh |RGB camera",
):
    """Render tar-backed frames through the same H.264 implementation."""
    return render_arrays(
        load_rgb_frame_sources(frame_sources),
        gt_position,
        pred_position,
        parents,
        names,
        output_path,
        fps,
        clip=clip,
        camera_root_position=camera_root_position,
        pred_camera_root_position=pred_camera_root_position,
        camera_intrinsics=camera_intrinsics,
        auxiliary_pose_position=auxiliary_pose_position,
        prediction_title=prediction_title,
        mesh_rgb_frames=mesh_rgb_frames,
        mesh_title=mesh_title,
    )
