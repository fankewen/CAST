#!/usr/bin/env python3
"""Render aligned input RGB, GT proxy mesh, and predicted proxy mesh."""

import argparse
import io
import json
from pathlib import Path
import tarfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from PIL import Image

from utils.render_skeleton_proxy_mesh_compare import (
    bone_color,
    draw_proxy,
    equal_limits,
    visual_coordinates,
)


def load_rgb_frames(metadata_path, frame_count):
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    tar_path = metadata.get("rgb_tar_path")
    members = metadata.get("rgb_member_names")
    if not tar_path or not members:
        raise ValueError(f"Missing RGB source metadata in {metadata_path}")
    if len(members) < frame_count:
        raise ValueError(f"Only {len(members)} RGB frames for {frame_count} poses")

    frames = []
    with tarfile.open(tar_path, "r") as archive:
        for member in members[:frame_count]:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"{member} not found in {tar_path}")
            with Image.open(io.BytesIO(extracted.read())) as image:
                frames.append(np.asarray(image.convert("RGB")))
    return frames, metadata


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


def setup_pose_axis(ax, title, limits, elev, azim):
    ax.set_title(title, fontsize=16, color="white", pad=8)
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor("#20242a")


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
    elev=18,
    azim=60,
    clip="",
    camera_root_position=None,
    pred_camera_root_position=None,
    camera_intrinsics=None,
    auxiliary_pose_position=None,
    prediction_title="Decoder FK",
    mesh_rgb_frames=None,
    mesh_title="Driven mesh |RGB camera",
):
    """Render arrays with the repository's H.264 RGB/proxy comparison code."""
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
    camera_aligned = (
        camera_root_position is not None or camera_intrinsics is not None
    )
    if camera_aligned and (
        camera_root_position is None or camera_intrinsics is None
    ):
        raise ValueError(
            "camera_root_position and camera_intrinsics must be provided together"
        )
    gt = visual_coordinates(gt_source)
    pred = visual_coordinates(pred_source)
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
        else visual_coordinates(auxiliary_source[:frame_count])
    )
    rgb_frames = rgb_frames[:frame_count]
    mesh_rgb_frames = (
        None if mesh_rgb_frames is None else mesh_rgb_frames[:frame_count]
    )
    if camera_aligned:
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
    else:
        comparison = (
            pred
            if auxiliary is None
            else np.concatenate((pred, auxiliary), axis=0)
        )
        limits = equal_limits(gt, comparison)
        extent = max(high - low for low, high in limits)

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
            gt_ax = fig.add_subplot(
                grid[0, 1], **({} if camera_aligned else {"projection": "3d"})
            )
            pred_ax = fig.add_subplot(
                grid[0, 2], **({} if camera_aligned else {"projection": "3d"})
            )
            auxiliary_ax = (
                fig.add_subplot(
                    grid[0, 3],
                    **({} if camera_aligned else {"projection": "3d"}),
                )
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
            if camera_aligned:
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
            else:
                setup_pose_axis(gt_ax, "GT proxy mesh", limits, elev, azim)
                setup_pose_axis(pred_ax, prediction_title, limits, elev, azim)
                draw_proxy(gt_ax, gt[frame], parents, names, extent)
                draw_proxy(pred_ax, pred[frame], parents, names, extent)
                if auxiliary_ax is not None:
                    setup_pose_axis(
                        auxiliary_ax, "Encoder head", limits, elev, azim
                    )
                    draw_proxy(
                        auxiliary_ax,
                        auxiliary[frame],
                        parents,
                        names,
                        extent,
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
                    (
                        "    camera-aligned; predicted root xyz"
                        if pred_camera_root_position is not None
                        else "    camera-aligned; shared GT root xyz"
                    )
                    if camera_aligned else ""
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
    elev=18,
    azim=60,
    clip="",
    camera_root_position=None,
    pred_camera_root_position=None,
    camera_intrinsics=None,
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
        elev=elev,
        azim=azim,
        clip=clip,
        camera_root_position=camera_root_position,
        pred_camera_root_position=pred_camera_root_position,
        camera_intrinsics=camera_intrinsics,
        auxiliary_pose_position=auxiliary_pose_position,
        prediction_title=prediction_title,
        mesh_rgb_frames=mesh_rgb_frames,
        mesh_title=mesh_title,
    )


def render(gt_path, pred_path, metadata_path, static_path, output_path, fps, elev, azim):
    static = np.load(static_path, allow_pickle=True).item()
    parents = np.asarray(static["parents"], dtype=np.int64)
    names = np.asarray(static["joint_names"])
    joint_count = len(parents)
    gt = np.load(gt_path)[:, :joint_count]
    pred = np.load(pred_path)[:, :joint_count]
    frame_count = min(len(gt), len(pred))
    gt, pred = gt[:frame_count], pred[:frame_count]
    rgb_frames, metadata = load_rgb_frames(metadata_path, frame_count)
    frames = render_arrays(
        rgb_frames,
        gt,
        pred,
        parents,
        names,
        output_path,
        fps,
        elev=elev,
        azim=azim,
        clip=str(metadata.get("rel", metadata.get("species", ""))).split("/")[-1],
    )
    return frames, metadata.get("rel")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/skeleton_dataset"))
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--species", nargs="+", default=["em024_ganglong", "em007_jiaolong"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--elev", type=float, default=18)
    parser.add_argument("--azim", type=float, default=60)
    args = parser.parse_args()

    source = args.checkpoint_dir / f"vis_compare_epoch{args.epoch}"
    output = args.output_dir or args.checkpoint_dir / "vis_rgb_gt_pred_best_video"
    for species in args.species:
        prefix = f"test_val_{species}"
        paths = {
            "gt": source / f"{prefix}_gt.npy",
            "pred": source / f"{prefix}_pred.npy",
            "metadata": source / f"{prefix}_metadata.json",
            "static": args.dataset_root / species / "static.npy",
        }
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        target = output / f"{prefix}_rgb_gt_pred.mp4"
        frames, clip = render(
            paths["gt"], paths["pred"], paths["metadata"], paths["static"],
            target, args.fps, args.elev, args.azim,
        )
        print(f"Saved {frames} frames ({clip}): {target}")


if __name__ == "__main__":
    main()
