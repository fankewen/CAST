#!/usr/bin/env python3
"""Render GT/predicted skeleton joints as a shaded white proxy mesh video."""

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


# Colors are keyed by the child joint that terminates each rendered bone.
# Left/right suffixes are removed before lookup, so e.g. Hip_L and Hip_R use
# exactly the same color.  Keep this explicit: it makes colors stable across
# rigs, checkpoints, and rendering machines.
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


def visual_coordinates(joints):
    joints = np.asarray(joints, dtype=np.float32)[..., [0, 2, 1]].copy()
    joints[..., 0] *= -1
    return joints


def equal_limits(gt, pred):
    points = np.concatenate((gt.reshape(-1, 3), pred.reshape(-1, 3)))
    low, high = points.min(0), points.max(0)
    center = (low + high) * 0.5
    radius = max(float((high - low).max()) * 0.56, 1e-3)
    return [(float(c - radius), float(c + radius)) for c in center]


def bone_radius(name, parent_name, scale):
    label = f"{parent_name} {name}".lower()
    if any(token in label for token in ("root", "waist", "chest")):
        return 0.035 * scale
    if any(token in label for token in ("neck", "head", "hip")):
        return 0.025 * scale
    if "tail" in label:
        digits = [int(c) for c in name if c.isdigit()]
        taper = 1.0 - 0.09 * (digits[0] if digits else 0)
        return max(0.010, 0.024 * taper) * scale
    if any(token in label for token in ("wingfinger", "support", "toe", "chin")):
        return 0.006 * scale
    return 0.012 * scale


def cylinder_faces(start, end, radius, sides=8):
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return []
    direction /= length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    axis_a = np.cross(direction, helper)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(direction, axis_a)
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring_a = np.array([start + radius * (np.cos(a) * axis_a + np.sin(a) * axis_b) for a in angles])
    ring_b = ring_a + direction * length
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.extend(((ring_a[i], ring_a[j], ring_b[j]), (ring_a[i], ring_b[j], ring_b[i])))
    faces.append(tuple(ring_a))
    faces.append(tuple(ring_b[::-1]))
    return faces


def wing_faces(joints, names, side):
    lookup = {str(name).lower(): i for i, name in enumerate(names)}
    keys = [key for key in lookup if "wingfinger" in key and key.endswith(f"_{side.lower()}")]
    if not keys:
        return []
    keys.sort(key=lambda key: int("".join(c for c in key if c.isdigit()) or 0))
    wing_key = f"wing_{side.lower()}"
    if wing_key not in lookup:
        return []
    anchor = joints[lookup[wing_key]]
    tips = [joints[lookup[key]] for key in keys]
    return [(anchor, tips[i], tips[i + 1]) for i in range(len(tips) - 1)]


def draw_proxy(ax, joints, parents, names, scale):
    faces = []
    facecolors = []
    for joint, parent in enumerate(parents):
        if parent >= 0:
            bone_faces = cylinder_faces(
                joints[parent], joints[joint],
                bone_radius(str(names[joint]), str(names[parent]), scale),
            )
            faces.extend(bone_faces)
            facecolors.extend([bone_color(names[joint])] * len(bone_faces))
    body = Poly3DCollection(
        faces, facecolors=facecolors, edgecolors="#17202a", linewidth=0.18,
        shade=True, lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45),
    )
    ax.add_collection3d(body)
    membranes = wing_faces(joints, names, "L") + wing_faces(joints, names, "R")
    if membranes:
        wing = Poly3DCollection(
            membranes, facecolor=BONE_COLOR_BY_NAME["wing"], edgecolor="#4a1942",
            linewidth=0.25, alpha=0.72,
        )
        ax.add_collection3d(wing)


def setup_axis(ax, title, limits, elev, azim):
    ax.set_title(title, fontsize=17, pad=8)
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor("#f4f6f8")


def render(gt_path, pred_path, static_path, output_path, fps, elev, azim):
    static = np.load(static_path, allow_pickle=True).item()
    parents = np.asarray(static["parents"], dtype=np.int64)
    names = np.asarray(static["joint_names"])
    count = len(parents)
    gt = visual_coordinates(np.load(gt_path)[:, :count])
    pred = visual_coordinates(np.load(pred_path)[:, :count])
    frames = min(len(gt), len(pred))
    gt, pred = gt[:frames], pred[:frames]
    limits = equal_limits(gt, pred)
    extent = max(high - low for low, high in limits)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="#f4f6f8")
    writer = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(output_path), dpi=100):
        for frame in range(frames):
            fig.clear()
            gt_ax = fig.add_subplot(121, projection="3d")
            pred_ax = fig.add_subplot(122, projection="3d")
            setup_axis(gt_ax, "GT proxy mesh", limits, elev, azim)
            setup_axis(pred_ax, "Predict proxy mesh", limits, elev, azim)
            draw_proxy(gt_ax, gt[frame], parents, names, extent)
            draw_proxy(pred_ax, pred[frame], parents, names, extent)
            mpjpe = np.linalg.norm(gt[frame] - pred[frame], axis=-1).mean()
            fig.text(0.5, 0.035, f"Frame {frame + 1:02d}/{frames:02d}    MPJPE {mpjpe:.4f} m", ha="center", fontsize=12)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.08, wspace=0.01)
            writer.grab_frame()
    plt.close(fig)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--species", nargs="+", default=["em024_ganglong", "em007_jiaolong"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--elev", type=float, default=18)
    parser.add_argument("--azim", type=float, default=60)
    args = parser.parse_args()

    source = args.checkpoint_dir / f"vis_compare_epoch{args.epoch}"
    output = args.output_dir or args.checkpoint_dir / "vis_proxy_mesh_best_video"
    for species in args.species:
        prefix = f"test_val_{species}"
        paths = {
            "gt": source / f"{prefix}_gt.npy",
            "pred": source / f"{prefix}_pred.npy",
            "static": args.dataset_root / species / "static.npy",
        }
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        target = output / f"{prefix}_gt_left_pred_right_proxy_mesh.mp4"
        frames = render(paths["gt"], paths["pred"], paths["static"], target, args.fps, args.elev, args.azim)
        print(f"Saved {frames} frames: {target}")


if __name__ == "__main__":
    main()
