"""Motion visualization for labeled validation and unlabeled RGB inference.

Share FK, skeleton panels, driven-mesh rendering, and per-motion output handling.
Unlabeled overlays use a weak-perspective fit without camera calibration.
"""

from __future__ import annotations

from contextlib import contextmanager
from urllib.parse import quote
import io
import json
import shutil
import subprocess
import sys
import tempfile
import os
import tarfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


def local_rotation_fk(
    local_rotation: np.ndarray,
    rest_offset: np.ndarray,
    parents: np.ndarray,
    fixed_root_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Build root-centered positions from root-global/child-local rotations.

    Joint zero contains the predicted camera/global root rotation, which
    ``fixed_root_rotation`` replaces when it is supplied.
    """
    local_rotation = np.asarray(local_rotation, dtype=np.float32)
    rest_offset = np.asarray(rest_offset, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    if local_rotation.ndim != 4 or local_rotation.shape[-2:] != (3, 3):
        raise ValueError(f"local_rotation must be [T,J,3,3], got {local_rotation.shape}")
    frames, joints = local_rotation.shape[:2]
    if rest_offset.shape != (joints, 3) or parents.shape != (joints,):
        raise ValueError("rest_offset/parents do not match rotation joint count")
    if fixed_root_rotation is not None:
        fixed_root_rotation = np.asarray(fixed_root_rotation, dtype=np.float32)
        if fixed_root_rotation.shape != (3, 3):
            raise ValueError("fixed_root_rotation must be [3,3]")
    if parents[0] != -1 or any(not (0 <= int(p) < j) for j, p in enumerate(parents[1:], 1)):
        raise ValueError("parents must be root-0, parent-before-child order")

    position = np.zeros((frames, joints, 3), dtype=np.float32)
    global_rotation = np.zeros((frames, joints, 3, 3), dtype=np.float32)
    global_rotation[:, 0] = (
        local_rotation[:, 0]
        if fixed_root_rotation is None
        else fixed_root_rotation
    )
    for joint in range(1, joints):
        parent = int(parents[joint])
        position[:, joint] = position[:, parent] + np.einsum(
            "tij,j->ti", global_rotation[:, parent], rest_offset[joint]
        )
        global_rotation[:, joint] = (
            global_rotation[:, parent] @ local_rotation[:, joint]
        )
    return position


def _view_coordinates(position: np.ndarray) -> np.ndarray:
    """Stable oblique orthographic view; preserve source Y-down convention."""
    x, y, z = position[..., 0], position[..., 1], position[..., 2]
    return np.stack((0.8660254 * x - 0.5 * z, y + 0.18 * z), axis=-1)


def _projection_bounds(gt_position: np.ndarray, pred_position: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.concatenate(
        (_view_coordinates(gt_position).reshape(-1, 2), _view_coordinates(pred_position).reshape(-1, 2)),
        axis=0,
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    half_range = max(float((maximum - minimum).max()) * 0.58, 1e-4)
    return center, half_range


def _draw_skeleton_panel(
    position: np.ndarray,
    parents: np.ndarray,
    center: np.ndarray,
    half_range: float,
    panel_size: int,
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    import cv2

    canvas = np.full((panel_size, panel_size, 3), 248, dtype=np.uint8)
    point = _view_coordinates(position)
    scale = (panel_size * 0.78) / (2.0 * half_range)
    pixel = np.empty_like(point)
    pixel[:, 0] = (point[:, 0] - center[0]) * scale + panel_size * 0.5
    pixel[:, 1] = (point[:, 1] - center[1]) * scale + panel_size * 0.54
    pixel = np.rint(pixel).astype(np.int32)
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        cv2.line(canvas, tuple(pixel[parent]), tuple(pixel[joint]), color, 2, cv2.LINE_AA)
    for joint, coordinate in enumerate(pixel):
        joint_color = (20, 20, 220) if joint == 0 else color
        cv2.circle(canvas, tuple(coordinate), 3 if joint == 0 else 2, joint_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "root-centered FK",
        (16, panel_size - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_rotation_comparison_video(
    frame_sources: Sequence[tuple[str, str]],
    gt_position: np.ndarray,
    pred_position: np.ndarray,
    parents: np.ndarray,
    output_path: str,
    fps: float,
    panel_size: int = 512,
) -> None:
    """Write ``original RGB | GT FK | predicted FK`` as an MP4."""
    import cv2

    frame_count = len(frame_sources)
    if frame_count == 0 or gt_position.shape[0] != frame_count or pred_position.shape[0] != frame_count:
        raise ValueError("RGB, GT, and prediction must have the same nonzero frame count")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    if panel_size < 64:
        raise ValueError(f"panel_size must be at least 64, got {panel_size}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    temporary_path = str(Path(output_path).with_suffix(f".tmp.{os.getpid()}.mp4"))
    writer = cv2.VideoWriter(
        temporary_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (panel_size * 3, panel_size),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the mp4v video writer")

    center, half_range = _projection_bounds(gt_position, pred_position)
    archives: dict[str, tarfile.TarFile] = {}
    try:
        for frame, (tar_path, member_name) in enumerate(frame_sources):
            archive = archives.get(tar_path)
            if archive is None:
                archive = tarfile.open(tar_path, "r")
                archives[tar_path] = archive
            file_obj = archive.extractfile(archive.getmember(member_name))
            if file_obj is None:
                raise FileNotFoundError(f"Cannot extract {tar_path}:{member_name}")
            with Image.open(io.BytesIO(file_obj.read())) as image:
                rgb = np.asarray(image.convert("RGB"))
            left = _letterbox_bgr(rgb, panel_size, "Original RGB", font_scale=0.8)
            middle = _draw_skeleton_panel(
                gt_position[frame], parents, center, half_range, panel_size, "GT skeleton", (40, 155, 40)
            )
            right = _draw_skeleton_panel(
                pred_position[frame], parents, center, half_range, panel_size, "Pred skeleton", (220, 115, 25)
            )
            writer.write(np.concatenate((left, middle, right), axis=1))
    finally:
        writer.release()
        for archive in archives.values():
            archive.close()
    if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) == 0:
        raise RuntimeError("Validation visualization video is empty")
    os.replace(temporary_path, output_path)


def rotation_matrices_from_fused_sum(rotation_sum: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Project overlap-averaged matrices back to proper SO(3) on CPU."""
    from utils.rotation import project_rotmat_to_so3

    if np.any(count == 0):
        raise ValueError(f"Incomplete visualization coverage: {np.flatnonzero(count == 0)[:10].tolist()}")
    matrix = torch.from_numpy(rotation_sum / count[:, None, None, None]).float()
    return project_rotmat_to_so3(matrix).numpy()


def component(value):
    value = str(value)
    if not value or value in {".", ".."}:
        raise ValueError(f"Invalid output component: {value!r}")
    result = quote(value, safe="-_.() ")
    # Windows forbids trailing spaces/dots; encode them without aliasing names.
    body = result.rstrip(" .")
    tail = result[len(body):].replace(" ", "%20").replace(".", "%2E")
    return body + tail


def motion_output_dir(output_root, skeleton_id, motion):
    """Keep validation-source roots, then exactly skeleton/animation folders."""
    skeleton_id = str(skeleton_id)
    motion = str(motion).replace("\\", "/")
    if motion.startswith(skeleton_id + "/"):
        motion = motion[len(skeleton_id) + 1:]
    target = Path(output_root) / component(skeleton_id) / component(motion)
    target.mkdir(parents=True, exist_ok=True)
    return target


@contextmanager
def video_output(target, filename="visualization.mp4"):
    """Publish one video; discard intermediate frames only after success.

    On failure the prior final video remains intact and the private work
    directory remains available for diagnosis. Changing panel counts replaces
    the same filename, so a three-column and four-column video never accumulate.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".render-", dir=target))
    staged = work / "visualization.mp4"
    yield staged, work / "mesh_frames"
    if not staged.is_file() or staged.stat().st_size == 0:
        raise RuntimeError(f"Renderer did not produce a nonempty video: {staged}")
    os.replace(staged, target / filename)
    shutil.rmtree(work)


def weak_perspective_pixels(
    position: np.ndarray,
    crop_box: tuple[int, int, int, int],
    fit_fraction: float = 0.82,
) -> np.ndarray:
    """Fit root-centered camera-axis XY to one image crop using one sequence transform."""
    position = np.asarray(position, dtype=np.float32)
    if position.ndim != 3 or position.shape[-1] != 3:
        raise ValueError(f"position must be [T,J,3], got {position.shape}")
    if not np.isfinite(position).all():
        raise ValueError("position contains NaN/Inf")
    left, top, right, bottom = (float(value) for value in crop_box)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop_box {crop_box}")
    fit_fraction = float(fit_fraction)
    if not np.isfinite(fit_fraction) or not 0 < fit_fraction <= 1:
        raise ValueError(f"fit_fraction must be in (0,1], got {fit_fraction}")

    xy = position[..., :2]
    low = xy.reshape(-1, 2).min(axis=0)
    high = xy.reshape(-1, 2).max(axis=0)
    center = (low + high) * 0.5
    extent = np.maximum(high - low, 1e-6)
    scale = fit_fraction * min((right - left) / extent[0], (bottom - top) / extent[1])
    target_center = np.asarray([(left + right) * 0.5, (top + bottom) * 0.5])
    return ((xy - center) * scale + target_center).astype(np.float32)


def write_unlabeled_motion_npz(
    output_path: str | Path,
    *,
    checkpoint_path: str | Path,
    skeleton_id: str,
    sequence: str,
    static: dict,
    fps: float,
    rotation_matrix: np.ndarray,
    fk_position: np.ndarray,
) -> None:
    """Save lossless in-place motion without fabricating camera calibration."""
    output_path = Path(output_path)
    rotation = np.asarray(rotation_matrix, dtype=np.float32)
    fk = np.asarray(fk_position, dtype=np.float32)
    parents = np.asarray(static["parents"], dtype=np.int32)
    if rotation.ndim != 4 or rotation.shape[1:] != (len(parents), 3, 3):
        raise ValueError(f"rotation_matrix has invalid shape {rotation.shape}")
    if fk.shape != rotation.shape[:2] + (3,):
        raise ValueError(f"fk_position has invalid shape {fk.shape}")
    if not np.isfinite(rotation).all() or not np.isfinite(fk).all():
        raise ValueError("Motion contains NaN/Inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = output_path.with_suffix(output_path.suffix + ".staged")
    with staged.open("wb") as file:
        np.savez_compressed(
            file,
            version=np.asarray(1, dtype=np.int32),
            checkpoint=np.asarray(str(checkpoint_path)),
            skeleton_id=np.asarray(str(skeleton_id)),
            source_fbx=np.asarray(str(static.get("source_fbx", ""))),
            source_cache=np.asarray(str(static.get("boundary_pruning", {}).get("source", ""))),
            sequence=np.asarray(str(sequence)),
            skeleton_hash=np.asarray(str(static["skeleton_hash"])),
            fps=np.asarray(float(fps), dtype=np.float32),
            frame_count=np.asarray(rotation.shape[0], dtype=np.int32),
            coordinate_system=np.asarray("opencv_right_handed_ydown_zforward_meters"),
            units=np.asarray("meters"),
            joint_names=np.asarray([str(name) for name in static["joint_names"]]),
            parents=parents,
            rest_translations=np.asarray(static["rest_translations"], dtype=np.float32),
            rest_rotations_quat_xyzw=np.asarray(
                static["rest_rotations_quat"], dtype=np.float32
            ),
            pred_local_rotation_matrix=rotation,
            pred_fk_position=fk,
            pred_root_translation=np.zeros((rotation.shape[0], 3), dtype=np.float32),
            has_predicted_root_translation=np.asarray(False),
            camera_intrinsics_available=np.asarray(False),
            projection_semantics=np.asarray(
                "root_centered_camera_axes; visualization_uses_orthographic_camera"
            ),
        )
    os.replace(staged, output_path)


def _letterbox_bgr(
    rgb: np.ndarray, panel_size: int, title: str, *, font_scale: float = 0.76,
) -> np.ndarray:
    import cv2

    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    scale = min(panel_size / width, panel_size / height)
    resized = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )[..., ::-1]
    canvas = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    top = (panel_size - resized.shape[0]) // 2
    left = (panel_size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    cv2.putText(
        canvas, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return canvas


def _overlay_skeleton(
    rgb: np.ndarray,
    pixel: np.ndarray,
    parents: np.ndarray,
    panel_size: int,
) -> np.ndarray:
    import cv2

    canvas = _letterbox_bgr(rgb, panel_size, "RGB + predicted skeleton")
    height, width = rgb.shape[:2]
    scale = min(panel_size / width, panel_size / height)
    left = (panel_size - round(width * scale)) // 2
    top = (panel_size - round(height * scale)) // 2
    panel_pixel = np.rint(pixel * scale + np.asarray([left, top])).astype(np.int32)
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        cv2.line(
            canvas, tuple(panel_pixel[parent]), tuple(panel_pixel[joint]),
            (35, 230, 255), 3, cv2.LINE_AA,
        )
    for joint, point in enumerate(panel_pixel):
        color = (35, 35, 245) if joint == 0 else (30, 245, 90)
        cv2.circle(canvas, tuple(point), 4 if joint == 0 else 3, color, -1, cv2.LINE_AA)
    cv2.putText(
        canvas, "weak-perspective fit (no K / no root xyz)",
        (16, panel_size - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (225, 225, 225), 1, cv2.LINE_AA,
    )
    return canvas


def visualization_panel_names(show_skeleton_overlay: bool = False) -> tuple[str, ...]:
    """Return visible columns; the uncalibrated skeleton overlay is opt-in."""
    if not isinstance(show_skeleton_overlay, (bool, np.bool_)):
        raise ValueError("show_skeleton_overlay must be boolean")
    if show_skeleton_overlay:
        return ("input_rgb", "rgb_predicted_skeleton", "driven_mesh")
    return ("input_rgb", "driven_mesh")


def render_textured_mesh_frames(
    *,
    mesh_path: str | Path,
    motion_path: str | Path,
    output_dir: str | Path,
    frame_count: int,
    panel_size: int = 512,
    allow_root_translation: bool = False,
    lighting_preset: str,
    lighting_args: list[str] | None = None,
) -> tuple[list[Path], dict]:
    """Render a prepared mesh asset, using its UV texture when available."""
    mesh_path = Path(mesh_path)
    motion_path = Path(motion_path)
    output_dir = Path(output_dir)
    frame_paths = [output_dir / f"frame_{index:04d}.png" for index in range(frame_count)]
    manifest_path = output_dir / "manifest.json"

    expected_view = (
        "root_motion_front_orthographic_camera_axes"
        if allow_root_translation
        else "root_centered_orthographic_camera_axes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("frame_*.png"):
        stale.unlink()
    repo_root = Path(__file__).resolve().parents[1]
    renderer_script = (
        "pyrender_render_mobj_motion.py"
        if lighting_preset == "mobj" else "blender_render_mesh_motion.py"
    )
    command = [
        sys.executable,
        str(repo_root / "utils" / renderer_script),
        "--mesh-path", str(mesh_path),
        "--motion", str(motion_path),
        "--output-dir", str(output_dir),
        "--resolution", str(panel_size),
        "--lighting-preset", lighting_preset,
        *(lighting_args or []),
    ]
    if allow_root_translation:
        command.append("--allow-root-translation")
    subprocess.run(command, cwd=repo_root, check=True)
    if not manifest_path.is_file():
        raise RuntimeError("Mesh renderer did not write manifest.json")
    render_report = json.loads(manifest_path.read_text(encoding="utf-8"))

    if render_report.get("view") != expected_view:
        raise RuntimeError(f"Unexpected unlabeled mesh view: {render_report.get('view')}")
    if render_report.get("lighting_preset") != lighting_preset:
        raise RuntimeError(
            f"Unexpected unlabeled lighting preset: {render_report.get('lighting_preset')}"
        )
    missing = [str(path) for path in frame_paths if not path.is_file() or path.stat().st_size <= 1024]
    if missing:
        raise RuntimeError(f"Mesh render is incomplete: {missing[:3]}")
    return frame_paths, render_report


def render_unlabeled_video(
    *,
    rgb_frames,
    fk_position: np.ndarray,
    static: dict,
    mesh_path: str | Path,
    motion_path: str | Path,
    mesh_frame_dir: str | Path,
    crop_box: tuple[int, int, int, int],
    output_path: str | Path,
    fps: float,
    panel_size: int = 512,
    show_skeleton_overlay: bool = False,
    allow_root_translation: bool = False,
    lighting_preset: str,
    lighting_args: list[str] | None = None,
    background: str = "white",
) -> dict:
    """Write ``RGB | fully textured driven mesh`` using rendered mesh frames."""
    import cv2

    if panel_size < 128:
        raise ValueError("panel_size must be at least 128")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    fk = np.asarray(fk_position, dtype=np.float32)
    parents = np.asarray(static["parents"], dtype=np.int64)
    panel_names = visualization_panel_names(show_skeleton_overlay)
    pixels = weak_perspective_pixels(fk, crop_box) if show_skeleton_overlay else None
    mesh_frames, mesh_render_report = render_textured_mesh_frames(
        mesh_path=mesh_path,
        motion_path=motion_path,
        output_dir=mesh_frame_dir,
        frame_count=len(fk),
        panel_size=panel_size,
        allow_root_translation=allow_root_translation,
        lighting_preset=lighting_preset,
        lighting_args=lighting_args,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".mp4v.{os.getpid()}.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (panel_size * len(panel_names), panel_size),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MP4 writer")
    frame_count = 0
    try:
        for frame_count, rgb in enumerate(rgb_frames, 1):
            index = frame_count - 1
            if index >= len(fk):
                raise RuntimeError("RGB stream contains more frames than prediction")
            panels = [_letterbox_bgr(rgb, panel_size, "Input RGB")]
            if show_skeleton_overlay:
                panels.append(_overlay_skeleton(rgb, pixels[index], parents, panel_size))
            mesh_panel = cv2.imread(str(mesh_frames[index]), cv2.IMREAD_UNCHANGED)
            if mesh_panel is None:
                raise RuntimeError(f"Cannot read mesh frame {mesh_frames[index]}")
            if mesh_panel.ndim == 3 and mesh_panel.shape[2] == 4:
                alpha = mesh_panel[:, :, 3:4].astype(np.float32) / 255.0
                background_value = 255 if background == "white" else 0
                mesh_panel = np.clip(
                    mesh_panel[:, :, :3].astype(np.float32) * alpha
                    + background_value * (1.0 - alpha), 0, 255,
                ).astype(np.uint8)
            if mesh_panel.shape[:2] != (panel_size, panel_size):
                mesh_panel = cv2.resize(mesh_panel, (panel_size, panel_size), interpolation=cv2.INTER_AREA)
            mesh_label = (
                "Driven mesh + root XYZ"
                if allow_root_translation else "Driven mesh"
            )
            cv2.putText(
                mesh_panel, mesh_label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            panels.append(mesh_panel)
            writer.write(np.concatenate(panels, axis=1))
    finally:
        writer.release()
    if frame_count != len(fk):
        raise RuntimeError(f"RGB/prediction frame mismatch: {frame_count} != {len(fk)}")
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("Temporary visualization video is empty")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        os.replace(temporary, output_path)
        codec = "mp4v"
    else:
        staged = output_path.with_suffix(f".h264.{os.getpid()}.mp4")
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(staged),
            ],
            check=True,
        )
        temporary.unlink()
        os.replace(staged, output_path)
        codec = "h264"
    return {
        "frame_count": frame_count,
        "fps": fps,
        "panel_size": panel_size,
        "codec": codec,
        "panels": list(panel_names),
        "skeleton_overlay_enabled": bool(show_skeleton_overlay),
        "projection": (
            "weak_perspective_sequence_fit_no_intrinsics"
            if show_skeleton_overlay else None
        ),
        "mesh_view": mesh_render_report["view"],
        "background": background,
        "textured_materials": bool(mesh_render_report.get("textured_materials")),
        "root_translation_enabled": bool(allow_root_translation),
        "mesh_render": mesh_render_report,
    }
