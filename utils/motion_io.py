"""Motion output paths and lossless motion-file export."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import numpy as np


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
