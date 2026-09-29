"""Bind-aware mesh retargeting and linear-blend skinning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

COORDINATE_SYSTEM = "opencv_right_handed_ydown_zforward_meters"
CV_TO_UE_AXES = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
CV_TO_BLENDER_CAMERA_AXES = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def opencv_camera_points_to_blender(points: np.ndarray) -> np.ndarray:
    """Map OpenCV camera XYZ to Blender camera-local XYZ.

    Both coordinate systems are right-handed. OpenCV looks along +Z with +Y
    down, while an unrotated Blender camera looks along -Z with +Y up. The
    conversion is therefore a 180-degree proper rotation around X, not a
    left/right reflection.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("OpenCV camera points must be finite [...,3]")
    return (points @ CV_TO_BLENDER_CAMERA_AXES.T).astype(np.float32)


def blender_camera_calibration_from_opencv(
    intrinsics: np.ndarray,
    image_size: np.ndarray,
    sensor_width_mm: float = 36.0,
) -> dict[str, float | int]:
    """Convert ``fx,fy,cx,cy,width,height`` to Blender camera settings."""
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    image_size = np.asarray(image_size, dtype=np.float64)
    if intrinsics.shape != (4,) or image_size.shape != (2,):
        raise ValueError("intrinsics/image_size must be [4]/[2]")
    if not np.isfinite(intrinsics).all() or not np.isfinite(image_size).all():
        raise ValueError("Camera calibration must be finite")
    fx, fy, cx, cy = intrinsics
    width, height = image_size
    if fx <= 0 or fy <= 0 or width <= 0 or height <= 0 or sensor_width_mm <= 0:
        raise ValueError("Focal lengths, image size, and sensor width must be positive")
    pixel_aspect_y = fx / fy
    return {
        "width": int(round(width)),
        "height": int(round(height)),
        "lens_mm": float(fx * sensor_width_mm / width),
        "sensor_width_mm": float(sensor_width_mm),
        "pixel_aspect_x": 1.0,
        "pixel_aspect_y": float(pixel_aspect_y),
        "shift_x": float((0.5 * width - cx) / width),
        "shift_y": float((cy - 0.5 * height) * pixel_aspect_y / width),
    }


def opencv_points_to_unreal_cm(points: np.ndarray) -> np.ndarray:
    """Map CV X-right/Y-down/Z-forward meters to UE X-forward/Y-right/Z-up cm."""
    points = np.asarray(points, dtype=np.float64)
    return (points @ CV_TO_UE_AXES.T * 100.0).astype(np.float32)


def unreal_points_cm_to_opencv(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (points @ np.linalg.inv(CV_TO_UE_AXES).T / 100.0).astype(np.float32)


def opencv_rotations_to_unreal(rotations: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float64)
    inverse = np.linalg.inv(CV_TO_UE_AXES)
    return CV_TO_UE_AXES @ rotations @ inverse


def unreal_rotations_to_opencv(rotations: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float64)
    inverse = np.linalg.inv(CV_TO_UE_AXES)
    return inverse @ rotations @ CV_TO_UE_AXES


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected XYZW quaternion [...,4], got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12) or not np.isfinite(quaternion).all():
        raise ValueError("Quaternion must be finite and nonzero")
    x, y, z, w = np.moveaxis(quaternion / norm, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def validate_parents(parents: np.ndarray) -> np.ndarray:
    parents = np.asarray(parents, dtype=np.int64)
    if parents.ndim != 1 or len(parents) == 0 or parents[0] != -1:
        raise ValueError("Parents must be a non-empty root-first vector")
    if any(not (0 <= int(parent) < joint) for joint, parent in enumerate(parents[1:], 1)):
        raise ValueError("Parents must be in parent-before-child topological order")
    return parents


def rest_global_transforms(
    rest_translations: np.ndarray,
    rest_rotations_quat_xyzw: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    parents = validate_parents(parents)
    translations = np.asarray(rest_translations, dtype=np.float64)
    rotations = quaternion_xyzw_to_matrix(rest_rotations_quat_xyzw)
    if translations.shape != (len(parents), 3) or rotations.shape != (
        len(parents), 3, 3
    ):
        raise ValueError("Rest translation/rotation shapes do not match parents")
    global_transform = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(parents), 4, 4)
    ).copy()
    for joint, parent in enumerate(parents):
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = rotations[joint]
        local[:3, 3] = translations[joint]
        global_transform[joint] = (
            local if parent < 0 else global_transform[parent] @ local
        )
    return global_transform


def animation_global_transforms(
    rotation_matrix: np.ndarray,
    rest_translations: np.ndarray,
    parents: np.ndarray,
    root_translation: np.ndarray | None = None,
) -> np.ndarray:
    """FK transforms for root-global/non-root-local learned rotations."""
    parents = validate_parents(parents)
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    translations = np.asarray(rest_translations, dtype=np.float64)
    if rotation.ndim != 4 or rotation.shape[1:] != (len(parents), 3, 3):
        raise ValueError(f"rotation_matrix must be [T,J,3,3], got {rotation.shape}")
    if translations.shape != (len(parents), 3):
        raise ValueError("rest_translations must be [J,3]")
    frames = rotation.shape[0]
    if root_translation is None:
        root_translation = np.zeros((frames, 3), dtype=np.float64)
    root_translation = np.asarray(root_translation, dtype=np.float64)
    if root_translation.shape != (frames, 3):
        raise ValueError("root_translation must be [T,3]")
    global_transform = np.broadcast_to(
        np.eye(4, dtype=np.float64), (frames, len(parents), 4, 4)
    ).copy()
    global_transform[:, 0, :3, :3] = rotation[:, 0]
    global_transform[:, 0, :3, 3] = translations[0] + root_translation
    for joint, parent in enumerate(parents[1:], 1):
        local = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
        local[:, :3, :3] = rotation[:, joint]
        local[:, :3, 3] = translations[joint]
        global_transform[:, joint] = global_transform[:, parent] @ local
    return global_transform


def fit_proper_similarity(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Fit target = scale * rotation @ source + translation, det(rotation)=+1."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity point sets must both be [N,3]")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    rotated = source_zero @ rotation.T
    scale = float(np.sum(rotated * target_zero) / np.sum(source_zero**2))
    translation = target_center - scale * (rotation @ source_center)
    fitted = scale * (source @ rotation.T) + translation
    errors = np.linalg.norm(fitted - target, axis=-1)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "errors": errors,
        "fitted": fitted,
    }


def similarity_homogeneous(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = float(scale) * np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def transform_points(points: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    similarity = np.asarray(similarity, dtype=np.float64)
    return points @ similarity[:3, :3].T + similarity[:3, 3]


def conjugate_transforms(transforms: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    transforms = np.asarray(transforms, dtype=np.float64)
    similarity = np.asarray(similarity, dtype=np.float64)
    return similarity @ transforms @ np.linalg.inv(similarity)


def source_driver_indices(
    processing: dict[str, Any],
    source_bone_names: list[str],
    source_parent_names: dict[str, str | None] | None = None,
) -> np.ndarray:
    """Map every source FBX bone to its nearest modeled/final ancestor."""
    source_joints = processing["source_skeleton"]["joints"]
    source_by_name = {joint["source_name"]: joint for joint in source_joints}
    final_by_source = {
        joint["source_name"]: int(joint["final_index"])
        for joint in processing["final_skeleton"]["joints"]
    }
    result = np.full(len(source_bone_names), -1, dtype=np.int32)
    for index, name in enumerate(source_bone_names):
        current_name: str | None = name
        visited = set()
        while current_name is not None:
            if current_name in final_by_source:
                result[index] = final_by_source[current_name]
                break
            if current_name in visited:
                raise ValueError("Cycle in source skeleton")
            visited.add(current_name)
            current = source_by_name.get(current_name)
            if current is not None:
                parent_name = current.get("parent_source_name")
                current_name = None if parent_name in (None, "") else str(parent_name)
            elif source_parent_names is not None and current_name in source_parent_names:
                current_name = source_parent_names[current_name]
            else:
                current_name = None
    return result


def collapse_skin_weights(
    vertex_count: int,
    final_joint_count: int,
    weight_vertex_indices: np.ndarray,
    weight_source_bone_indices: np.ndarray,
    weight_values: np.ndarray,
    source_drivers: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float | int]]:
    weights = np.zeros((vertex_count, final_joint_count), dtype=np.float32)
    unmodeled = 0.0
    for vertex, source_bone, value in zip(
        np.asarray(weight_vertex_indices, dtype=np.int64),
        np.asarray(weight_source_bone_indices, dtype=np.int64),
        np.asarray(weight_values, dtype=np.float32),
    ):
        driver = int(source_drivers[source_bone])
        if driver < 0:
            unmodeled += float(value)
        else:
            weights[vertex, driver] += value
    sums = weights.sum(axis=1)
    if unmodeled > 1e-6:
        raise ValueError(f"Skin contains {unmodeled:.6g} weight without a modeled ancestor")
    if not np.isfinite(sums).all() or np.any(sums <= 1e-8):
        zero = np.flatnonzero(sums <= 1e-8)
        raise ValueError(
            "Skin has non-finite or zero-weight vertices: "
            f"zero_vertices={len(zero)}, first_zero={zero[:16].tolist()}"
        )
    corrected = np.abs(sums - 1.0) > 1e-4
    stats: dict[str, float | int] = {
        "sum_min_before_normalization": float(sums.min()),
        "sum_max_before_normalization": float(sums.max()),
        "normalized_vertex_count": int(np.count_nonzero(corrected)),
        "max_normalization_correction": float(np.max(np.abs(sums - 1.0))),
    }
    weights /= sums[:, None]
    if np.max(np.abs(weights.sum(axis=1) - 1.0)) > 2e-6:
        raise ValueError("Normalized collapsed skin weights do not sum to one")
    return weights, unmodeled, stats


def deformation_deltas(
    rotation_matrix: np.ndarray,
    rest_translations: np.ndarray,
    rest_rotations_quat_xyzw: np.ndarray,
    parents: np.ndarray,
    root_translation: np.ndarray | None = None,
) -> np.ndarray:
    bind = rest_global_transforms(
        rest_translations, rest_rotations_quat_xyzw, parents
    )
    animated = animation_global_transforms(
        rotation_matrix, rest_translations, parents, root_translation
    )
    return animated @ np.linalg.inv(bind)[None]


def linear_blend_skinning(
    bind_vertices: np.ndarray,
    weights: np.ndarray,
    joint_deltas: np.ndarray,
) -> np.ndarray:
    """Skin vertices with [V,J] weights and [T,J,4,4] world deltas."""
    vertices = np.asarray(bind_vertices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    deltas = np.asarray(joint_deltas, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("bind_vertices must be [V,3]")
    if weights.shape[0] != len(vertices) or weights.shape[1] != deltas.shape[1]:
        raise ValueError("Skin weight shape does not match vertices/joints")
    output = np.zeros((deltas.shape[0], len(vertices), 3), dtype=np.float64)
    for joint in range(weights.shape[1]):
        active = np.flatnonzero(weights[:, joint] > 0)
        if not len(active):
            continue
        transformed = (
            np.einsum("txy,vy->tvx", deltas[:, joint, :3, :3], vertices[active])
            + deltas[:, joint, None, :3, 3]
        )
        output[:, active] += transformed * weights[active, joint][None, :, None]
    return output.astype(np.float32)


def load_mesh_asset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}
