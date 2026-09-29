"""The 2 m normalization for the final stored rest skeleton."""
import numpy as np


def metric_scale_from_static(static):
    """Return 2 / valid rest-joint AABB diagonal as a float32 scalar.

    Input translations are already in cache meters. This additional scale
    does not modify geometry, source-unit conversion, cameras, or animations.
    Rest rotations are local xyzw quaternions. Invalid joints are excluded
    from the bounds but still participate in FK when they are ancestors.
    """
    parents = np.asarray(static["parents"])
    translations = np.asarray(static["rest_translations"], dtype=np.float32)
    quaternions = np.asarray(static["rest_rotations_quat"], dtype=np.float32)
    n = int(static["joints"])
    valid = np.asarray(static.get("valid_joint_mask", np.ones(n, bool)), dtype=bool)
    norm = np.sum(quaternions * quaternions, axis=-1)
    x, y, z, w = np.moveaxis(quaternions, -1, 0)
    s = 2.0 / norm
    rotation = np.empty((n, 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = 1 - (y*y + z*z)*s
    rotation[:, 0, 1] = (x*y - w*z)*s
    rotation[:, 0, 2] = (x*z + w*y)*s
    rotation[:, 1, 0] = (x*y + w*z)*s
    rotation[:, 1, 1] = 1 - (x*x + z*z)*s
    rotation[:, 1, 2] = (y*z - w*x)*s
    rotation[:, 2, 0] = (x*z - w*y)*s
    rotation[:, 2, 1] = (y*z + w*x)*s
    rotation[:, 2, 2] = 1 - (x*x + y*y)*s
    positions = np.zeros_like(translations)
    global_rotation = np.empty_like(rotation)
    for joint, parent in enumerate(parents):
        if parent < 0:
            positions[joint] = translations[joint]
            global_rotation[joint] = rotation[joint]
        else:
            positions[joint] = positions[parent] + global_rotation[parent] @ translations[joint]
            global_rotation[joint] = global_rotation[parent] @ rotation[joint]
    positions -= positions[0:1]
    selected = positions[valid].astype(np.float64)
    diagonal = float(np.linalg.norm(np.ptp(selected, axis=0)))
    return np.float32(2.0 / diagonal)
