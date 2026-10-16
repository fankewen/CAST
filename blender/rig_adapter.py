"""Blender-only rig export and rot6d driver, shared by local and remote CAST clients.
Run inside Blender. No torch, network, or model loading in this module.
"""
import os
import numpy as np
import bpy
from mathutils import Matrix, Vector

OPENCV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0])
COORDINATE_SYSTEM = "opencv_right_handed_ydown_zforward_meters"


# --------------------------------------------------------------------------- math
def rot6d_to_matrix(values) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    c0, c1 = a[..., :3], a[..., 3:6]
    n0 = np.linalg.norm(c0, axis=-1, keepdims=True)
    b1 = np.divide(c0, np.where(n0 > 1e-6, n0, 1.0))
    c1 = c1 - (b1 * c1).sum(-1, keepdims=True) * b1
    n1 = np.linalg.norm(c1, axis=-1, keepdims=True)
    b2 = np.divide(c1, np.where(n1 > 1e-6, n1, 1.0))
    out = np.empty(a.shape[:-1] + (3, 3))
    out[..., :, 0] = b1
    out[..., :, 1] = b2
    out[..., :, 2] = np.cross(b1, b2)
    return out


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    quat = Matrix(np.asarray(m, dtype=np.float64).tolist()).to_quaternion()
    return np.array([quat.x, quat.y, quat.z, quat.w])


def camera_to_world_opencv(camera_object) -> np.ndarray:
    """Blender camera -> OpenCV ``camera_to_world`` (x right, y down, z forward)."""
    world = np.asarray(camera_object.matrix_world, dtype=np.float64).copy()
    rot = world[:3, :3]
    scale = float(np.linalg.norm(rot[:, 0]))
    world[:3, :3] = (rot / scale) @ OPENCV_TO_BLENDER_CAMERA
    return world


def rigid_matrices(world_matrices: np.ndarray) -> np.ndarray:
    """Strip scale so every rest matrix is a proper rigid transform."""
    out = np.array(world_matrices, dtype=np.float64, copy=True)
    for index in range(len(out)):
        rot = out[index][:3, :3]
        scale = float(np.linalg.norm(rot[:, 0]))
        if scale > 0:
            out[index][:3, :3] = rot / scale
    return out


def ordered_bones(armature_object):
    ordered, seen = [], set()

    def visit(bone):
        if bone.name in seen:
            return
        if bone.parent is not None:
            visit(bone.parent)
        seen.add(bone.name)
        ordered.append(bone)

    for bone in armature_object.data.bones:
        visit(bone)
    return ordered


def build_rig_asset(armature_object, camera_object, out_dir: str) -> dict:
    """Armature + scene camera -> CAST ``static.npy`` + ``bind_transforms.npz``."""
    bones = ordered_bones(armature_object)
    names = [bone.name for bone in bones]
    index = {name: i for i, name in enumerate(names)}
    parents = np.array(
        [-1 if bone.parent is None else index[bone.parent.name] for bone in bones],
        dtype=np.int32,
    )
    armature_world = np.asarray(armature_object.matrix_world, dtype=np.float64)
    world_rest = rigid_matrices(
        np.stack([armature_world @ np.asarray(bone.matrix_local, dtype=np.float64) for bone in bones])
    )

    camera_to_world = camera_to_world_opencv(camera_object)
    camera_from_world = np.linalg.inv(camera_to_world)
    camera_frame = np.einsum("ab,jbc->jac", camera_from_world, world_rest)

    joint_count = len(bones)
    rest_local = np.zeros((joint_count, 4, 4))
    rest_translations = np.zeros((joint_count, 3))
    rest_rotations = np.zeros((joint_count, 4))
    for joint, parent in enumerate(parents):
        local = camera_frame[joint] if parent < 0 else np.linalg.inv(camera_frame[parent]) @ camera_frame[joint]
        rest_local[joint] = local
        rest_translations[joint] = local[:3, 3]
        rest_rotations[joint] = matrix_to_quat_xyzw(local[:3, :3])

    # The cache defines metric_scale as 2 / AABB diagonal of the rest skeleton
    # in the camera frame, so measure it there and not in world space.
    points = camera_frame[:, :3, 3]
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    metric_scale = 2.0 / diagonal if diagonal > 0 else 1.0

    os.makedirs(out_dir, exist_ok=True)
    static_path = os.path.join(out_dir, "static.npy")
    bind_path = os.path.join(out_dir, "bind_transforms.npz")
    np.save(
        static_path,
        {
            "rig_id": armature_object.name,
            "joints": joint_count,
            "joint_names": np.array(names),
            "parents": parents,
            "rest_translations": rest_translations.astype(np.float32),
            "rest_rotations_quat": rest_rotations.astype(np.float32),
            "metric_scale": np.float32(metric_scale),
            "coordinate_system": COORDINATE_SYSTEM,
            "units": "meters",
            "skeleton_hash": "adapted-%s-%d" % (armature_object.name, joint_count),
        },
        allow_pickle=True,
    )
    np.savez(
        bind_path,
        world_rest=world_rest,
        camera_from_world=camera_from_world,
        parents=parents,
        joint_names=np.array(names),
        rest_local=rest_local,
    )
    return {
        "static": static_path,
        "bind_transforms": bind_path,
        "joint_count": joint_count,
        "metric_scale": metric_scale,
        "names": names,
        "rest_translations": rest_translations,
        "bone_lengths": np.linalg.norm(rest_translations[1:], axis=-1),
    }


# ------------------------------------------------------------------- pose driver
class RigDriver:
    """Writes camera-frame rot6d frames into the armature pose."""

    def __init__(self, armature_object, bind_path: str):
        bind = np.load(bind_path, allow_pickle=True)
        self.names = [str(x) for x in bind["joint_names"]]
        self.parents = np.asarray(bind["parents"]).astype(int)
        self.rest_translations = np.asarray(bind["rest_local"], dtype=np.float64)[:, :3, 3]
        camera_to_world = np.linalg.inv(np.asarray(bind["camera_from_world"], dtype=np.float64))

        self.rig = armature_object
        missing = [name for name in self.names if name not in self.rig.data.bones]
        if missing:
            raise RuntimeError(f"armature is missing {len(missing)} bones, e.g. {missing[:4]}")

        matrix_world = np.asarray(self.rig.matrix_world, dtype=np.float64)
        scale = float(np.linalg.norm(matrix_world[:3, :3][:, 0]))
        rigid_world = np.eye(4)
        rigid_world[:3, :3] = matrix_world[:3, :3] / scale
        rigid_world[:3, 3] = matrix_world[:3, 3] / scale
        self.to_armature = np.linalg.inv(rigid_world) @ camera_to_world
        self.armature_scale = scale

        self.rest_local = []
        for name in self.names:
            bone = self.rig.data.bones[name]
            local = bone.matrix_local.copy()
            if bone.parent is not None:
                if bone.parent.name not in self.names:
                    raise RuntimeError(f"parent {bone.parent.name!r} of {name!r} is not in the asset")
                local = bone.parent.matrix_local.inverted() @ local
            self.rest_local.append(local)
        self.applied = 0

    def apply_rot6d(self, values) -> None:
        rotations = rot6d_to_matrix(values)
        count = len(self.parents)
        globals_ = np.zeros((count, 4, 4))
        for joint, parent in enumerate(self.parents):
            local = np.eye(4)
            local[:3, :3] = rotations[joint]
            local[:3, 3] = self.rest_translations[joint]
            globals_[joint] = local if parent < 0 else globals_[parent] @ local
        world = np.einsum("ab,jbc->jac", self.to_armature, globals_)
        world[:, :3, 3] /= self.armature_scale
        targets = [Matrix(world[joint].tolist()) for joint in range(count)]
        bones = self.rig.pose.bones
        for joint in range(count):
            parent = self.parents[joint]
            local = targets[joint] if parent < 0 else targets[parent].inverted() @ targets[joint]
            bones[self.names[joint]].matrix_basis = self.rest_local[joint].inverted() @ local
        self.applied += 1
