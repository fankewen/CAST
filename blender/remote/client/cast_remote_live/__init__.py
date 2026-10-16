bl_info = {
    "name": "CAST Remote Live",
    "author": "CAST Contributors",
    "version": (1, 7, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > CAST Remote",
    "description": "Camera to remote CAST DINOv3 service and live armature driving",
    "category": "Animation",
}

import os
import sys
import shutil
import json
import http.client
import hashlib
import re
import socket
import struct
import subprocess
import tempfile
import threading
import time
from collections import deque
from math import radians, atan2, cos, sin, exp
from pathlib import Path
from urllib.parse import urlsplit

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from mathutils import Euler, Matrix, Vector


POSE_HEADER = struct.Struct("<II")
OPENCV_TO_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0])
COORDINATE_SYSTEM = "opencv_right_handed_ydown_zforward_meters"
# Source checkout keeps scripts next to the addon; installation bundles them inside it.
CLIENT_ROOT = Path(__file__).resolve().parent
if not (CLIENT_ROOT / "transport.py").is_file():
    CLIENT_ROOT = CLIENT_ROOT.parent
if str(CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIENT_ROOT))
from client_config import default_config_path, initialize, load_config, write_json

DEFAULT_ROOT = default_config_path().parent
ACTIVE_SESSION_PATH = DEFAULT_ROOT / "active_session.json"
CAMERA_DAEMON_LOG = DEFAULT_ROOT / "camera_daemon.log"
IMPORTED_OBJECT_TAG = "tosc_remote_imported"
MODEL_SETTINGS_PATH = DEFAULT_ROOT / "fbx_user_settings.json"


def prepare_client(scene):
    """Resolve writable state and persist the non-secret connection settings."""
    global DEFAULT_ROOT, ACTIVE_SESSION_PATH, CAMERA_DAEMON_LOG, MODEL_SETTINGS_PATH
    path = initialize(Path(bpy.path.abspath(scene.tosc_remote_config)))
    state_root = path.parent
    if state_root != DEFAULT_ROOT and (STATE.initializing or STATE.remote_ready or
            (STATE.process is not None and STATE.process.poll() is None) or
            (STATE.camera_daemon is not None and STATE.camera_daemon.poll() is None)):
        raise RuntimeError("Stop the current client before changing its config directory")
    DEFAULT_ROOT = state_root
    ACTIVE_SESSION_PATH = state_root / "active_session.json"
    CAMERA_DAEMON_LOG = state_root / "camera_daemon.log"
    MODEL_SETTINGS_PATH = state_root / "fbx_user_settings.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if scene.tosc_remote_server_url.strip():
        data["server_url"] = scene.tosc_remote_server_url.strip().rstrip("/")
    data["target_mode"] = scene.tosc_remote_target_mode
    data["fps"] = float(scene.tosc_remote_fps)
    data["pose_port"] = int(scene.tosc_remote_pose_port)
    data["control_port"] = int(scene.tosc_remote_control_port)
    write_json(path, data)
    return load_config(path)


def client_environment(scene):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if scene.tosc_remote_api_key.strip():
        env["CAST_API_KEY"] = scene.tosc_remote_api_key.strip()
    return env


MODEL_SETTING_NAMES = (
    "tosc_remote_root_mode", "tosc_remote_forward_axis", "tosc_remote_root_offset",
    "tosc_remote_fix_bones", "tosc_remote_fixed_bones",
    "tosc_remote_filter_enabled", "tosc_remote_filter_ms",
    "tosc_remote_rig_convention",
    "tosc_remote_fps", "tosc_remote_show_preview",
    "tosc_remote_sync_crab_legs",
    "tosc_remote_fixed_offset_bone", "tosc_remote_fixed_offset",
)
_model_settings_loading = 0
MODEL_SETTING_DEFAULTS = {
    "tosc_remote_root_mode": "PREDICTED", "tosc_remote_forward_axis": "AUTO",
    "tosc_remote_root_offset": [0.0, 0.0, 0.0],
    "tosc_remote_fix_bones": False, "tosc_remote_fixed_bones": "",
    "tosc_remote_filter_enabled": True, "tosc_remote_filter_ms": 60.0,
    "tosc_remote_rig_convention": "AUTO", "tosc_remote_fps": 24.0,
    "tosc_remote_show_preview": False,
    "tosc_remote_sync_crab_legs": False,
    "tosc_remote_fixed_offset_bone": "", "tosc_remote_fixed_offset": [0.0, 0.0, 0.0],
}
_model_settings_lock = threading.RLock()
CANONICAL_REST_ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
ZOO_FBX_HASHES = {
    "eb798cc4ad7cedb478c554daa8637360622864582e9b53b4d62d58b1152fda69",  # Crab
    "be019155295ca12e31f4b4030d9d64de7d3c9186e878080d641ccbd0d9b883e6",  # Horse
    "4a16208dab92e02b597912d38f1cf0d9b300b60301628f1d0280f2d44c7e5ffb",  # Ostrich
    "1cfb2f2e4ef16ba62016241496a3d0ad7f48eda1e2a8a48573c0e717f4226f9f",  # Anaconda
    "1fb0a64e31a9347283963ffc0013a51c1d5583735a31dccf1b660da7c8b15b86",  # Ant
    "87d0bfb491e73e9b30e02e9c9c75c4cb7402334fdedf15007e084c8cf3a92073",  # Eagle
}


def fbx_rig_convention(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "ZOO_RESET" if digest.hexdigest() in ZOO_FBX_HASHES else "FBX_LOCAL"

# Content-specific neck/head defaults include the verified numbered humanoid.
# User edits override defaults after their first migration to this revision.
NECK_HEAD_DEFAULT_REVISION = 1
NECK_HEAD_DEFAULTS = {'eb798cc4ad7cedb478c554daa8637360622864582e9b53b4d62d58b1152fda69': [],
 'be019155295ca12e31f4b4030d9d64de7d3c9186e878080d641ccbd0d9b883e6': ['Bip01_Neck',
                                                                      'Bip01_Neck1',
                                                                      'Bip01_Head',
                                                                      'Bip01_HeadNub'],
 '64224297dce370c6a51aae81314a73624883ab751d8060f40388135ef2780eaa': ['mixamorig_Neck',
                                                                      'mixamorig_Head'],
 '413c4fe52c3152a641d70797de2dce73d128d3c40adc18f1521fd53613f94407': ['mixamorig:Neck',
                                                                      'mixamorig:Head',
                                                                      'mixamorig:HeadTop_End',
                                                                      'mixamorig:HeadTop_End1'],
 'ba123dece31c76186e7f495f2adcac7f5899a3da7e4c53e2c0ae7938a2bf648b': ['mixamorig:Neck',
                                                                      'mixamorig:Neck1',
                                                                      'mixamorig:Head',
                                                                      'mixamorig:HeadTop_End'],
 '6bbbe461dd47b237cc7431c5867eab998536c0b14fa709e9165580ca8c8e8385': ['mixamorig:Neck',
                                                                      'mixamorig:Head',
                                                                      'mixamorig:HeadTop_End'],
 'f92d4392f18c8d102fc82c827c43111f51c7c09fdbcbf718a0d98cabcaf3d0c3': [],
 'c4e55f1c2309a2146521856345d110ca64f74ca841593b0cd413a010297039a9': ['joint_017', 'joint_018'],
 '68b0a6b4670521188d6dc507234866c257ef3434b135aaab8d7542f46820da36': ['Neck', 'Head'],
 '4a16208dab92e02b597912d38f1cf0d9b300b60301628f1d0280f2d44c7e5ffb': ['Bip01_Neck',
                                                                      'Bip01_Neck1',
                                                                      'Bip01_Neck2',
                                                                      'Bip01_Neck3',
                                                                      'Bip01_Neck4',
                                                                      'Bip01_Head',
                                                                      'Bip01_HeadNub'],
 '3614a6583a12a4e3b64d85cc2e203fc475c01fe293bab57ecd4aa90e84b520e8': ['mixamorig:Neck',
                                                                      'mixamorig:Head',
                                                                      'mixamorig:HeadTop_End'],
 '1cfb2f2e4ef16ba62016241496a3d0ad7f48eda1e2a8a48573c0e717f4226f9f': ['BN_Neck', 'BN_Head'],
 '1fb0a64e31a9347283963ffc0013a51c1d5583735a31dccf1b660da7c8b15b86': ['Bip01_Neck',
                                                                      'Bip01_Head',
                                                                      'Bip01_HeadNub'],
 '87d0bfb491e73e9b30e02e9c9c75c4cb7402334fdedf15007e084c8cf3a92073': ['Bip01_Neck',
                                                                      'Bip01_Neck1',
                                                                      'Bip01_Neck2',
                                                                      'Bip01_Head']}
FBX_POSE_PROFILES = {'eb798cc4ad7cedb478c554daa8637360622864582e9b53b4d62d58b1152fda69': {'name': 'Crab-TPOSE',
                                                                      'forward_axis': 'POS_X',
                                                                      'bones': [],
                                                                      'root_mode': 'PLUS_X',
                                                                      'sync_crab_legs': True},
 'be019155295ca12e31f4b4030d9d64de7d3c9186e878080d641ccbd0d9b883e6': {'name': 'HorseALL-TPOSE',
                                                                      'forward_axis': 'POS_X',
                                                                      'bones': ['Bip01_Neck',
                                                                                'Bip01_Neck1',
                                                                                'Bip01_Head',
                                                                                'Bip01_HeadNub']},
 '64224297dce370c6a51aae81314a73624883ab751d8060f40388135ef2780eaa': {'name': 'human04',
                                                                      'forward_axis': 'NEG_Y',
                                                                      'bones': ['mixamorig_Neck',
                                                                                'mixamorig_Head']},
 '4a16208dab92e02b597912d38f1cf0d9b300b60301628f1d0280f2d44c7e5ffb': {'name': 'Ostrich-Tpose',
                                                                      'forward_axis': 'POS_X',
                                                                      'root_mode': 'PLUS_X',
                                                                      'bones': ['Bip01_Neck',
                                                                                'Bip01_Neck1',
                                                                                'Bip01_Neck2',
                                                                                'Bip01_Neck3',
                                                                                'Bip01_Neck4',
                                                                                'Bip01_Head',
                                                                                'Bip01_HeadNub']},
 '3614a6583a12a4e3b64d85cc2e203fc475c01fe293bab57ecd4aa90e84b520e8': {'name': 'SandMouse_rest',
                                                                      'forward_axis': 'NEG_Y',
                                                                      'root_mode': 'PLUS_X',
                                                                      'bones': ['mixamorig:Neck',
                                                                                'mixamorig:Head',
                                                                                'mixamorig:HeadTop_End']},
 '68b0a6b4670521188d6dc507234866c257ef3434b135aaab8d7542f46820da36': {'name': 'mobj_001345',
                                                                      'forward_axis': 'NEG_Y',
                                                                      'root_mode': 'PLUS_X',
                                                                      'bones': ['Neck', 'Head']}}

def rot6d_to_matrix(values) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError("rot6d must be a finite [J, 6] array")
    c0, c1 = a[..., :3], a[..., 3:6]
    n0 = np.linalg.norm(c0, axis=-1, keepdims=True)
    b1 = np.divide(c0, np.where(n0 > 1e-6, n0, 1.0))
    b1 = np.where(n0 > 1e-6, b1, np.array([1.0, 0.0, 0.0]))
    c1 = c1 - (b1 * c1).sum(-1, keepdims=True) * b1
    n1 = np.linalg.norm(c1, axis=-1, keepdims=True)
    b2 = np.divide(c1, np.where(n1 > 1e-6, n1, 1.0))
    # Match the model decoder for zero or parallel predicted columns.
    fallback = np.eye(3)[np.argmin(np.abs(b1), axis=-1)]
    fallback -= (fallback * b1).sum(-1, keepdims=True) * b1
    fallback /= np.linalg.norm(fallback, axis=-1, keepdims=True)
    b2 = np.where(n1 > 1e-6, b2, fallback)
    out = np.empty(a.shape[:-1] + (3, 3))
    out[..., :, 0] = b1
    out[..., :, 1] = b2
    out[..., :, 2] = np.cross(b1, b2)
    return out


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    quat = Matrix(np.asarray(matrix, dtype=np.float64).tolist()).to_quaternion()
    return np.array([quat.x, quat.y, quat.z, quat.w])


def camera_to_world_opencv(camera_object) -> np.ndarray:
    world = np.asarray(camera_object.matrix_world, dtype=np.float64).copy()
    rotation = world[:3, :3]
    scale = float(np.linalg.norm(rotation[:, 0]))
    world[:3, :3] = (rotation / scale) @ OPENCV_TO_BLENDER_CAMERA
    return world


def rigid_matrices(world_matrices: np.ndarray) -> np.ndarray:
    out = np.array(world_matrices, dtype=np.float64, copy=True)
    for index in range(len(out)):
        rotation = out[index][:3, :3]
        scale = float(np.linalg.norm(rotation[:, 0]))
        if scale > 0:
            out[index][:3, :3] = rotation / scale
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


def inference_bones(armature_object):
    """Keep a single inference tree; never delete bones in the Blender rig.

    Only a detached root leaf with no skinning, parenting or rig dependency may
    be omitted. A genuine forest requires an explicit rig conversion.
    """
    bones = ordered_bones(armature_object)
    roots = [bone for bone in bones if bone.parent is None]
    if len(roots) <= 1:
        return bones, []
    referenced = set()
    for obj in bpy.data.objects:
        if obj.parent == armature_object and obj.parent_type == "BONE":
            referenced.add(obj.parent_bone)
        # Be conservative even when a mesh modifier is disabled or absent.
        if obj.type == "MESH":
            if any(mod.type == "ARMATURE" and mod.object == armature_object and
                   mod.use_bone_envelopes for mod in obj.modifiers):
                referenced.update(b.name for b in roots if b.use_deform)
            groups = {group.index: group.name for group in obj.vertex_groups}
            for vertex in obj.data.vertices:
                referenced.update(groups[g.group] for g in vertex.groups
                                  if g.weight > 0 and g.group in groups)
        holders = [obj] + (list(obj.pose.bones) if obj.type == "ARMATURE" and obj.pose else [])
        for holder in holders:
            for constraint in holder.constraints:
                for prop in constraint.bl_rna.properties:
                    if prop.type == "STRING":
                        value = getattr(constraint, prop.identifier, "")
                        if value in armature_object.data.bones:
                            referenced.add(value)
                for target in getattr(constraint, "targets", ()):
                    if target.target == armature_object:
                        referenced.add(target.subtarget)
        datablocks = [obj, getattr(obj, "data", None),
                      getattr(getattr(obj, "data", None), "shape_keys", None)]
        for datablock in datablocks:
            animation = getattr(datablock, "animation_data", None)
            for curve in animation.drivers if animation else ():
                for variable in curve.driver.variables:
                    for target in variable.targets:
                        if target.id == armature_object:
                            referenced.add(target.bone_target)
                            referenced.update(b.name for b in roots if b.name in target.data_path)
    ignored = [b.name for b in roots if not b.children and b.name not in referenced and
               not armature_object.pose.bones[b.name].constraints]
    kept = [b for b in bones if b.name not in ignored]
    kept_roots = [b.name for b in kept if b.parent is None]
    if len(kept_roots) != 1:
        raise ValueError("Model requires one root; multiple meaningful roots remain: " +
                         ", ".join(kept_roots or [b.name for b in roots]))
    return kept, ignored


def canonical_rig_frames(world_rest, parents, convention="ZOO_RESET", mesh_span=None):
    """Match the selected cache's camera-independent static skeleton.

    Zoo shares source joint axes; Mobj retains non-root FBX local frames.
    Blender skinning axes and world placement are kept in the output binding.
    """
    points = np.asarray(world_rest, dtype=np.float64)[:, :3, 3]
    edges = [[] for _ in parents]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            length = float(np.linalg.norm(points[joint] - points[parent]))
            edges[joint].append((parent, length))
            edges[parent].append((joint, length))
    diameter = 0.0
    for start in range(len(parents)):
        pending = [(start, -1, 0.0)]
        while pending:
            joint, previous, distance = pending.pop()
            diameter = max(diameter, distance)
            pending.extend((child, joint, distance + length)
                           for child, length in edges[joint] if child != previous)
    if not np.isfinite(diameter) or diameter <= 1e-8:
        raise ValueError("Skeleton must have nonzero finite bone lengths")
    local = np.tile(np.eye(4), (len(parents), 1, 1))
    if convention == "FBX_LOCAL":
        # Mobj retains the original non-root joint frames and resets only root.
        # Cache normalization is mesh extent 1 m ([-1,1] followed by scale .5).
        scale = float(mesh_span) if mesh_span is not None else diameter
        if not np.isfinite(scale) or scale <= 1e-8:
            raise ValueError("Rest mesh must have a finite nonzero extent")
        for joint, parent in enumerate(parents):
            if parent >= 0:
                local[joint] = np.linalg.inv(world_rest[parent]) @ world_rest[joint]
                local[joint, :3, 3] /= scale
        points = (points - points[0]) @ world_rest[0, :3, :3] / scale
        diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
        return local, np.tile(np.eye(3), (len(parents), 1, 1)), scale, 2.0 / diagonal
    if convention != "ZOO_RESET":
        raise ValueError("Unknown rig convention: " + str(convention))
    for joint, parent in enumerate(parents):
        if parent < 0:
            local[joint, :3, :3] = CANONICAL_REST_ROTATION
        else:
            local[joint, :3, 3] = (CANONICAL_REST_ROTATION.T @
                                   (points[joint] - points[parent])) / diameter
    model_to_bone = np.einsum("ab,jbc->jac", CANONICAL_REST_ROTATION.T,
                              world_rest[:, :3, :3])
    diagonal = float(np.linalg.norm(np.ptp(points / diameter, axis=0)))
    return local, model_to_bone, diameter, 2.0 / diagonal


def build_rig_asset(armature_object, camera_object, out_dir: str) -> dict:
    bones, ignored = inference_bones(armature_object)
    if not bones:
        raise ValueError("The armature has no bones")
    names = [bone.name for bone in bones]
    index = {name: i for i, name in enumerate(names)}
    parents = np.array(
        [-1 if bone.parent is None else index[bone.parent.name] for bone in bones],
        dtype=np.int32,
    )
    if parents[0] != -1 or any(not 0 <= int(p) < j for j, p in enumerate(parents[1:], 1)):
        raise ValueError("Export requires one root at index 0 and every parent before its child")
    armature_world = np.asarray(armature_object.matrix_world, dtype=np.float64)
    world_rest = rigid_matrices(
        np.stack(
            [
                armature_world @ np.asarray(bone.matrix_local, dtype=np.float64)
                for bone in bones
            ]
        )
    )

    camera_to_world = camera_to_world_opencv(camera_object)
    camera_from_world = np.linalg.inv(camera_to_world)
    joint_count = len(bones)
    convention = armature_object.get("tosc_rig_convention", "FBX_LOCAL")
    mesh_points = []
    if convention == "FBX_LOCAL":
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH" or not any(mod.type == "ARMATURE" and mod.object == armature_object for mod in obj.modifiers):
                continue
            # Use original rest vertices, never the currently evaluated pose.
            coords = np.empty((len(obj.data.vertices), 3), dtype=np.float64)
            obj.data.vertices.foreach_get("co", coords.ravel())
            transform = np.asarray(obj.matrix_world)
            mesh_points.append(coords @ transform[:3, :3].T + transform[:3, 3])
    mesh_span = None
    if mesh_points:
        source_points = np.concatenate(mesh_points) @ world_rest[0, :3, :3]
        mesh_span = float(np.ptp(source_points, axis=0).max())
    rest_local, model_to_bone, position_scale, metric_scale = canonical_rig_frames(world_rest, parents, convention, mesh_span)
    rest_translations = rest_local[:, :3, 3]
    rest_rotations = np.zeros((joint_count, 4))
    for joint in range(joint_count):
        rest_rotations[joint] = matrix_to_quat_xyzw(rest_local[joint, :3, :3])

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
        binding_version=np.int32(2),
        rig_convention=np.array(convention),
        model_to_bone=model_to_bone,
        position_scale=np.float64(position_scale),
        root_anchor_camera=(camera_from_world @ world_rest[0])[:3, 3],
    )
    return {
        "static": static_path,
        "bind_transforms": bind_path,
        "joint_count": joint_count,
        "metric_scale": metric_scale,
        "names": names,
        "ignored_bones": ignored,
        "rig_convention": convention,
        "rest_translations": rest_translations,
        "static": static_path,
    }


class RotationLowPass:
    """Time-based shortest-arc quaternion low-pass; never averages matrix columns."""

    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def apply(self, rotations, dt, tau):
        current = [Matrix(r.tolist()).to_quaternion().normalized() for r in rotations]
        if self.previous is not None and tau > 0:
            alpha = 1.0 - exp(-max(float(dt), 1e-6) / tau)
            filtered = []
            for old, new in zip(self.previous, current):
                if old.dot(new) < 0:
                    new.negate()
                filtered.append(old.slerp(new, alpha).normalized())
            current = filtered
        self.previous = [q.copy() for q in current]
        return np.array([q.to_matrix() for q in current], dtype=np.float64)


def stabilization_options(scene):
    # An unregistered module keeps diagnostic callers on the original raw path.
    return {
        "root_mode": getattr(scene, "tosc_remote_root_mode", "PREDICTED"),
        "forward_axis": getattr(scene, "tosc_remote_forward_axis", "AUTO"),
        "fix_bones": getattr(scene, "tosc_remote_fix_bones", False),
        "bone_names": getattr(scene, "tosc_remote_fixed_bones", "Neck,Neck1,Head"),
        "filter_enabled": getattr(scene, "tosc_remote_filter_enabled", False),
        "filter_ms": getattr(scene, "tosc_remote_filter_ms", 60.0),
        "sync_crab_legs": getattr(scene, "tosc_remote_sync_crab_legs", False),
        "fixed_offset_bone": getattr(scene, "tosc_remote_fixed_offset_bone", ""),
        "fixed_offset": tuple(getattr(scene, "tosc_remote_fixed_offset", (0.0, 0.0, 0.0))),
        "root_offset": tuple(getattr(scene, "tosc_remote_root_offset", (0.0, 0.0, 0.0))),
    }


class RigDriver:
    def __init__(self, armature_object, bind_path: str) -> None:
        bind = np.load(bind_path, allow_pickle=True)
        self.names = [str(value) for value in bind["joint_names"]]
        self.parents = np.asarray(bind["parents"]).astype(int)
        self.bind_rest = np.asarray(bind["rest_local"], dtype=np.float64)
        self.rest_translations = self.bind_rest[:, :3, 3]
        self.world_rest = np.asarray(bind["world_rest"], dtype=np.float64)
        self.camera_from_world = np.asarray(bind["camera_from_world"], dtype=np.float64)
        # Old saved bindings remain readable, but new imports use canonical axes.
        self.model_to_bone = (np.asarray(bind["model_to_bone"], dtype=np.float64)
                              if "model_to_bone" in bind else np.tile(np.eye(3), (len(self.names), 1, 1)))
        self.position_scale = float(bind["position_scale"]) if "position_scale" in bind else 1.0
        self.root_anchor_camera = (np.asarray(bind["root_anchor_camera"], dtype=np.float64)
                                   if "root_anchor_camera" in bind else np.zeros(3))
        camera_to_world = np.linalg.inv(
            np.asarray(bind["camera_from_world"], dtype=np.float64)
        )
        self.rig = armature_object
        missing = [name for name in self.names if name not in self.rig.data.bones]
        if missing:
            raise RuntimeError(
                f"armature is missing {len(missing)} bones, for example {missing[:4]}"
            )

        matrix_world = np.asarray(self.rig.matrix_world, dtype=np.float64)
        scale = float(np.linalg.norm(matrix_world[:3, :3][:, 0]))
        rigid_world = np.eye(4)
        rigid_world[:3, :3] = matrix_world[:3, :3] / scale
        # Translation is already in world units; only bone coordinates below
        # are divided by armature scale after subtracting this translation.
        rigid_world[:3, 3] = matrix_world[:3, 3]
        self.to_armature = np.linalg.inv(rigid_world) @ camera_to_world
        self.armature_scale = scale

        self.rest_local = []
        for name in self.names:
            bone = self.rig.data.bones[name]
            local = bone.matrix_local.copy()
            if bone.parent is not None:
                local = bone.parent.matrix_local.inverted() @ local
            self.rest_local.append(local)
        self.applied = 0
        self._last_processed_rotations = None
        self.rotation_filter = RotationLowPass()
        self.reset_stabilization()
        self.stabilization_status = "Raw prediction"
        self.crab_leg_groups = self._crab_leg_groups()
        self.crab_mirror_pairs = self._crab_mirror_pairs()
        self._mirror_crab_pose = False
        self._crab_fixed_indices = set()

    def _crab_leg_groups(self):
        """Middle leg leads its two neighbours on the same side, joint by joint."""
        indices = {name.rsplit(":", 1)[-1].lower(): j for j, name in enumerate(self.names)}
        pelvis = indices.get("bn_bip01_pelvis")
        if pelvis is None:
            return []
        groups = []
        for side in ("l", "r"):
            chains = []
            for start in (1, 6, 11):
                names = [f"bn_leg_{side}_{n:02d}" for n in range(start, start + 5)]
                names.append(f"bn_leg_{side}_{start + 4:02d}nub")
                if any(name not in indices for name in names):
                    return []
                chain = [indices[name] for name in names]
                if self.parents[chain[0]] != pelvis or any(self.parents[j] != p for p, j in zip(chain, chain[1:])):
                    return []
                chains.append(chain)
            groups.extend((middle, (first, last)) for first, middle, last in zip(*chains))
        return groups

    def _crab_mirror_pairs(self):
        if not self.crab_leg_groups:
            return []
        left = self.crab_leg_groups[:6]
        right = self.crab_leg_groups[6:]
        normal = self.world_rest[left[0][0], :3, 3] - self.world_rest[right[0][0], :3, 3]
        length = np.linalg.norm(normal)
        if length < 1e-8:
            return []
        normal = normal / length
        self.crab_rest_normal = normal
        self.crab_rest_center = (self.world_rest[left[0][0], :3, 3] + self.world_rest[right[0][0], :3, 3]) / 2.0
        self.crab_handedness = {}
        reflection = np.eye(3) - 2.0 * np.outer(normal, normal)
        pairs = []
        for (ls, lt), (rs, rt) in zip(left, right):
            for r, l in zip((rs, *rt), (ls, *lt)):
                # Reflect the motion in the body's rest symmetry plane, then
                # express it in the target joint's own model rest frame.
                right_frame = self.world_rest[r, :3, :3] @ self.model_to_bone[r].T
                left_frame = self.world_rest[l, :3, :3] @ self.model_to_bone[l].T
                transfer = left_frame.T @ reflection @ right_frame
                u, _, vt = np.linalg.svd(transfer)
                pairs.append((r, l, u @ vt))
                # FBX bone axes need not point along the skinned leg segment.
                # Align the actual child-joint directions, keeping the closest
                # rest roll, instead of assuming native Y is the limb axis.
                rb = self.world_rest[r, :3, :3]
                lb = self.world_rest[l, :3, :3]
                rc = next((j for j, p in enumerate(self.parents) if p == r), None)
                lc = next((j for j, p in enumerate(self.parents) if p == l), None)
                rv = rb.T @ (self.world_rest[rc, :3, 3] - self.world_rest[r, :3, 3]) if rc is not None else np.array([0., 1., 0.])
                lv = lb.T @ (self.world_rest[lc, :3, 3] - self.world_rest[l, :3, 3]) if lc is not None else np.array([0., 1., 0.])
                handedness = rb.T @ reflection @ lb
                align = np.asarray(Vector(handedness @ lv).rotation_difference(Vector(rv)).to_matrix())
                handedness = align @ handedness
                u, _, vt = np.linalg.svd(handedness)
                self.crab_handedness[l] = u @ vt
        return pairs

    def synchronize_crab_legs(self, rotations):
        result = rotations.copy()
        if not self.crab_mirror_pairs:
            return result
        # The right middle leg is the sole motion source for all six legs.
        for source, targets in self.crab_leg_groups[6:]:
            # Copy Blender-local rest-relative motion, preserving each leg's
            # own rest spread and bone axes. Model outputs remain final rotations.
            basis = self.model_to_bone[source]
            delta = basis.T @ self.bind_rest[source, :3, :3].T @ rotations[source] @ basis
            for target in targets:
                basis = self.model_to_bone[target]
                result[target] = self.bind_rest[target, :3, :3] @ basis @ delta @ basis.T
        for right, left, transfer in self.crab_mirror_pairs:
            delta = self.bind_rest[right, :3, :3].T @ result[right]
            result[left] = self.bind_rest[left, :3, :3] @ transfer @ delta @ transfer.T
        return result

    def reset_stabilization(self):
        self.rotation_filter.reset()
        self._last_frame_index = None
        self._last_pose_time = None
        self._stabilization_key = None

    def _rest_forward(self, axis):
        axes = {"POS_X": (1, 0, 0), "NEG_X": (-1, 0, 0),
                "POS_Y": (0, 1, 0), "NEG_Y": (0, -1, 0)}
        if axis in axes:
            return np.array(axes[axis], dtype=np.float64)
        # Only infer a heading from identifiable humanoid feet. Unknown skeletons
        # retain their rest orientation until the user supplies a forward axis.
        indices = {name.rsplit(":", 1)[-1].lower(): i for i, name in enumerate(self.names)}
        forward = np.zeros(3)
        for side in ("left", "right"):
            foot, toe = indices.get(side + "foot"), indices.get(side + "toebase")
            if foot is not None and toe is not None:
                forward += self.world_rest[toe, :3, 3] - self.world_rest[foot, :3, 3]
        forward[2] = 0
        norm = np.linalg.norm(forward)
        return forward / norm if norm > 1e-6 else None

    def stabilize_rotations(self, rotations, options, frame_index=None, fps=24.0, now=None):
        fps = max(float(fps), 1.0)
        key = (options.get("root_mode", "PREDICTED"), options.get("forward_axis", "AUTO"),
               bool(options.get("fix_bones", False)), str(options.get("bone_names", "")),
               bool(options.get("filter_enabled", False)), float(options.get("filter_ms", 60)), fps,
               bool(options.get("sync_crab_legs", False)),
               str(options.get("fixed_offset_bone", "")).strip().lower(),
               tuple(options.get("fixed_offset", (0.0, 0.0, 0.0))))
        now = time.monotonic() if now is None else float(now)
        dt = 1.0 / fps
        discontinuity = key != self._stabilization_key
        if frame_index is not None and self._last_frame_index is not None:
            gap = int(frame_index) - self._last_frame_index
            discontinuity |= gap <= 0 or gap / fps > 0.5
            if gap > 0:
                dt = gap / fps
        if self._last_pose_time is not None:
            discontinuity |= now - self._last_pose_time > 0.5 or now < self._last_pose_time
        if discontinuity:
            self.rotation_filter.reset()
        self._stabilization_key = key
        self._last_frame_index = int(frame_index) if frame_index is not None else None
        self._last_pose_time = now
        result = np.array(rotations, dtype=np.float64, copy=True)
        locked = {}
        root_mode = key[0]
        root_note = "Root predicted"
        if root_mode != "PREDICTED":
            turn = np.eye(3)
            root_note = "Root at rest"
            if root_mode == "PLUS_X":
                forward = self._rest_forward(key[1])
                if forward is not None:
                    angle = -atan2(forward[1], forward[0]); c, s = cos(angle), sin(angle)
                    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    root_note = "Root facing +X"
                else:
                    root_note = "Root at rest; choose Rest Forward for +X"
            for j, parent in enumerate(self.parents):
                if parent < 0:
                    locked[j] = (self.camera_from_world[:3, :3] @ turn @ self.world_rest[j, :3, :3]
                                 @ self.model_to_bone[j].T)
        requested = {n.strip().lower() for n in key[3].replace(";", ",").split(",") if n.strip()}
        fixed_names = []
        if key[2]:
            for j, name in enumerate(self.names):
                if self.parents[j] >= 0 and (name.lower() in requested or name.rsplit(":", 1)[-1].lower() in requested):
                    locked[j] = self.bind_rest[j, :3, :3]
                    if key[8] in (name.lower(), name.rsplit(":", 1)[-1].lower()) and any(key[9]):
                        # Offset axes are the FBX's Blender-world rest axes.
                        # Conjugate into the MODEL parent frame, then follow the
                        # moving parent. Apply once at the selected chain base.
                        parent = self.parents[j]
                        parent_rest = (self.camera_from_world[:3, :3]
                                       @ self.world_rest[parent, :3, :3]
                                       @ self.model_to_bone[parent].T)
                        axes = np.linalg.inv(parent_rest) @ self.camera_from_world[:3, :3]
                        offset = np.asarray(Euler(key[9], "XYZ").to_matrix())
                        locked[j] = axes @ offset @ np.linalg.inv(axes) @ locked[j]
                    fixed_names.append(name.rsplit(":", 1)[-1])
        for j, rotation in locked.items():
            result[j] = rotation
        if key[4] and key[5] > 0:
            result = self.rotation_filter.apply(result, dt, key[5] / 1000.0)
        else:
            self.rotation_filter.reset()
        if key[7] and self.crab_leg_groups:
            result = self.synchronize_crab_legs(result)
        # Locks are exact, even on a toggle or a filter history change.
        for j, rotation in locked.items():
            result[j] = rotation
        self._mirror_crab_pose = bool(key[7] and self.crab_mirror_pairs)
        self._crab_fixed_indices = set(locked)
        self.stabilization_status = root_note + "; fixed: " + (", ".join(fixed_names) or "none")
        return result

    def apply_rot6d(self, values, *, options=None, frame_index=None, fps=24.0) -> None:
        rotations = rot6d_to_matrix(values)
        count = len(self.parents)
        if len(rotations) != count:
            raise ValueError(f"pose has {len(rotations)} joints; bound rig has {count}")
        if options is not None:
            rotations = self.stabilize_rotations(rotations, options, frame_index, fps)
        else:
            self.reset_stabilization()
            self._mirror_crab_pose = False
        self._last_processed_rotations = rotations.copy()
        self.apply_processed_rotations(rotations, (options or {}).get("root_offset", (0, 0, 0)))
        self.applied += 1

    def apply_processed_rotations(self, rotations, root_offset):
        # Offset is applied AFTER smoothing, in Blender world axes, about the
        # root's position. Never feed it back into the filter or cached pose.
        rotations = np.array(rotations, dtype=np.float64, copy=True)
        if any(root_offset):
            offset_world = np.asarray(Euler(tuple(root_offset), "XYZ").to_matrix())
            camera = self.camera_from_world[:3, :3]
            offset_camera = camera @ offset_world @ np.linalg.inv(camera)
            for joint, parent in enumerate(self.parents):
                if parent < 0:
                    rotations[joint] = offset_camera @ rotations[joint]
        count = len(self.parents)
        globals_ = np.zeros((count, 4, 4))
        for joint, parent in enumerate(self.parents):
            local = np.eye(4)
            local[:3, :3] = rotations[joint]
            local[:3, 3] = self.rest_translations[joint]
            globals_[joint] = local if parent < 0 else globals_[parent] @ local
        # Predictions are FINAL local rotations in MODEL axes, not rest deltas.
        # Change each global joint frame to the Blender bone frame only after FK.
        globals_[:, :3, :3] = globals_[:, :3, :3] @ self.model_to_bone
        globals_[:, :3, 3] = globals_[:, :3, 3] * self.position_scale + self.root_anchor_camera
        world = np.einsum("ab,jbc->jac", self.to_armature, globals_)
        world[:, :3, 3] /= self.armature_scale
        if self._mirror_crab_pose:
            self.mirror_crab_pose_targets(world)
        targets = [Matrix(world[joint].tolist()) for joint in range(count)]
        bones = self.rig.pose.bones
        for joint in range(count):
            parent = self.parents[joint]
            local = (
                targets[joint]
                if parent < 0
                else targets[parent].inverted() @ targets[joint]
            )
            bones[self.names[joint]].matrix_basis = (
                self.rest_local[joint].inverted() @ local
            )

    def mirror_crab_pose_targets(self, targets):
        """Mirror displayed left legs, including the FBX's asymmetric rest offsets."""
        pelvis = int(self.parents[self.crab_leg_groups[0][0]])
        rig_world = np.asarray(self.rig.matrix_world, dtype=np.float64)
        rest_body = np.asarray(self.rig.data.bones[self.names[pelvis]].matrix_local)
        body_turn = targets[pelvis, :3, :3] @ rest_body[:3, :3].T
        normal = np.linalg.inv(rig_world[:3, :3]) @ self.crab_rest_normal
        normal /= np.linalg.norm(normal)
        normal = body_turn @ normal
        reflection = np.eye(3) - 2.0 * np.outer(normal, normal)
        center_rest = (np.linalg.inv(rig_world) @ np.append(self.crab_rest_center, 1.0))[:3]
        center = targets[pelvis, :3, 3] + body_turn @ (center_rest - rest_body[:3, 3])
        for right, left, _ in self.crab_mirror_pairs:
            if left in self._crab_fixed_indices:
                continue
            targets[left, :3, :3] = reflection @ targets[right, :3, :3] @ self.crab_handedness[left]
            targets[left, :3, 3] = center + reflection @ (targets[right, :3, 3] - center)


class PoseReceiver:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.queue = deque(maxlen=256)
        self.server = None
        self.stop_event = threading.Event()
        self.thread = None
        self.received = 0

    def start(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

    def _accept_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                connection, _address = self.server.accept()
            except OSError:
                return
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            buffer = bytearray()
            try:
                while not self.stop_event.is_set():
                    try:
                        chunk = connection.recv(1 << 16)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    while True:
                        if len(buffer) < POSE_HEADER.size:
                            break
                        frame_index, joint_count = POSE_HEADER.unpack_from(buffer, 0)
                        payload_size = joint_count * 6 * 4
                        if len(buffer) < POSE_HEADER.size + payload_size:
                            break
                        values = struct.unpack_from(
                            f"<{joint_count * 6}f",
                            buffer,
                            POSE_HEADER.size,
                        )
                        del buffer[: POSE_HEADER.size + payload_size]
                        self.received += 1
                        self.queue.append(
                            (
                                frame_index,
                                np.asarray(values, dtype=np.float64).reshape(
                                    joint_count,
                                    6,
                                ),
                            )
                        )
            finally:
                connection.close()

    def stop(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass
            self.server = None


class RemoteState:
    def __init__(self) -> None:
        self.process = None
        self.camera_daemon = None
        self.process_reader = None
        self.receiver = None
        self.driver = None
        self.asset = None
        self.status = "idle"
        self.log_lines = deque(maxlen=120)
        self.runtime_dir = None
        self.initializing = False
        self.remote_ready = False
        self.camera_ready = False
        self.camera_waiting = False
        self.camera_wait_started = 0.0
        self.init_thread = None
        self.init_result = None
        self.init_error = None
        self.init_generation = 0

    def note(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        self.status = message
        self.log_lines.append(line)
        print("CAST Remote " + line, flush=True)


STATE = RemoteState()


def model_settings_snapshot(scene):
    return {name: (list(getattr(scene, name)) if name in ("tosc_remote_root_offset", "tosc_remote_fixed_offset")
                   else getattr(scene, name)) for name in MODEL_SETTING_NAMES}


def read_model_settings():
    data = (json.loads(MODEL_SETTINGS_PATH.read_text(encoding="utf-8"))
            if MODEL_SETTINGS_PATH.exists() else {"version": 1, "models": {}})
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("models"), dict):
        raise ValueError("Invalid FBX settings file: " + str(MODEL_SETTINGS_PATH))
    # Each model's own file is authoritative; the old aggregate remains a
    # compatibility index for existing installations and settings backups.
    for path in model_config_directory().glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if (not re.fullmatch(r"[0-9a-f]{64}", path.stem)
                or item.get("fbx_sha256") != path.stem or not isinstance(item.get("settings"), dict)):
            raise ValueError("Invalid per-FBX configuration: " + str(path))
        data["models"][path.stem] = item
    return data


def model_config_directory():
    return MODEL_SETTINGS_PATH.parent / (MODEL_SETTINGS_PATH.stem + "_models")


def atomic_settings_json(path, data):
    raw = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".fbx-settings-", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def neck_head_default_bones(key, names):
    if key in NECK_HEAD_DEFAULTS:
        bones = NECK_HEAD_DEFAULTS[key]
        missing = set(bones) - set(names)
        if missing:
            raise ValueError("Neck/head defaults have missing bones: " + ", ".join(sorted(missing)))
        return list(bones)
    return [name for name in names if re.search(
        r"(?:^|[:_ ])(?:neck\d*|head\d*|headnub\d*|headtop_end\d*)$", name, re.I)]


def add_neck_head_defaults(settings, bones):
    settings = dict(settings)
    if bones:
        # Keep only enabled user locks; disabled lists can contain old suggestions.
        previous = re.split(r"[,;]", settings.get("tosc_remote_fixed_bones", "")) if settings.get("tosc_remote_fix_bones") else []
        names = list(dict.fromkeys([n.strip() for n in previous if n.strip()] + list(bones)))
        settings.update(tosc_remote_fixed_bones=",".join(names), tosc_remote_fix_bones=True)
    return settings


def saved_settings_with_defaults(saved, key, names=None):
    settings = default_model_settings(key, names)
    settings.update({name: value for name, value in saved["settings"].items() if name in MODEL_SETTING_NAMES})
    bones = neck_head_default_bones(key, names) if names is not None else NECK_HEAD_DEFAULTS.get(key, [])
    if saved.get("neck_head_default_revision", 0) < NECK_HEAD_DEFAULT_REVISION:
        settings = add_neck_head_defaults(settings, bones)
    return settings


def default_model_settings(key, names=None):
    settings = dict(MODEL_SETTING_DEFAULTS)
    bones = neck_head_default_bones(key, names) if names is not None else NECK_HEAD_DEFAULTS.get(key, [])
    settings.update(tosc_remote_fixed_bones=",".join(bones), tosc_remote_fix_bones=bool(bones))
    profile = FBX_POSE_PROFILES.get(key)
    if profile is not None:
        settings["tosc_remote_forward_axis"] = profile["forward_axis"]
        settings["tosc_remote_root_mode"] = profile.get("root_mode", "PREDICTED")
        settings["tosc_remote_sync_crab_legs"] = profile.get("sync_crab_legs", False)
    return settings


def save_active_model_settings(scene):
    key = getattr(scene, "tosc_remote_settings_key", "")
    if _model_settings_loading or not key:
        return
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Invalid active FBX identity")
    with _model_settings_lock:
        data = read_model_settings()
        settings = model_settings_snapshot(scene)
        source = scene.tosc_remote_settings_source
        old = data["models"].get(key, {})
        if (old.get("settings") == settings and old.get("source_path") == source
                and old.get("neck_head_default_revision") == NECK_HEAD_DEFAULT_REVISION
                and (model_config_directory() / (key + ".json")).is_file()):
            return
        data["models"][key] = {"name": Path(source).name, "source_path": source,
                                "fbx_sha256": key,
                                "neck_head_default_revision": NECK_HEAD_DEFAULT_REVISION,
                                "settings": settings, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        atomic_settings_json(model_config_directory() / (key + ".json"), data["models"][key])
        atomic_settings_json(MODEL_SETTINGS_PATH, data)
    scene.tosc_remote_settings_status = "Auto-saved: " + Path(source).name


def update_model_settings(scene, context):
    if _model_settings_loading:
        return
    try:
        save_active_model_settings(scene)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        scene.tosc_remote_settings_status = "Save failed: " + str(exc)
        STATE.note(scene.tosc_remote_settings_status)


def sync_input_fps(scene):
    path = Path(bpy.path.abspath(scene.tosc_remote_config))
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    fps = float(scene.tosc_remote_fps)
    if data.get("fps") != fps:
        data["fps"] = fps
        atomic_settings_json(path, data)


def update_input_fps(scene, context):
    if _model_settings_loading:
        return
    update_model_settings(scene, context)
    try:
        sync_input_fps(scene)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        STATE.note("Input FPS update failed: " + str(exc))


def restore_model_settings(scene, settings):
    global _model_settings_loading
    if not isinstance(settings, dict):
        raise ValueError("Invalid saved model settings")
    _model_settings_loading += 1
    try:
        for name in MODEL_SETTING_NAMES:
            if name in settings:
                setattr(scene, name, settings[name])
    finally:
        _model_settings_loading -= 1


def armature_rest_signature(armature):
    digest = hashlib.sha256()
    for bone in ordered_bones(armature):
        digest.update(json.dumps([bone.name, bone.parent.name if bone.parent else None]).encode())
        digest.update(np.asarray(bone.matrix_local, dtype=np.float32).tobytes())
    return digest.hexdigest()


def prepare_model_switch(scene):
    """Save outgoing settings before deleting any character objects."""
    save_active_model_settings(scene)
    read_model_settings()  # Fail before changing the scene if the file is unreadable.
    if not scene.tosc_remote_settings_key:
        rigs = [obj for obj in scene.objects if obj.type == "ARMATURE" and obj.get(IMPORTED_OBJECT_TAG, False)]
        if len(rigs) == 1:
            return {"signature": armature_rest_signature(rigs[0]), "settings": model_settings_snapshot(scene)}
    return None


def activate_model_settings(scene, path, armature, legacy=None):
    global _model_settings_loading
    with open(path, "rb") as stream:
        key = hashlib.file_digest(stream, "sha256").hexdigest()
    saved = read_model_settings()["models"].get(key)
    _model_settings_loading += 1
    try:
        scene.tosc_remote_settings_key = key
        scene.tosc_remote_settings_source = str(Path(path).resolve())
        scene.tosc_remote_fbx = str(Path(path).resolve())
        if saved is not None:
            restore_model_settings(scene, saved_settings_with_defaults(saved, key,
                [bone.name for bone in armature.data.bones]))
            scene.tosc_remote_pose_profile = "Saved: " + Path(path).name
        elif legacy is not None and legacy["signature"] == armature_rest_signature(armature):
            restore_model_settings(scene, add_neck_head_defaults(legacy["settings"],
                neck_head_default_bones(key, [bone.name for bone in armature.data.bones])))
            scene.tosc_remote_pose_profile = "Migrated from scene: " + Path(path).name
        else:
            restore_model_settings(scene, default_model_settings(key,
                [bone.name for bone in armature.data.bones]))
            apply_fbx_pose_profile(scene, path, [bone.name for bone in armature.data.bones])
        armature["tosc_source_sha256"] = key
        armature["tosc_source_path"] = str(Path(path).resolve())
    finally:
        _model_settings_loading -= 1
    save_active_model_settings(scene)
    sync_input_fps(scene)
    scene.tosc_remote_settings_status = "Auto-save enabled: " + Path(path).name


@persistent
def restore_saved_model_settings(_unused=None):
    for scene in bpy.data.scenes:
        key = getattr(scene, "tosc_remote_settings_key", "")
        if not key:
            continue
        try:
            saved = read_model_settings()["models"].get(key)
            if saved is not None:
                rig = next((obj for obj in scene.objects if obj.type == "ARMATURE"
                            and obj.get("tosc_source_sha256") == key), None)
                names = [bone.name for bone in rig.data.bones] if rig is not None else None
                restore_model_settings(scene, saved_settings_with_defaults(saved, key, names))
                save_active_model_settings(scene)
                if scene == bpy.context.scene:
                    sync_input_fps(scene)
                scene.tosc_remote_pose_profile = "Saved: " + saved["name"]
                scene.tosc_remote_settings_status = "Restored: " + saved["name"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            scene.tosc_remote_settings_status = "Restore failed: " + str(exc)
            STATE.note(scene.tosc_remote_settings_status)


def apply_fbx_pose_profile(scene, path, names):
    """Set heading metadata and default neck/head locks for a new model."""
    scene.tosc_remote_pose_profile = "Custom settings"
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    key = digest.hexdigest()
    bones = neck_head_default_bones(key, names)
    profile = FBX_POSE_PROFILES.get(key)
    scene.tosc_remote_fixed_bones = ",".join(bones)
    scene.tosc_remote_fix_bones = bool(bones)
    if profile is None:
        scene.tosc_remote_pose_profile = "Neck/head defaults: " + Path(path).name
        return None
    missing = set(profile["bones"]) - set(names)
    if missing:
        raise ValueError("Tested FBX profile has missing bones: " + ", ".join(sorted(missing)))
    scene.tosc_remote_root_mode = profile.get("root_mode", "PREDICTED")
    scene.tosc_remote_forward_axis = profile["forward_axis"]
    scene.tosc_remote_root_offset = (0.0, 0.0, 0.0)
    scene.tosc_remote_pose_profile = profile["name"]
    return profile


def update_root_offset(scene, context):
    if _model_settings_loading:
        return
    update_model_settings(scene, context)
    driver = STATE.driver
    if driver is None or driver._last_processed_rotations is None:
        return
    try:
        driver.apply_processed_rotations(driver._last_processed_rotations, scene.tosc_remote_root_offset)
        if context is not None and context.view_layer is not None:
            context.view_layer.update()
    except (ReferenceError, RuntimeError, KeyError) as exc:
        STATE.note(f"Root offset preview unavailable: {exc}")




def read_process_output(process) -> None:
    try:
        for raw in iter(process.stdout.readline, ""):
            line = raw.strip()
            if line and ("STATUS" in line or "ERROR" in line):
                STATE.log_lines.append(line[:240])
    except Exception:
        pass


def init_timer() -> float:
    if not STATE.initializing:
        return 0.25
    if STATE.init_thread is not None and STATE.init_thread.is_alive():
        return 0.25
    try:
        finish_initialization()
    except Exception as exc:
        STATE.remote_ready = False
        STATE.init_error = str(exc)
        STATE.note(f"remote skeleton initialization failed: {exc}")
    finally:
        STATE.initializing = False
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    return 0.25


def finish_initialization():
    if STATE.init_error:
        raise RuntimeError(STATE.init_error)
    if STATE.init_result:
        if STATE.driver is None or STATE.init_result.get("joint_names") != STATE.driver.names:
            raise RuntimeError("Session skeleton differs from bound Blender rig; import again")
        active_path = Path(ACTIVE_SESSION_PATH)
        temp_path = active_path.with_suffix(".new.json")
        temp_path.write_text(json.dumps(STATE.init_result, ensure_ascii=True), encoding="utf-8")
        os.replace(temp_path, active_path)
        STATE.remote_ready = True
        STATE.note(
            "remote skeleton ready: "
            f"session={STATE.init_result['session_id']} "
            f"joints={STATE.init_result.get('joints')}"
        )
    else:
        raise RuntimeError("Session helper stopped without a result; import again")


def camera_wait_timer() -> float:
    if not STATE.camera_waiting:
        return 0.5
    if (
        STATE.camera_daemon is not None
        and STATE.camera_daemon.poll() is not None
    ):
        STATE.camera_waiting = False
        STATE.camera_ready = False
        STATE.note("camera daemon stopped during warm-up")
        return 0.5

    try:
        lines = Path(CAMERA_DAEMON_LOG).read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        lines = []
    fps_values = []
    for line in lines[-30:]:
        if "STATUS CAPTURE" not in line:
            continue
        match = re.search(r"fps=([0-9.]+)", line)
        if match:
            fps_values.append(float(match.group(1)))

    if len(fps_values) >= 3:
        recent = fps_values[-3:]
        average = sum(recent) / len(recent)
        if average > 0.0 and max(recent) - min(recent) <= max(1.0, average * 0.15):
            STATE.camera_waiting = False
            STATE.camera_ready = True
            STATE.note(f"camera ready and stable: fps={average:.1f}")
            return 0.5

    if time.time() - STATE.camera_wait_started > 90.0:
        STATE.camera_waiting = False
        STATE.camera_ready = False
        STATE.note("camera warm-up timed out")
    return 0.5


def camera_feed_ready() -> bool:
    path = Path(CAMERA_DAEMON_LOG)
    try:
        if time.time() - path.stat().st_mtime > 10.0:
            return False
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    for line in reversed(lines[-20:]):
        if "STATUS CAMERA_STOPPED" in line:
            return False
        if "STATUS CAPTURE" in line and "fps=" in line:
            return True
    return False


def send_control(port: int, command: str) -> dict:
    with socket.create_connection(("127.0.0.1", port), timeout=3.0) as connection:
        connection.sendall(
            (json.dumps({"cmd": command}) + "\n").encode("utf-8")
        )
        buffer = b""
        while b"\n" not in buffer:
            chunk = connection.recv(4096)
            if not chunk:
                break
            buffer += chunk
    if not buffer:
        raise RuntimeError("camera worker returned no response")
    return json.loads(buffer.split(b"\n", 1)[0].decode("utf-8"))


def stop_worker() -> None:
    process = STATE.process
    STATE.process = None
    if process is not None and process.poll() is None:
        try:
            process.wait(timeout=3.0)
            return
        except subprocess.TimeoutExpired:
            pass
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def stop_camera():
    process, STATE.camera_daemon = STATE.camera_daemon, None
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        Path(CAMERA_DAEMON_LOG).unlink(missing_ok=True)
    STATE.camera_ready = False
    STATE.camera_waiting = False


def ensure_camera_daemon(scene) -> None:
    process = STATE.camera_daemon
    if process is not None and process.poll() is None:
        return

    if camera_feed_ready():
        return

    python = bpy.path.abspath(scene.tosc_remote_python)
    config = bpy.path.abspath(scene.tosc_remote_config)
    daemon_script = CLIENT_ROOT / "camera_standalone.py"
    if not Path(python).is_file():
        raise RuntimeError(f"python not found: {python}")
    if not daemon_script.is_file():
        raise RuntimeError(f"camera daemon not found: {daemon_script}")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    STATE.camera_daemon = subprocess.Popen(
        [
            python,
            str(daemon_script),
            "--config",
            config,
            "--preview" if scene.tosc_remote_show_preview else "--no-preview",
        ],
        cwd=str(CLIENT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    time.sleep(0.5)
    if STATE.camera_daemon.poll() is not None:
        raise RuntimeError("camera daemon failed to start")




def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def rig_json_from_static(static_path: str) -> dict:
    static = np.load(static_path, allow_pickle=True).item()
    return jsonable(
        {
            "rig_id": static["rig_id"],
            "joints": static["joints"],
            "joint_names": static["joint_names"],
            "parents": static["parents"],
            "rest_translations": static["rest_translations"],
            "rest_rotations_quat": static["rest_rotations_quat"],
            "metric_scale": static["metric_scale"],
            "coordinate_system": static["coordinate_system"],
            "units": static["units"],
        }
    )


def api_key_for_scene(scene) -> str:
    key = scene.tosc_remote_api_key.strip()
    if key:
        return key
    config_path = bpy.path.abspath(scene.tosc_remote_config)
    data = load_config(Path(config_path))
    return str(data["api_key"])






def close_remote_session(meta, server_url, api_key):
    parsed = urlsplit(meta.get("server_url", server_url))
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parsed.hostname, parsed.port, timeout=30)
    try:
        connection.request(
            "DELETE", parsed.path.rstrip("/") + "/v1/sessions/" + meta["session_id"],
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read()
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            payload = None
        # The deployed API uses 409 (rather than 404) for an expired session.
        # Treat only that exact application response as an idempotent deletion.
        absent = (response.status == 409 and isinstance(payload, dict)
                  and payload.get("detail") == "unknown session")
        if response.status not in (200, 202, 204, 404, 410) and not absent:
            detail = body.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Session release failed: HTTP {response.status}: {detail}; session metadata retained")
    finally:
        connection.close()


def close_saved_session(scene) -> None:
    path = Path(ACTIVE_SESSION_PATH)
    if not path.is_file():
        return
    meta = json.loads(path.read_text(encoding="utf-8"))
    config = load_config(Path(bpy.path.abspath(scene.tosc_remote_config)))
    close_remote_session(meta, config["server_url"], api_key_for_scene(scene))
    path.unlink(missing_ok=True)


def remove_previous_character(scene) -> int:
    """Remove the previous FBX character while preserving scene helpers."""
    tagged = {
        obj for obj in scene.objects if bool(obj.get(IMPORTED_OBJECT_TAG, False))
    }
    if tagged:
        targets = tagged
    else:
        targets = set()

    if not targets:
        return 0
    try:
        active = bpy.context.object
        if active is not None and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    for obj in list(targets):
        bpy.data.objects.remove(obj, do_unlink=True)
    return len(targets)


class TOSC_REMOTE_OT_import_fbx(bpy.types.Operator):
    bl_idname = "tosc_remote.import_fbx"
    bl_label = "Import FBX"
    bl_description = "Import an FBX, upload its skeleton, and initialize a CAST session"
    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.fbx", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        scene = context.scene
        path = bpy.path.abspath(self.filepath)
        if STATE.init_thread is not None and STATE.init_thread.is_alive():
            self.report({"WARNING"}, "Previous model is still initializing; wait for it to finish before importing another")
            return {"CANCELLED"}
        if STATE.initializing:
            init_timer()
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"FBX not found: {path}")
            return {"CANCELLED"}

        try:
            prepare_client(scene)
            legacy_settings = prepare_model_switch(scene)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            self.report({"ERROR"}, "Cannot save model settings: " + str(exc))
            return {"CANCELLED"}

        if STATE.process is not None and STATE.process.poll() is None:
            try:
                send_control(scene.tosc_remote_control_port, "stop")
            except Exception:
                pass
            stop_worker()
        if STATE.receiver is not None:
            STATE.receiver.stop()
            STATE.receiver = None
        try:
            close_saved_session(scene)
        except Exception as exc:
            self.report({"ERROR"}, "Cannot release previous session: " + str(exc))
            return {"CANCELLED"}
        removed = remove_previous_character(scene)
        STATE.driver = None
        STATE.asset = None
        STATE.initializing = False
        STATE.remote_ready = False
        STATE.init_result = None
        STATE.init_error = None
        STATE.init_generation += 1
        generation = STATE.init_generation
        Path(DEFAULT_ROOT, "active_rig.json").unlink(missing_ok=True)
        if removed:
            STATE.note(f"removed previous FBX character objects={removed}")

        before = set(scene.objects)
        try:
            bpy.ops.import_scene.fbx(filepath=path)
        except Exception as exc:
            self.report({"ERROR"}, f"FBX import failed: {exc}")
            return {"CANCELLED"}

        imported_objects = set(scene.objects) - before
        for obj in imported_objects:
            obj[IMPORTED_OBJECT_TAG] = True
        armatures = [
            obj
            for obj in imported_objects
            if obj.type == "ARMATURE"
        ]
        if not armatures:
            armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
        if not armatures:
            self.report({"ERROR"}, "no armature found in the FBX")
            return {"CANCELLED"}
        armature = armatures[0]
        try:
            activate_model_settings(scene, path, armature, legacy_settings)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.report({"ERROR"}, "Cannot restore model settings: " + str(exc))
            return {"CANCELLED"}
        requested_convention = getattr(scene, "tosc_remote_rig_convention", "AUTO")
        armature["tosc_rig_convention"] = (fbx_rig_convention(path)
                                          if requested_convention == "AUTO" else requested_convention)
        context.view_layer.objects.active = armature
        armature.select_set(True)

        if scene.camera is None:
            bpy.ops.object.camera_add(
                location=(0.0, -4.0, 1.8),
                rotation=(radians(75.0), 0.0, 0.0),
            )
            scene.camera = context.object

        try:
            runtime_parent = Path(tempfile.gettempdir()) / "tosc_remote_live"
            runtime_parent.mkdir(parents=True, exist_ok=True)
            runtime_dir = tempfile.mkdtemp(prefix="rig_", dir=runtime_parent)
            STATE.asset = build_rig_asset(armature, scene.camera, runtime_dir)
            if STATE.asset["ignored_bones"]:
                STATE.note("Excluded unused isolated roots from inference: " +
                           ", ".join(STATE.asset["ignored_bones"]))
            rig = rig_json_from_static(STATE.asset["static"])
            rig_path = Path(runtime_dir) / "rig.json"
            rig_path.write_text(
                json.dumps(rig, ensure_ascii=True),
                encoding="utf-8",
            )
            # Compatibility snapshot only; the helper reads the immutable per-import file.
            (Path(DEFAULT_ROOT) / "active_rig.json").write_bytes(rig_path.read_bytes())
            result_path = Path(runtime_dir) / "session.json"
            helper = CLIENT_ROOT / "create_session.py"
            STATE.driver = RigDriver(armature, STATE.asset["bind_transforms"])
            STATE.initializing = True
            STATE.remote_ready = False
            STATE.init_result = None
            STATE.init_error = None
            python_path = bpy.path.abspath(scene.tosc_remote_python)
            config_path = bpy.path.abspath(scene.tosc_remote_config)

            helper_env = client_environment(scene)
            connection_config = load_config(Path(config_path))
            session_key = api_key_for_scene(scene)

            def initialize_remote() -> None:
                try:
                    result = subprocess.run(
                        [
                            python_path,
                            str(helper),
                            "--config",
                            config_path,
                            "--rig-file",
                            str(rig_path),
                            "--active-session",
                            str(result_path),
                        ],
                        cwd=str(CLIENT_ROOT),
                        env=helper_env,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=900,
                        check=False,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            (
                                result.stderr
                                or result.stdout
                                or "session helper failed"
                            ).strip()
                        )
                    result_meta = json.loads(result_path.read_text(encoding="utf-8"))
                    if generation != STATE.init_generation:
                        close_remote_session(result_meta, connection_config["server_url"], session_key)
                        return
                    STATE.init_result = result_meta
                except Exception as exc:
                    if generation == STATE.init_generation:
                        STATE.init_error = str(exc)

            STATE.init_thread = threading.Thread(
                target=initialize_remote,
                daemon=True,
            )
            # Timers can be removed after an exception or a file reload.
            # Re-arm completion polling for every new request.
            if not bpy.app.timers.is_registered(init_timer):
                bpy.app.timers.register(init_timer, persistent=True)
            STATE.init_thread.start()
        except Exception as exc:
            self.report({"ERROR"}, f"skeleton initialization failed: {exc}")
            return {"CANCELLED"}

        STATE.note(
            "FBX imported; initializing remote skeleton: "
            f"{armature.name} joints={STATE.asset['joint_count']}"
        )
        return {"FINISHED"}


class TOSC_REMOTE_OT_start_camera(bpy.types.Operator):
    bl_idname = "tosc_remote.start_camera"
    bl_label = "Start Camera"
    bl_description = "Start and warm up the persistent camera daemon"

    def execute(self, context):
        scene = context.scene
        try:
            prepare_client(scene)
            ensure_camera_daemon(scene)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        STATE.camera_ready = False
        STATE.camera_waiting = True
        STATE.camera_wait_started = time.time()
        STATE.note("camera daemon started; waiting for stable frame rate")
        return {"FINISHED"}


class TOSC_REMOTE_OT_start(bpy.types.Operator):
    bl_idname = "tosc_remote.start"
    bl_label = "Start Transfer"
    bl_description = "Start frame transfer after the camera is stable"

    def execute(self, context):
        scene = context.scene
        try:
            prepare_client(scene)
            sync_input_fps(scene)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            self.report({"ERROR"}, "Cannot set input FPS: " + str(exc))
            return {"CANCELLED"}
        if STATE.process is not None and STATE.process.poll() is None:
            STATE.note("transport is already running")
            return {"FINISHED"}

        python = bpy.path.abspath(scene.tosc_remote_python)
        worker = bpy.path.abspath(scene.tosc_remote_worker)
        config = bpy.path.abspath(scene.tosc_remote_config)
        if not os.path.isfile(python):
            self.report({"ERROR"}, f"python not found: {python}")
            return {"CANCELLED"}
        if not os.path.isfile(worker):
            self.report({"ERROR"}, f"worker not found: {worker}")
            return {"CANCELLED"}
        if not os.path.isfile(config):
            self.report({"ERROR"}, f"config not found: {config}")
            return {"CANCELLED"}

        if STATE.asset is None or STATE.driver is None:
            self.report({"ERROR"}, "import an FBX first")
            return {"CANCELLED"}
        # Drain a completed helper even when the periodic callback was delayed.
        if STATE.initializing:
            init_timer()
        if STATE.initializing:
            self.report({"ERROR"}, "remote skeleton is still initializing")
            return {"CANCELLED"}
        if not STATE.remote_ready:
            self.report({"ERROR"}, "remote skeleton is not ready")
            return {"CANCELLED"}
        if not camera_feed_ready():
            self.report(
                {"ERROR"},
                "click Start Camera and wait for stable camera frames",
            )
            return {"CANCELLED"}
        if not Path(ACTIVE_SESSION_PATH).is_file():
            self.report({"ERROR"}, "remote skeleton session is missing; import the FBX again")
            return {"CANCELLED"}
        metadata = json.loads(Path(ACTIVE_SESSION_PATH).read_text(encoding="utf-8"))
        connection_config = load_config(Path(config))
        if (metadata.get("server_url") != connection_config["server_url"] or
                metadata.get("target_mode") != connection_config.get("target_mode", "human")):
            self.report({"ERROR"}, "Connection changed; import the FBX again")
            return {"CANCELLED"}
        if STATE.receiver is None:
            STATE.receiver = PoseReceiver("127.0.0.1", scene.tosc_remote_pose_port)
            STATE.receiver.start()

        env = client_environment(scene)
        command = [
            python,
            worker,
            "--config",
            config,
            "--pose-host",
            "127.0.0.1",
            "--pose-port",
            str(scene.tosc_remote_pose_port),
            "--control-port",
            str(scene.tosc_remote_control_port),
            "--fps",
            str(scene.tosc_remote_fps),
            "--preview" if scene.tosc_remote_show_preview else "--no-preview",
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            STATE.process = subprocess.Popen(
                command,
                cwd=os.path.dirname(worker),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"camera worker failed to start: {exc}")
            return {"CANCELLED"}

        STATE.runtime_dir = os.path.join(tempfile.gettempdir(), "tosc_remote_live")
        STATE.process_reader = threading.Thread(
            target=read_process_output,
            args=(STATE.process,),
            daemon=True,
        )
        STATE.process_reader.start()
        STATE.note(f"camera worker started pid={STATE.process.pid}")
        return {"FINISHED"}


class TOSC_REMOTE_OT_stop(bpy.types.Operator):
    bl_idname = "tosc_remote.stop"
    bl_label = "Stop"
    bl_description = "Close the remote CAST session and stop the camera process"

    def execute(self, context):
        if STATE.initializing and (STATE.init_thread is None or not STATE.init_thread.is_alive()):
            init_timer()
        STATE.init_generation += 1
        STATE.initializing = False
        if STATE.process is not None and STATE.process.poll() is None:
            try:
                send_control(context.scene.tosc_remote_control_port, "stop")
            except Exception:
                pass
        stop_worker()
        stop_camera()
        try:
            close_saved_session(context.scene)
        except Exception as exc:
            self.report({"ERROR"}, "Stream stopped, but session release failed: " + str(exc))
            return {"CANCELLED"}
        STATE.remote_ready = False
        if STATE.receiver is not None:
            STATE.receiver.stop()
            STATE.receiver = None
        STATE.driver = None
        STATE.note("remote session closed; camera process stopped")
        return {"FINISHED"}


class TOSC_REMOTE_OT_pause(bpy.types.Operator):
    bl_idname = "tosc_remote.pause"
    bl_label = "Pause"
    bl_description = "Stop feeding frames but keep the connection and remote cache"

    def execute(self, context):
        if STATE.process is None or STATE.process.poll() is not None:
            self.report({"ERROR"}, "camera worker is not running")
            return {"CANCELLED"}
        try:
            reply = send_control(context.scene.tosc_remote_control_port, "pause")
        except Exception as exc:
            self.report({"ERROR"}, f"pause failed: {exc}")
            return {"CANCELLED"}
        if not reply.get("ok"):
            self.report({"ERROR"}, f"pause failed: {reply}")
            return {"CANCELLED"}
        if STATE.driver is not None:
            STATE.driver.reset_stabilization()
        STATE.note("feed paused; connection and cache preserved")
        return {"FINISHED"}


class TOSC_REMOTE_OT_resume(bpy.types.Operator):
    bl_idname = "tosc_remote.resume"
    bl_label = "Continue"
    bl_description = "Resume feeding frames from the same session and cache"

    def execute(self, context):
        if STATE.process is None or STATE.process.poll() is not None:
            self.report({"ERROR"}, "camera worker is not running")
            return {"CANCELLED"}
        try:
            reply = send_control(context.scene.tosc_remote_control_port, "resume")
        except Exception as exc:
            self.report({"ERROR"}, f"resume failed: {exc}")
            return {"CANCELLED"}
        if not reply.get("ok"):
            self.report({"ERROR"}, f"resume failed: {reply}")
            return {"CANCELLED"}
        if STATE.driver is not None:
            STATE.driver.reset_stabilization()
        STATE.note("feed resumed; session and cache preserved")
        return {"FINISHED"}


class TOSC_REMOTE_OT_reset_cache(bpy.types.Operator):
    bl_idname = "tosc_remote.reset_cache"
    bl_label = "Reset Cache"
    bl_description = (
        "Clear the remote CAST motion cache and visual reference, then restart from seq=0"
    )

    def execute(self, context):
        scene = context.scene
        if STATE.process is None or STATE.process.poll() is not None:
            self.report({"ERROR"}, "camera worker is not running")
            return {"CANCELLED"}
        try:
            reply = send_control(
                scene.tosc_remote_control_port,
                "reset",
            )
        except Exception as exc:
            self.report({"ERROR"}, f"reset failed: {exc}")
            return {"CANCELLED"}
        if not reply.get("ok"):
            self.report({"ERROR"}, f"reset failed: {reply}")
            return {"CANCELLED"}
        if STATE.receiver is not None:
            STATE.receiver.queue.clear()
        if STATE.driver is not None:
            STATE.driver.reset_stabilization()
        STATE.note("remote CAST cache reset requested; next pose starts at seq=0")
        return {"FINISHED"}


def pose_timer() -> float:
    receiver = STATE.receiver
    driver = STATE.driver
    if receiver is not None and driver is not None and receiver.queue:
        scene = bpy.context.scene
        applied = 0
        while receiver.queue and applied < 16:
            frame_index, values = receiver.queue.popleft()
            # Evaluate imported animation first, then write the live pose.
            # Doing this afterwards overwrites the received bone transforms.
            scene.frame_set(driver.applied + 1)
            driver.apply_rot6d(
                values, options=stabilization_options(scene), frame_index=frame_index,
                fps=getattr(scene, "tosc_remote_fps", 24.0),
            )
            applied += 1
        if applied:
            bpy.context.view_layer.update()
    return 0.0


class TOSC_REMOTE_PT_panel(bpy.types.Panel):
    bl_label = "CAST Remote"
    bl_idname = "TOSC_REMOTE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CAST Remote"

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        box = layout.box()
        box.label(text="Runtime", icon="SYSTEM")
        box.prop(scene, "tosc_remote_python", text="Python")
        box.prop(scene, "tosc_remote_worker", text="Worker")
        box.prop(scene, "tosc_remote_config", text="Config")
        box.prop(scene, "tosc_remote_fbx", text="FBX")

        box = layout.box()
        box.label(text="Connection", icon="URL")
        box.prop(scene, "tosc_remote_server_url", text="URL")
        box.prop(scene, "tosc_remote_api_key", text="API Key")
        box.prop(scene, "tosc_remote_target_mode", text="Input Video")
        box.prop(scene, "tosc_remote_pose_port", text="Pose Port")
        box.prop(scene, "tosc_remote_control_port", text="Control Port")
        box.prop(scene, "tosc_remote_fps", text="Input FPS")
        box.prop(scene, "tosc_remote_show_preview", text="Camera Preview")

        box = layout.box()
        box.label(text="Control", icon="PLAY")
        box.operator("tosc_remote.import_fbx", icon="IMPORT")
        box.operator("tosc_remote.start_camera", icon="OUTLINER_OB_CAMERA")
        box.operator("tosc_remote.start", icon="PLAY")
        row = box.row(align=True)
        row.operator("tosc_remote.pause", icon="PAUSE")
        row.operator("tosc_remote.resume", icon="PLAY")
        row = box.row(align=True)
        row.operator("tosc_remote.reset_cache", icon="FILE_REFRESH")
        row.operator("tosc_remote.stop", icon="X")

        box = layout.box()
        box.label(text="Status", icon="INFO")
        box.label(text=STATE.status[:80])
        for line in list(STATE.log_lines)[-6:]:
            box.label(text=line[:90])


class TOSC_REMOTE_PT_pose_settings(bpy.types.Panel):
    bl_label = "高级姿态设置"
    bl_idname = "TOSC_REMOTE_PT_pose_settings"
    bl_parent_id = "TOSC_REMOTE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CAST Remote"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 100

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        box = layout.box()
        box.label(text="Per-FBX settings save automatically")
        if scene.tosc_remote_settings_status:
            box.label(text=scene.tosc_remote_settings_status[:90])
        box.prop(scene, "tosc_remote_rig_convention", text="Skeleton Input (on Import)")
        if STATE.asset is not None:
            box.label(text="Bound input: " + STATE.asset.get("rig_convention", "Legacy"))
        box.label(text="Imported preset: " + scene.tosc_remote_pose_profile)
        box.prop(scene, "tosc_remote_root_mode", text="Root")
        if scene.tosc_remote_root_mode == "PLUS_X":
            box.prop(scene, "tosc_remote_forward_axis", text="Rest Forward")
        box.prop(scene, "tosc_remote_root_offset", text="Root Rotation Offset")
        box.label(text="World axes; Z adjusts heading")
        box.prop(scene, "tosc_remote_fix_bones", text="Fix Bones to Rest")
        row = box.row()
        row.enabled = scene.tosc_remote_fix_bones
        row.prop(scene, "tosc_remote_fixed_bones", text="Bones")
        box.prop(scene, "tosc_remote_fixed_offset_bone", text="Fixed Offset Bone")
        box.prop(scene, "tosc_remote_fixed_offset", text="Fixed Bone Offset")
        box.prop(scene, "tosc_remote_sync_crab_legs", text="Crab Legs: Mirror Right to Left")
        box.prop(scene, "tosc_remote_filter_enabled", text="Rotation Filter")
        row = box.row()
        row.enabled = scene.tosc_remote_filter_enabled
        row.prop(scene, "tosc_remote_filter_ms", text="Smoothing (ms)")
        if STATE.driver is not None:
            status = STATE.driver.stabilization_status.split("; ")
            for line in status:
                box.label(text=line)


CLASSES = (
    TOSC_REMOTE_OT_import_fbx,
    TOSC_REMOTE_OT_start_camera,
    TOSC_REMOTE_OT_start,
    TOSC_REMOTE_OT_pause,
    TOSC_REMOTE_OT_stop,
    TOSC_REMOTE_OT_resume,
    TOSC_REMOTE_OT_reset_cache,
    TOSC_REMOTE_PT_panel,
    TOSC_REMOTE_PT_pose_settings,
)


def register() -> None:
    config = load_config(initialize(default_config_path()))
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    scene = bpy.types.Scene
    scene.tosc_remote_python = StringProperty(
        default=shutil.which("python") or "",
        subtype="FILE_PATH",
    )
    scene.tosc_remote_worker = StringProperty(
        default=str(CLIENT_ROOT / "transport.py"),
        subtype="FILE_PATH",
    )
    scene.tosc_remote_config = StringProperty(
        default=str(default_config_path()),
        subtype="FILE_PATH",
    )
    scene.tosc_remote_fbx = StringProperty(
        default="",
        subtype="FILE_PATH",
    )
    scene.tosc_remote_server_url = StringProperty(
        default="", description="Blank uses server_url in Config",
    )
    scene.tosc_remote_api_key = StringProperty(
        default="", options={"SKIP_SAVE"},
        subtype="PASSWORD",
    )
    scene.tosc_remote_target_mode = EnumProperty(
        items=[("human", "Human video", "Track the person in the camera image"),
               ("generic", "Generic video", "Use the full frame for a non-human subject")],
        default=config.get("target_mode", "human"),
    )
    scene.tosc_remote_pose_port = IntProperty(
        default=int(config.get("pose_port", 5599)),
        min=1024,
        max=65535,
    )
    scene.tosc_remote_control_port = IntProperty(
        default=int(config.get("control_port", 5601)),
        min=1024,
        max=65535,
    )
    scene.tosc_remote_fps = FloatProperty(
        default=24.0,
        min=1.0,
        max=120.0,
        update=update_input_fps,
        description="Camera input and inference submission rate; applied live and saved per FBX",
    )
    scene.tosc_remote_show_preview = BoolProperty(
        default=False,
        update=update_model_settings,
    )
    scene.tosc_remote_rig_convention = EnumProperty(
        name="Skeleton Input",
        items=[("AUTO", "Auto", "Recognize verified Zoo FBX files by content; otherwise preserve FBX local axes"),
               ("ZOO_RESET", "Zoo / BVH reset axes", "Shared source joint axes and skeleton-length normalization; reimport to apply"),
               ("FBX_LOCAL", "FBX / Mobj local axes", "Preserve child rest axes, reset root and normalize rest mesh; reimport to apply")],
        default="AUTO",
        update=update_model_settings,
    )
    scene.tosc_remote_pose_profile = StringProperty(default="Custom settings")
    scene.tosc_remote_settings_key = StringProperty(default="", options={"HIDDEN"})
    scene.tosc_remote_settings_source = StringProperty(default="", options={"HIDDEN"})
    scene.tosc_remote_settings_status = StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})
    scene.tosc_remote_root_mode = EnumProperty(
        name="Root", items=[("PREDICTED", "Predicted", "Use predicted root rotation"),
                            ("REST", "Fixed Rest", "Keep the bound rest orientation"),
                            ("PLUS_X", "Fixed +X", "Keep the rest posture facing world +X")],
        default="PREDICTED",
        update=update_model_settings,
    )
    scene.tosc_remote_forward_axis = EnumProperty(
        name="Rest Forward", items=[("AUTO", "Auto (humanoid feet)", "Infer from foot-to-toe direction; unknown skeletons keep rest orientation"),
                                    ("NEG_Y", "-Y", "Rest character faces world -Y"),
                                    ("POS_Y", "+Y", "Rest character faces world +Y"),
                                    ("NEG_X", "-X", "Rest character faces world -X"),
                                    ("POS_X", "+X", "Rest character faces world +X")],
        default="AUTO",
        update=update_model_settings,
    )
    scene.tosc_remote_root_offset = FloatVectorProperty(
        name="Root Rotation Offset", size=3, subtype="EULER", unit="ROTATION",
        default=(0.0, 0.0, 0.0), update=update_root_offset,
        description="XYZ Euler offset in Blender world axes, after root mode and filtering; Z adjusts heading. Zero resets the offset",
    )
    scene.tosc_remote_fix_bones = BoolProperty(
        default=False, description="Optional: hold only the listed non-root joints at their rest rotations",
        update=update_model_settings,
    )
    scene.tosc_remote_fixed_bones = StringProperty(
        default="Neck,Neck1,Head",
        description="Non-root bone names, comma-separated; namespaces optional, missing names skipped",
        update=update_model_settings,
    )
    scene.tosc_remote_fixed_offset_bone = StringProperty(
        default="", update=update_model_settings,
        description="One fixed bone to offset from rest; must also be included in Bones",
    )
    scene.tosc_remote_fixed_offset = FloatVectorProperty(
        name="Fixed Bone Offset", size=3, subtype="EULER", unit="ROTATION",
        default=(0.0, 0.0, 0.0), update=update_model_settings,
        description="XYZ offset in the FBX's Blender-world rest axes; follows the parent and affects descendants once",
    )
    scene.tosc_remote_filter_enabled = BoolProperty(
        default=True, description="Optional: smooth rotations locally; independent of bone fixing and root offset",
        update=update_model_settings,
    )
    scene.tosc_remote_sync_crab_legs = BoolProperty(
        default=False, update=update_model_settings,
        description="Use only the right middle leg: synchronize the right three legs and mirror them to the left; preserve rest spread",
    )
    scene.tosc_remote_filter_ms = FloatProperty(
        default=60.0, min=0.0, max=300.0,
        description="Quaternion low-pass time constant; more smoothing adds more motion delay",
        update=update_model_settings,
    )
    if restore_saved_model_settings not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(restore_saved_model_settings)
    # Blender restricts bpy.data while enabling add-ons. Scene access must wait
    # until registration completes, including startup with an existing .blend.
    if not bpy.app.timers.is_registered(restore_saved_model_settings):
        bpy.app.timers.register(restore_saved_model_settings, first_interval=0.0)
    if not bpy.app.timers.is_registered(camera_wait_timer):
        bpy.app.timers.register(camera_wait_timer, persistent=True)
    if not bpy.app.timers.is_registered(pose_timer):
        bpy.app.timers.register(pose_timer, persistent=True)
    if not bpy.app.timers.is_registered(init_timer):
        bpy.app.timers.register(init_timer, persistent=True)


def unregister() -> None:
    STATE.init_generation += 1
    if bpy.app.timers.is_registered(restore_saved_model_settings):
        bpy.app.timers.unregister(restore_saved_model_settings)
    if restore_saved_model_settings in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(restore_saved_model_settings)
    if STATE.process is not None and STATE.process.poll() is None:
        try:
            send_control(bpy.context.scene.tosc_remote_control_port, "stop")
        except Exception:
            pass
    stop_worker()
    stop_camera()
    try:
        close_saved_session(bpy.context.scene)
    except Exception as exc:
        STATE.note("Session cleanup failed: " + str(exc))
    if STATE.receiver is not None:
        STATE.receiver.stop()
        STATE.receiver = None
    if bpy.app.timers.is_registered(camera_wait_timer):
        bpy.app.timers.unregister(camera_wait_timer)
    if bpy.app.timers.is_registered(pose_timer):
        bpy.app.timers.unregister(pose_timer)
    if bpy.app.timers.is_registered(init_timer):
        bpy.app.timers.unregister(init_timer)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    scene = bpy.types.Scene
    for name in (
        "tosc_remote_python",
        "tosc_remote_worker",
        "tosc_remote_config",
        "tosc_remote_fbx",
        "tosc_remote_server_url",
        "tosc_remote_api_key",
        "tosc_remote_target_mode",
        "tosc_remote_pose_port",
        "tosc_remote_control_port",
        "tosc_remote_fps",
        "tosc_remote_show_preview",
        "tosc_remote_root_mode",
        "tosc_remote_rig_convention",
        "tosc_remote_pose_profile",
        "tosc_remote_settings_key",
        "tosc_remote_settings_source",
        "tosc_remote_settings_status",
        "tosc_remote_forward_axis",
        "tosc_remote_root_offset",
        "tosc_remote_fix_bones",
        "tosc_remote_fixed_bones",
        "tosc_remote_fixed_offset_bone",
        "tosc_remote_fixed_offset",
        "tosc_remote_filter_enabled",
        "tosc_remote_sync_crab_legs",
        "tosc_remote_filter_ms",
    ):
        if hasattr(scene, name):
            delattr(scene, name)


if __name__ == "__main__":
    register()
