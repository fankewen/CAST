"""CAST Live - realtime video-to-motion driving inside Blender.

Panel flow (View3D > Sidebar > CAST Live):

1. ``Load Weights``          start the engine process and load every model in the
                             Weights box: the CAST checkpoint and SAM 2.1.  The RGB
                             backbone is built from the CAST config (target + repo +
                             weights), so it is not a separate input.  No skeleton is
                             involved yet, which is why this comes before the
                             character is picked.
2. ``Import Character``      import the character with Blender's own importers
                             (FBX / GLB / blend) and, as soon as it lands,
                             precompute the rig and warm the streaming step up.
3. ``Run Video``             pick a video; SAM2 segmentation runs in the engine,
                             then frames are fed uniformly at ``fps`` (the KV cache
                             grows for the first ``cache_length`` frames and is a
                             ring buffer after that) and every pose is written into
                             the armature the moment it exists.

Torch/SAM2 never load inside Blender: the panel talks to ``blender/cast_engine_server.py``
over a local socket and owns only the UI, the rig conversion and the pose
application.
"""

bl_info = {
    "name": "CAST Live",
    "author": "CAST demo",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > CAST Live",
    "description": "Realtime video -> causal motion model -> armature/mesh",
    "category": "Animation",
}

import json
import os
import socket
import struct
import subprocess
import threading
import time
from collections import deque

import bpy
import numpy as np
from bpy.props import FloatProperty, IntProperty, StringProperty
from mathutils import Matrix, Vector

POSE_HEADER = struct.Struct("<II")
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
    # *in the camera frame* (verified against the shipped Mousey cache: 1.7220 m
    # diagonal -> 1.161439), so measure it there and not in world space.
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


# --------------------------------------------------------------- pose receiver
class PoseReceiver:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.queue = deque(maxlen=4096)
        self.server = None
        self.connection = None
        self.thread = None
        self.stop = threading.Event()
        self.received = 0

    def start(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

    def _accept_loop(self) -> None:
        while not self.stop.is_set():
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            buffer = bytearray()
            while not self.stop.is_set():
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
                    payload = joint_count * 6 * 4
                    if len(buffer) < POSE_HEADER.size + payload:
                        break
                    values = struct.unpack_from(f"<{joint_count * 6}f", buffer, POSE_HEADER.size)
                    del buffer[: POSE_HEADER.size + payload]
                    self.received += 1
                    self.queue.append((self.received, np.asarray(values, dtype=np.float64).reshape(joint_count, 6)))
            try:
                connection.close()
            except OSError:
                pass
            self.connection = None

    def stop_server(self) -> None:
        self.stop.set()
        try:
            if self.server is not None:
                self.server.close()
        except OSError:
            pass
        self.server = None


# ------------------------------------------------------------------ engine link
class EngineClient:
    """One JSON command per line to the engine process."""

    def __init__(self):
        self.socket = None
        self.pending = None
        self.reply = None
        self.busy = False

    def connect(self, host: str, port: int, timeout: float = 20.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.socket = socket.create_connection((host, port), timeout=5.0)
                # Loading the checkpoint or compiling the streaming step takes
                # minutes, so the reply wait must not inherit the connect timeout.
                self.socket.settimeout(1800.0)
                return True
            except OSError:
                time.sleep(0.5)
        return False

    def send(self, request: dict, on_done=None) -> None:
        if self.busy:
            raise RuntimeError("engine is busy")
        self.busy = True
        self.reply = None

        def worker():
            try:
                self.socket.sendall((json.dumps(request) + "\n").encode("utf-8"))
                buffer = b""
                while b"\n" not in buffer:
                    chunk = self.socket.recv(1 << 16)
                    if not chunk:
                        raise RuntimeError("engine closed the connection")
                    buffer += chunk
                self.reply = json.loads(buffer.split(b"\n", 1)[0].decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                self.reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            finally:
                self.busy = False
                if on_done is not None:
                    on_done(self.reply)

        threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- orchestration
class CastState:
    def __init__(self):
        self.engine_process = None
        self.client = EngineClient()
        self.receiver = None
        self.driver = None
        self.status = "idle"
        self.detail = ""
        self.log_lines = deque(maxlen=200)
        self.asset = None
        self.armature = None
        self.segmenter = None
        # Strict step machine: "idle" | "loading_weights" | "warm_up".
        self.phase = "idle"
        self.weights_loaded = False
        self.rig_ready = False

    def note(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        self.log_lines.append(line)
        self.status = message
        print("CAST " + line, flush=True)


STATE = CastState()


def engine_python(scene) -> str:
    return scene.cast_python or "python"


def engine_script(scene) -> str:
    return scene.cast_engine_script or os.path.join(
        scene.cast_repo, "blender", "cast_engine_server.py"
    )


def start_engine(scene) -> None:
    if STATE.engine_process is not None and STATE.engine_process.poll() is None:
        STATE.note("engine already running")
        return
    runtime = os.path.join(scene.cast_repo, "runtime")
    os.makedirs(runtime, exist_ok=True)
    log_path = os.path.join(runtime, "engine_process.log")
    env = dict(os.environ)
    env["PYTHONPATH"] = scene.cast_repo
    env["CUDA_VISIBLE_DEVICES"] = scene.cast_gpu
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    handle = open(log_path, "ab")
    STATE.engine_process = subprocess.Popen(
        [
            engine_python(scene),
            engine_script(scene),
            "--repo", scene.cast_repo,
            "--runtime-dir", runtime,
            "--control-port", str(scene.cast_control_port),
            "--cache-length", str(scene.cast_cache_length),
            "--amp-dtype", scene.cast_amp_dtype,
            "--seg-image-size", str(scene.cast_seg_image_size),
        ],
        cwd=scene.cast_repo,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    STATE.note(f"engine process started pid={STATE.engine_process.pid}")


# ------------------------------------------------------------------- operators
class CAST_OT_load_weights(bpy.types.Operator):
    bl_idname = "cast.load_weights"
    bl_label = "1. Load Weights"
    bl_description = (
        "Start the engine and load the CAST checkpoint and SAM 2.1 from the "
        "Weights box; the RGB backbone comes from the CAST config"
    )

    def execute(self, context):
        scene = context.scene
        if STATE.phase == "loading_weights":
            self.report({"ERROR"}, "weights are already loading")
            return {"CANCELLED"}
        if STATE.phase != "idle":
            self.report({"ERROR"}, "wait for the current step to finish")
            return {"CANCELLED"}
        if STATE.weights_loaded:
            self.report(
                {"ERROR"},
                "weights are already loaded; press 'Shutdown Engine' to load again",
            )
            return {"CANCELLED"}
        missing = [
            label
            for label, path in (
                ("CAST cfg", scene.cast_config),
                ("CAST ckpt", scene.cast_checkpoint),
                ("SAM ckpt", scene.cast_sam_ckpt),
            )
            if not os.path.exists(bpy.path.abspath(path))
        ]
        # SAM cfg is a config *name* resolved inside the SAM2 install, not a path.
        if not scene.cast_sam_cfg.strip():
            missing.append("SAM cfg")
        if missing:
            self.report(
                {"ERROR"},
                "fill in the Weights box first: %s not found" % ", ".join(missing),
            )
            return {"CANCELLED"}
        start_engine(scene)
        if STATE.client.socket is None and not STATE.client.connect("127.0.0.1", scene.cast_control_port):
            self.report({"ERROR"}, "engine did not open its control port")
            return {"CANCELLED"}
        STATE.phase = "loading_weights"
        STATE.note("loading CAST + SAM (the RGB backbone comes from the config) ...")

        def done(reply):
            STATE.phase = "idle"
            if not reply.get("ok"):
                STATE.note(f"weights failed: {reply.get('error')}")
                return
            segmenter = reply.get("segmenter") or {}
            STATE.segmenter = segmenter
            STATE.weights_loaded = True
            STATE.note(
                "models ready: CAST image_size=%s, SAM %s@%s"
                % (
                    reply.get("image_size"),
                    segmenter.get("model", "?"),
                    segmenter.get("image_size", "?"),
                )
            )

        STATE.client.send(
            {
                "cmd": "load_weights",
                "config": scene.cast_config,
                "checkpoint": scene.cast_checkpoint,
                "sam_cfg": scene.cast_sam_cfg,
                "sam_ckpt": scene.cast_sam_ckpt,
            },
            on_done=done,
        )
        return {"FINISHED"}


class CAST_OT_import_character(bpy.types.Operator):
    bl_idname = "cast.import_character"
    bl_label = "2. Import Character"
    bl_description = (
        "Import an FBX / GLB / .blend with Blender's own importers, then precompute "
        "the rig and warm the streaming step up, all in this click"
    )
    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.fbx;*.glb;*.gltf;*.blend", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if STATE.phase != "idle":
            self.report({"ERROR"}, "wait for the current step to finish")
            return {"CANCELLED"}
        if not STATE.weights_loaded:
            self.report({"ERROR"}, "press '1. Load Weights' first")
            return {"CANCELLED"}
        path = bpy.path.abspath(self.filepath)
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"not a file: {path}")
            return {"CANCELLED"}
        suffix = os.path.splitext(path)[1].lower()
        existing_objects = {obj.as_pointer() for obj in bpy.context.scene.objects}
        if suffix == ".blend":
            bpy.ops.wm.open_mainfile(filepath=path)
        elif suffix == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path)
        else:
            bpy.ops.import_scene.gltf(filepath=path)
        arms = [
            obj for obj in bpy.context.scene.objects
            if obj.type == "ARMATURE"
            and (suffix == ".blend" or obj.as_pointer() not in existing_objects)
        ]
        if len(arms) != 1:
            self.report(
                {"ERROR"},
                f"expected one armature in the imported file, found {len(arms)}",
            )
            return {"CANCELLED"}
        bpy.context.view_layer.objects.active = arms[0]
        STATE.note(f"loaded {os.path.basename(path)} ({len(arms[0].data.bones)} bones)")
        # The character is in the scene, so the rig can be adapted right away:
        # there is nothing to prepare this for beforehand.
        if not adapt_rig_and_compile(self):
            return {"CANCELLED"}
        return {"FINISHED"}


def adapt_rig_and_compile(operator) -> bool:
    """Precompute the rig, then compile and warm up the streaming step.

    It has nothing to work on until a character is in the scene, so it runs as
    soon as the character import lands instead of asking for another click.
    """
    scene = bpy.context.scene
    armature = bpy.context.active_object
    if armature is None or armature.type != "ARMATURE":
        armature = next((obj for obj in scene.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        operator.report({"ERROR"}, "no armature in the scene")
        return False
    camera = scene.camera
    if camera is None:
        operator.report({"ERROR"}, "no active camera (Ctrl+Numpad0)")
        return False

    STATE.rig_ready = False
    out_dir = os.path.join(scene.cast_repo, "runtime", armature.name)
    try:
        STATE.asset = build_rig_asset(armature, camera, out_dir)
    except Exception as exc:  # noqa: BLE001
        operator.report({"ERROR"}, f"rig adaptation failed: {exc}")
        return False
    STATE.armature = armature
    STATE.note(
        "precomputing the rig: %d joints, metric_scale=%.4f"
        % (STATE.asset["joint_count"], STATE.asset["metric_scale"])
    )
    if STATE.client.socket is None and not STATE.client.connect("127.0.0.1", scene.cast_control_port):
        operator.report({"ERROR"}, "engine not reachable; press '1. Load Weights' first")
        return False
    # Nothing else may start while this runs: the warm-up owns the engine.
    STATE.phase = "warm_up"
    STATE.note("warming up the streaming step (~1 min) ...")

    def done(reply):
        STATE.phase = "idle"
        if reply.get("ok"):
            STATE.rig_ready = True
            STATE.note(
                "warm-up done (joints=%s, cache=%s)"
                % (reply.get("joints"), reply.get("cache_length"))
            )
        else:
            STATE.note(f"warm-up failed: {reply.get('error')}")

    STATE.client.send(
        {
            "cmd": "load_rig",
            "static": STATE.asset["static"],
            "bind_transforms": STATE.asset["bind_transforms"],
        },
        on_done=done,
    )
    return True


class CAST_OT_run_video(bpy.types.Operator):
    bl_idname = "cast.run_video"
    bl_label = "3. Run Video"
    bl_description = (
        "Pick a clip and feed it frame by frame; while the feed is paused this "
        "loads a new clip and waits there until you press Continue"
    )
    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.mp4;*.mov;*.mkv;*.avi", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        scene = context.scene
        if STATE.phase != "idle":
            self.report({"ERROR"}, "wait for the warm-up to finish")
            return {"CANCELLED"}
        if not STATE.rig_ready:
            self.report({"ERROR"}, "import a character first ('2. Import Character')")
            return {"CANCELLED"}
        video = bpy.path.abspath(self.filepath)
        if not os.path.isfile(video):
            self.report({"ERROR"}, f"not a file: {video}")
            return {"CANCELLED"}

        armature = STATE.armature
        try:
            rig_available = (
                armature is not None
                and scene.objects.get(armature.name) == armature
            )
        except ReferenceError:
            rig_available = False
        if not rig_available:
            STATE.rig_ready = False
            self.report({"ERROR"}, "the prepared armature is no longer in this scene; import it again")
            return {"CANCELLED"}
        try:
            STATE.driver = RigDriver(armature, STATE.asset["bind_transforms"])
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"driver init failed: {exc}")
            return {"CANCELLED"}
        if STATE.receiver is None:
            STATE.receiver = PoseReceiver("127.0.0.1", scene.cast_pose_port)
            STATE.receiver.start()
        STATE.note(f"playback armed on port {scene.cast_pose_port}")

        if STATE.client.socket is None and not STATE.client.connect("127.0.0.1", scene.cast_control_port):
            self.report({"ERROR"}, "engine not reachable")
            return {"CANCELLED"}
        STATE.client.send(
            {
                "cmd": "run_video",
                "video": video,
                "fps": float(scene.cast_fps),
                "after_end": scene.cast_after_end,
                "pose_port": int(scene.cast_pose_port),
            },
            on_done=lambda reply: STATE.note(
                (
                    "loaded "
                    + os.path.basename(video)
                    + " (paused, press Continue)"
                    if reply.get("paused")
                    else "feeding " + os.path.basename(video)
                )
                if reply.get("ok")
                else f"run failed: {reply.get('error')}"
            ),
        )
        return {"FINISHED"}


class CAST_OT_pause(bpy.types.Operator):
    bl_idname = "cast.pause"
    bl_label = "Pause"
    bl_description = "Hold the feed at the current frame; the mesh stops where it is"

    def execute(self, context):
        if STATE.client.socket is None:
            self.report({"ERROR"}, "engine not reachable")
            return {"CANCELLED"}
        STATE.note("pausing the feed ...")

        def done(reply):
            if reply.get("ok"):
                STATE.note(f"paused at frame {reply.get('frames', '?')}")
            else:
                STATE.note(f"pause failed: {reply.get('error')}")

        STATE.client.send({"cmd": "pause_feed"}, on_done=done)
        return {"FINISHED"}


class CAST_OT_resume(bpy.types.Operator):
    bl_idname = "cast.resume"
    bl_label = "Continue"
    bl_description = "Carry on feeding the frames after the one it held at"

    def execute(self, context):
        if STATE.client.socket is None:
            self.report({"ERROR"}, "engine not reachable")
            return {"CANCELLED"}
        STATE.note("resuming the feed ...")

        def done(reply):
            if reply.get("ok"):
                STATE.note(f"feeding again from frame {reply.get('frames', '?')}")
            else:
                STATE.note(f"resume failed: {reply.get('error')}")

        STATE.client.send({"cmd": "resume_feed"}, on_done=done)
        return {"FINISHED"}


class CAST_OT_stop(bpy.types.Operator):
    bl_idname = "cast.stop"
    bl_label = "Stop"
    bl_description = "Stop the feed (the model stays loaded)"

    def execute(self, context):
        if STATE.client.socket is not None:
            STATE.client.send({"cmd": "stop_feed"})
        STATE.note("feed stopped")
        return {"FINISHED"}


class CAST_OT_shutdown(bpy.types.Operator):
    bl_idname = "cast.shutdown"
    bl_label = "Shutdown Engine"
    bl_description = "Stop the feed and terminate the engine process"

    def execute(self, context):
        if STATE.client.socket is not None:
            STATE.client.send({"cmd": "shutdown"})
        if STATE.receiver is not None:
            STATE.receiver.stop_server()
            STATE.receiver = None
        # The engine process is gone, so the step machine starts over.
        STATE.phase = "idle"
        STATE.weights_loaded = False
        STATE.rig_ready = False
        STATE.asset = None
        STATE.segmenter = None
        STATE.client.socket = None
        STATE.client.busy = False
        STATE.note("engine shutdown requested")
        return {"FINISHED"}


# ---------------------------------------------------------------------- timers
def pose_timer() -> float:
    receiver = STATE.receiver
    driver = STATE.driver
    if receiver is not None and driver is not None and receiver.queue:
        scene = bpy.context.scene
        applied = 0
        while receiver.queue and applied < 32:
            _, values = receiver.queue.popleft()
            driver.apply_rot6d(values)
            scene.frame_set(driver.applied)
            applied += 1
        if applied:
            bpy.context.view_layer.update()
            if driver.applied % 30 == 0:
                STATE.note(
                    f"applied {driver.applied} frames, timeline={scene.frame_current}, "
                    f"queue={len(receiver.queue)}"
                )
    return 0.0


def status_timer() -> float:
    client = STATE.client
    if client is not None and client.reply is not None:
        reply = client.reply
        client.reply = None
        if not reply.get("ok"):
            STATE.note(f"engine error: {reply.get('error')}")
    return 0.25


# ------------------------------------------------------------------- UI panel
def labeled_row(box, scene, prop: str, label: str, factor: float = 0.42) -> None:
    """One line: plain-text label on the left, the widget without a label right.

    The label is always drawn by the layout, never by the widget itself: letting
    ``prop(text=...)`` draw it made the numeric boxes put the label *inside* the
    field whenever it did not fit, which looked different from the string rows.
    """
    split = box.split(factor=factor, align=True)
    split.label(text=label)
    split.prop(scene, prop, text="")


class CAST_PT_panel(bpy.types.Panel):
    bl_label = "CAST Live"
    bl_idname = "CAST_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CAST Live"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="Engine", icon="SYSTEM")
        labeled_row(box, scene, "cast_repo", "Repo:")
        labeled_row(box, scene, "cast_python", "Python:")
        labeled_row(box, scene, "cast_gpu", "GPU:")
        labeled_row(box, scene, "cast_control_port", "Control port:")
        labeled_row(box, scene, "cast_pose_port", "Pose port:")

        box = layout.box()
        box.label(text="Weights", icon="FILE_FOLDER")
        labeled_row(box, scene, "cast_config", "CAST cfg:")
        labeled_row(box, scene, "cast_checkpoint", "CAST ckpt:")
        labeled_row(box, scene, "cast_sam_cfg", "SAM cfg:")
        labeled_row(box, scene, "cast_sam_ckpt", "SAM ckpt:")

        box = layout.box()
        box.label(text="Steps", icon="PLAY")
        # While a step runs, the ones that would fight it for the engine are greyed
        # out: no second load, and no Run Video until the warm-up has finished.
        idle = STATE.phase == "idle"
        row = box.row()
        row.enabled = idle and not STATE.weights_loaded
        row.operator("cast.load_weights", icon="IMPORT")
        row = box.row()
        row.enabled = idle and STATE.weights_loaded
        row.operator("cast.import_character", icon="FILE_FOLDER")
        row = box.row()
        row.enabled = idle and STATE.rig_ready
        row.operator("cast.run_video", icon="SEQUENCE")
        row = box.row(align=True)
        row.operator("cast.pause", icon="PAUSE")
        row.operator("cast.resume", icon="PLAY")
        row = box.row(align=True)
        row.operator("cast.stop", icon="X")
        row.operator("cast.shutdown", icon="QUIT")

        box = layout.box()
        box.label(text="Feed", icon="TIME")
        labeled_row(box, scene, "cast_fps", "FPS:")
        labeled_row(box, scene, "cast_after_end", "After end:")
        labeled_row(box, scene, "cast_seg_image_size", "SAM res:")
        labeled_row(box, scene, "cast_amp_dtype", "AMP:")

        box = layout.box()
        box.label(text="Status", icon="INFO")
        box.label(text=STATE.status[:64])
        if STATE.asset is not None:
            box.label(text=f"joints {STATE.asset['joint_count']}")
        if STATE.segmenter and STATE.segmenter.get("loaded"):
            box.label(
                text="SAM2.1 %s@%s"
                % (STATE.segmenter.get("model"), STATE.segmenter.get("image_size"))
            )
        if STATE.receiver is not None:
            box.label(text=f"poses received {STATE.receiver.received}")
            box.label(text=f"applied {STATE.driver.applied if STATE.driver else 0}")
        for line in list(STATE.log_lines)[-6:]:
            box.label(text=line[:64])


# ------------------------------------------------------------------ properties
CLASSES = (
    CAST_OT_load_weights,
    CAST_OT_import_character,
    CAST_OT_run_video,
    CAST_OT_pause,
    CAST_OT_resume,
    CAST_OT_stop,
    CAST_OT_shutdown,
    CAST_PT_panel,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    scene = bpy.types.Scene
    scene.cast_repo = StringProperty(
        default="/path/to/CAST",
        subtype="DIR_PATH",
        name="Repo",
        description="repository root the engine runs from",
    )
    scene.cast_python = StringProperty(
        default="python",
        subtype="FILE_PATH",
        name="Python",
        description="interpreter that has torch; the engine is started with it",
    )
    scene.cast_engine_script = StringProperty(
        default="",
        subtype="FILE_PATH",
        name="Engine script",
        description="empty means <repo>/blender/cast_engine_server.py",
    )
    scene.cast_config = StringProperty(
        default="configs/experiment/CAST_L_dinov3_causal.yaml",
        subtype="FILE_PATH",
        name="CAST cfg",
        description="CAST model config the checkpoint was trained with",
    )
    scene.cast_checkpoint = StringProperty(
        default="/path/to/checkpoints/CAST_L_dinov3_causal/"
        "video2motion_ckpt_best.pt",
        subtype="FILE_PATH",
        name="CAST ckpt",
        description="CAST weights; the RGB backbone is NOT inside this file",
    )
    scene.cast_sam_cfg = StringProperty(
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        name="SAM cfg",
        description="SAM 2.1 config name, resolved inside the SAM2 install",
    )
    scene.cast_sam_ckpt = StringProperty(
        default="/path/to/sam2.1_hiera_tiny.pt",
        subtype="FILE_PATH",
        name="SAM ckpt",
        description="SAM 2.1 weights",
    )
    scene.cast_gpu = StringProperty(default="7", name="GPU")
    scene.cast_control_port = IntProperty(default=5598, name="Control port")
    scene.cast_pose_port = IntProperty(default=5599, name="Pose port")
    scene.cast_fps = FloatProperty(default=30.0, min=1.0, max=120.0, name="FPS")
    scene.cast_after_end = bpy.props.EnumProperty(
        name="After end",
        items=(("repeat-last", "Repeat last frame", "keep feeding the final frame"),
               ("loop", "Loop video", "restart the video")),
        default="repeat-last",
    )
    scene.cast_amp_dtype = bpy.props.EnumProperty(
        name="AMP", items=(("bf16", "bf16", ""), ("fp16", "fp16", "")), default="bf16"
    )
    scene.cast_cache_length = IntProperty(default=81, name="Cache length")
    scene.cast_seg_image_size = IntProperty(
        default=512, min=128, max=1024, name="SAM res",
        description="encoder resolution; 1024 is official, lower is faster",
    )
    bpy.app.timers.register(pose_timer, persistent=True)
    bpy.app.timers.register(status_timer, persistent=True)
    STATE.note("CAST Live ready")


def unregister() -> None:
    for timer in (pose_timer, status_timer):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
