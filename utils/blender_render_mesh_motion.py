#!/usr/bin/env python3
"""Render a predicted motion on an extracted, bind-aware mesh asset."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

try:
    import bpy
    from mathutils import Quaternion, Vector
except ModuleNotFoundError:
    bpy = None
    Quaternion = None
    Vector = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.mesh_skinning import (
    blender_camera_calibration_from_opencv,
    deformation_deltas,
    linear_blend_skinning,
    load_mesh_asset,
    opencv_camera_points_to_blender,
)


def parse_args():
    arguments = (
        sys.argv[sys.argv.index("--") + 1 :]
        if "--" in sys.argv
        else sys.argv[1:]
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--lighting-preset", choices=("zoo", "mixamo"), required=True)
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--light-energies", type=float, nargs="+")
    parser.add_argument("--world-strength", type=float, default=None)
    parser.add_argument("--camera-focal-scale", type=float, default=1.0)
    parser.add_argument("--camera-shift-y", type=float, default=0.0)
    parser.add_argument("--camera-advance", type=float, default=0.0)
    parser.add_argument(
        "--allow-root-translation",
        action="store_true",
        help="Render metric predicted root translation with an oblique orthographic camera.",
    )
    return parser.parse_args(arguments)


def launch_in_blender(arguments: list[str]) -> None:
    """Relaunch this file in Blender without a separate shell wrapper."""
    blender = os.environ.get("BLENDER_EXECUTABLE")
    if blender is None:
        blender_root = Path(
            os.environ.get(
                "BLENDER_ROOT",
                "/path/to/blender-4.5.12-linux-x64",
            )
        )
        blender = str(blender_root / "blender")
    blender_path = Path(blender).expanduser()
    if not blender_path.is_file():
        raise FileNotFoundError(
            "Blender executable not found. Set BLENDER_EXECUTABLE or "
            f"BLENDER_ROOT; resolved path: {blender_path}"
        )

    environment = os.environ.copy()
    library_root = environment.get(
        "BLENDER_LIBRARY_ROOT",
        "/path/to/blender-libs/root/usr/lib/x86_64-linux-gnu",
    )
    current_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{library_root}:{current_library_path}"
        if current_library_path
        else library_root
    )
    command = [
        str(blender_path),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        *arguments,
    ]
    subprocess.run(command, check=True, env=environment)


def look_at(camera, target):
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, energy, size):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    return obj


def main():
    args = parse_args()
    preset_energies = {
        "zoo": (1000.0, 500.0, 500.0, 500.0, 700.0),
        "mixamo": (44.0, 28.6, 44.0),
    }
    configured_energies = (
        tuple(args.light_energies)
        if args.light_energies is not None
        else preset_energies[args.lighting_preset]
    )
    if len(configured_energies) != len(preset_energies[args.lighting_preset]):
        raise ValueError(
            f"{args.lighting_preset} requires {len(preset_energies[args.lighting_preset])} light energies"
        )
    if any(not math.isfinite(energy) or energy < 0 for energy in configured_energies):
        raise ValueError("Light energies must be finite and non-negative")
    args.mesh_path = args.mesh_path.expanduser().resolve()
    args.motion = args.motion.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if bpy is None:
        launch_in_blender(sys.argv[1:])
        return
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if not np.isfinite(args.camera_focal_scale) or args.camera_focal_scale <= 0:
        raise ValueError("camera focal scale must be positive and finite")
    if not np.isfinite(args.camera_shift_y):
        raise ValueError("camera shift Y must be finite")
    if not np.isfinite(args.camera_advance) or args.camera_advance < 0:
        raise ValueError("camera advance must be finite and non-negative")
    if not np.isfinite(args.exposure):
        raise ValueError("exposure must be finite")
    if args.world_strength is not None and (not np.isfinite(args.world_strength) or args.world_strength < 0):
        raise ValueError("World strength must be finite and non-negative")
    asset = load_mesh_asset(args.mesh_path)
    with np.load(args.motion, allow_pickle=False) as archive:
        motion = {key: np.array(archive[key], copy=True) for key in archive.files}
    if str(asset["skeleton_hash"]) != str(motion["skeleton_hash"]):
        raise ValueError("Mesh and motion skeleton hashes differ")
    if not np.array_equal(asset["joint_names"], motion["joint_names"]):
        raise ValueError("Mesh and motion joint names differ")
    if not np.array_equal(asset["parents"], motion["parents"]):
        raise ValueError("Mesh and motion hierarchies differ")
    has_predicted_root_translation = bool(motion["has_predicted_root_translation"])
    root_translation = np.asarray(motion["pred_root_translation"], dtype=np.float64)
    has_nonzero_root_translation = bool(np.max(np.abs(root_translation)) > 1e-8)
    root_motion_enabled = bool(
        args.allow_root_translation
        and (has_predicted_root_translation or has_nonzero_root_translation)
    )
    if (has_predicted_root_translation or has_nonzero_root_translation) and not root_motion_enabled:
        raise ValueError(
            "Motion contains predicted root translation; pass --allow-root-translation "
            "for a trajectory preview"
        )
    required_camera_fields = (
        "visualization_camera_root_position",
        "visualization_camera_intrinsics",
        "visualization_camera_image_size",
    )
    available_camera_fields = [field in motion for field in required_camera_fields]
    if any(available_camera_fields) and not all(available_camera_fields):
        missing = [
            field for field, available in zip(required_camera_fields, available_camera_fields)
            if not available
        ]
        raise ValueError(f"Incomplete visualization camera fields: missing={missing}")
    has_camera = all(available_camera_fields)
    if root_motion_enabled and has_camera:
        raise ValueError(
            "Predicted root translation and visualization camera-root fields cannot "
            "be enabled together"
        )
    camera_root = None
    calibration = None
    if has_camera:
        camera_root = np.asarray(
            motion["visualization_camera_root_position"], dtype=np.float64
        )
        if camera_root.shape != (len(root_translation), 3):
            raise ValueError("visualization camera root must be [T,3]")
        calibration = blender_camera_calibration_from_opencv(
            motion["visualization_camera_intrinsics"],
            motion["visualization_camera_image_size"],
        )
    deltas = deformation_deltas(
        motion["pred_local_rotation_matrix"],
        asset["rest_translations"],
        asset["rest_rotations_quat"],
        asset["parents"],
        root_translation,
    )
    deformed = linear_blend_skinning(
        asset["vertices"], asset["weights"], deltas
    )
    if not np.isfinite(deformed).all():
        raise ValueError("Deformed mesh contains NaN/Inf")

    # The learned root rotation and root-centered mesh use OpenCV camera axes.
    # When calibration exists, preserve the original perspective-camera preview.
    # For unlabeled videos, keep the mesh in place and use a fitted orthographic
    # Blender camera; do not invent root XYZ or pinhole intrinsics.
    if has_camera:
        camera_vertices = deformed + camera_root[:, None, :]
        if np.any(camera_vertices[..., 2] <= 0.01):
            raise ValueError("Camera-aligned mesh contains vertices behind the camera")
        display = opencv_camera_points_to_blender(camera_vertices)
    else:
        camera_vertices = None
        display = opencv_camera_points_to_blender(deformed)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh_data = bpy.data.meshes.new("DrivenMesh")
    mesh_data.from_pydata(
        display[0].tolist(), [], np.asarray(asset["faces"], dtype=np.int32).tolist()
    )
    mesh_data.update()
    mesh_obj = bpy.data.objects.new("DrivenMesh", mesh_data)
    bpy.context.collection.objects.link(mesh_obj)

    # mesh_asset.npz stores per-face UVs and texture paths relative to the asset.
    texture_refs = [str(value) for value in asset.get("texture_files", [])]
    has_uv = "uvs" in asset and any(texture_refs)
    textured = False
    if has_uv:
        uv = np.asarray(asset["uvs"], dtype=np.float32)
        faces = np.asarray(asset["faces"], dtype=np.int32)
        if uv.shape == (len(faces) * 3, 2):
            face_uv = uv.reshape(len(faces), 3, 2)
        elif uv.shape == (len(asset["vertices"]), 2):
            face_uv = uv[faces]
        else:
            raise ValueError(f"Invalid mesh UV shape {uv.shape}")
        uv_layer = mesh_data.uv_layers.new(name="DiffuseUV")
        for polygon_index, polygon in enumerate(mesh_data.polygons):
            for corner, loop_index in enumerate(polygon.loop_indices):
                uv_layer.data[loop_index].uv = face_uv[polygon_index, corner]
    face_material = np.asarray(
        asset.get("face_material", np.zeros(len(mesh_data.polygons), dtype=np.int32)),
        dtype=np.int32,
    )
    if face_material.shape != (len(mesh_data.polygons),) or np.any(face_material < 0):
        raise ValueError("Invalid per-face material indices")
    material_count = max(1, len(texture_refs), int(face_material.max()) + 1)
    for index in range(material_count):
        material = bpy.data.materials.new(f"DrivenMeshMaterial_{index}")
        material.use_nodes = True
        material.diffuse_color = (0.68, 0.72, 0.78, 1.0)
        principled = material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Base Color"].default_value = (0.68, 0.72, 0.78, 1.0)
        principled.inputs["Roughness"].default_value = 0.72
        if has_uv and index < len(texture_refs) and texture_refs[index]:
            reference = Path(texture_refs[index].replace("\\", "/"))
            texture_path = args.mesh_path.parent / reference
            if texture_path.is_file():
                texture_image = bpy.data.images.load(str(texture_path), check_existing=True)
                image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
                image_node.image = texture_image
                texture_image.colorspace_settings.name = "sRGB"
                material.node_tree.links.new(image_node.outputs["Color"], principled.inputs["Base Color"])
                textured = True
        mesh_data.materials.append(material)
    for polygon, material_index in zip(mesh_data.polygons, face_material):
        polygon.material_index = int(material_index)
        polygon.use_smooth = True

    all_low = display.min(axis=(0, 1))
    all_high = display.max(axis=(0, 1))
    center = (all_low + all_high) * 0.5
    extent = float(np.max(all_high - all_low))
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    if has_camera:
        camera_data.type = "PERSP"
        camera_data.sensor_fit = "HORIZONTAL"
        camera_data.sensor_width = calibration["sensor_width_mm"]
        camera_data.lens = calibration["lens_mm"] * args.camera_focal_scale
        camera_data.shift_x = calibration["shift_x"]
        camera_data.shift_y = calibration["shift_y"] + args.camera_shift_y
        camera_data.clip_start = 0.01
        camera_data.clip_end = max(
            100.0, float(camera_vertices[..., 2].max()) * 2.0
        )
        # The mesh remains in its original camera/world position. Moving the
        # camera along its viewing axis changes framing without changing mesh
        # coordinates or the camera orientation.
        camera.location = (0.0, 0.0, -args.camera_advance)
        camera.rotation_euler = (0.0, 0.0, 0.0)
    else:
        xy_extent = np.maximum(all_high[:2] - all_low[:2], 1e-6)
        camera_distance = max(2.5 * extent, 1.0)
        camera_data.type = "ORTHO"
        camera_data.clip_start = 0.01
        camera_data.clip_end = max(100.0, camera_distance + 4.0 * extent)
        camera_data.ortho_scale = 1.12 * float(np.max(xy_extent))
        camera.location = tuple(center + np.asarray([0.0, 0.0, camera_distance]))
        look_at(camera, center)
    if args.lighting_preset == "mixamo":
        # The reference Mixamo renderer uses .55 * (80, 52, 80) * extent².
        # Its camera is fixed at azimuth 15° and elevation 18°; rotate the
        # world-space lamp offsets into this renderer's camera-local axes.
        mixamo_camera_from_world = np.asarray([
            [0.96592583, 0.25881905, 0.0],
            [0.07997948, -0.29848750, -0.95105652],
            [-0.24615154, 0.91865005, -0.30901699],
        ])
        cv_to_blender = np.diag([1.0, -1.0, -1.0])
        mixamo_world_vertices = display @ cv_to_blender @ mixamo_camera_from_world
        mixamo_extent = float(np.max(np.ptp(mixamo_world_vertices.reshape(-1, 3), axis=0)))
        first_world = mixamo_world_vertices[0]
        first_world_center = (first_world.min(axis=0) + first_world.max(axis=0)) * 0.5
        mixamo_light_center = first_world_center @ mixamo_camera_from_world.T @ cv_to_blender
        for name, offset, energy, size in (
            ("Key", (-0.8, -1.2, 1.5), configured_energies[0], 1.0),
            ("Fill", (1.0, -0.7, 0.6), configured_energies[1], 1.0),
            ("Rim", (0.0, 1.0, 1.4), configured_energies[2], 0.8),
        ):
            camera_offset = cv_to_blender @ mixamo_camera_from_world @ np.asarray(offset)
            light = add_area_light(
                name, tuple(mixamo_light_center + camera_offset * mixamo_extent),
                energy * mixamo_extent ** 2, size * mixamo_extent,
            )
            look_at(light, mixamo_light_center)
        light_names = ["key", "fill", "rim"]
        applied_energies = [energy * mixamo_extent ** 2 for energy in configured_energies]
        world_rgb = (0.17, 0.17, 0.17)
        default_world_strength = 0.385
        color_transform = "Standard"
        color_look = "Medium High Contrast"
    else:
        # Fixed Zoo blank.blend lamp positions in camera-local axes.
        for name, kind, offset, rotation, energy in (
            ("ZooPoint", "POINT", (4.07625, 5.97147, -0.45616), None, configured_energies[0]),
            ("ZooSpot1", "SPOT", (-0.22654, 3.72998, 0.34580), (0.73902, -0.67368, 0.0, 0.0), configured_energies[1]),
            ("ZooSpot2", "SPOT", (0.35470, 1.34626, -3.81589), (0.12869, -0.96645, 0.21979, -0.03327), configured_energies[2]),
            ("ZooSpot3", "SPOT", (5.69831, 4.24949, -1.38974), (0.47953, -0.74586, 0.20809, -0.41286), configured_energies[3]),
            ("ZooArea", "AREA", (2.00707, 1.47270, 2.29575), (0.78562, -0.32960, 0.13359, -0.50628), configured_energies[4]),
        ):
            data = bpy.data.lights.new(name, type=kind)
            data.energy = energy
            if kind == "POINT":
                data.shadow_soft_size = 0.1
            elif kind == "SPOT":
                data.spot_size = math.pi / 4
                data.spot_blend = 0.15
            elif kind == "AREA":
                data.shape = "SQUARE"
                data.size = 10.0
                data.color = (1.0, 0.98461133, 0.66479588)
            light = bpy.data.objects.new(name, data)
            bpy.context.collection.objects.link(light)
            light.location = offset
            if rotation is not None:
                light.rotation_mode = "QUATERNION"
                light.rotation_quaternion = Quaternion(rotation)
        light_names = ["point", "spot_1", "spot_2", "spot_3", "area"]
        applied_energies = list(configured_energies)
        world_rgb = (0.05087608844, 0.05087608844, 0.05087608844)
        default_world_strength = 1.0
        color_transform = "AgX"
        color_look = "None"

    scene = bpy.context.scene
    # Blender 4.x names the realtime engine EEVEE_NEXT; Blender 3.x uses
    # BLENDER_EEVEE. Keep the existing renderer and select the installed API.
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = (
        max(1, round(args.resolution * calibration["height"] / calibration["width"]))
        if has_camera else args.resolution
    )
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = calibration["pixel_aspect_x"] if has_camera else 1.0
    scene.render.pixel_aspect_y = calibration["pixel_aspect_y"] if has_camera else 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("MeshPreviewWorld")
    background_strength = default_world_strength if args.world_strength is None else args.world_strength
    scene.world.color = world_rgb
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (*world_rgb, 1.0)
        background.inputs["Strength"].default_value = background_strength
    try:
        scene.view_settings.view_transform = color_transform
    except TypeError:
        scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = color_look
    except TypeError:
        scene.view_settings.look = "None"
    scene.view_settings.exposure = args.exposure
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for frame, vertices in enumerate(display):
        mesh_data.vertices.foreach_set("co", vertices.reshape(-1))
        mesh_data.update()
        scene.render.filepath = str(
            (args.output_dir / f"frame_{frame:04d}.png").resolve()
        )
        bpy.ops.render.render(write_still=True)

    report = {
        "mesh_path": str(args.mesh_path.resolve()),
        "motion": str(args.motion.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "frames": len(display),
        "vertices": display.shape[1],
        "root_translation": (
            "shared_gt_camera_root_for_visualization"
            if has_camera
            else (
                "predicted_camera_xyz"
                if root_motion_enabled
                else "none_root_centered_in_place"
            )
        ),
        "coordinate_system": str(motion["coordinate_system"]),
        "view": (
            "opencv_perspective_camera"
            if has_camera
            else (
                "root_motion_front_orthographic_camera_axes"
                if root_motion_enabled
                else "root_centered_orthographic_camera_axes"
            )
        ),
        "root_motion_camera_elev_deg": 0.0 if root_motion_enabled else None,
        "root_motion_camera_azim_deg": 0.0 if root_motion_enabled else None,
        "camera_intrinsics": (
            np.asarray(motion["visualization_camera_intrinsics"]).tolist()
            if has_camera else None
        ),
        "camera_image_size": (
            np.asarray(motion["visualization_camera_image_size"]).tolist()
            if has_camera else None
        ),
        "camera_focal_scale": float(args.camera_focal_scale),
        "camera_shift_y": float(args.camera_shift_y),
        "camera_advance": float(args.camera_advance),
        "orthographic_scale": float(camera_data.ortho_scale) if not has_camera else None,
        "lighting_preset": args.lighting_preset,
        "background": "transparent",
        "lighting": {
            "light_names": light_names,
            "configured_energies": list(configured_energies),
            "applied_energies": applied_energies,
            "world_strength": background_strength,
            "world_rgb": list(world_rgb),
            "exposure": float(args.exposure),
            "color_transform": color_transform,
        },
        "textured_materials": textured,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
