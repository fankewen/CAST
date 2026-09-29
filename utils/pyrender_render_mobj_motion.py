#!/usr/bin/env python3
"""Render a driven Mobjaverse mesh with the dataset's pyrender lighting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image
import pyrender
import trimesh

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.mesh_skinning import (
    deformation_deltas,
    linear_blend_skinning,
    load_mesh_asset,
    opencv_camera_points_to_blender,
)

THIRD_PARTY = Path(__file__).resolve().parents[1] / "preprocess" / "mobjaverse" / "third_party"
sys.path.insert(0, str(THIRD_PARTY))
from render import make_textured_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lighting-preset", choices=("mobj",), required=True)
    parser.add_argument("--light-energies", type=float, nargs="+")
    parser.add_argument("--allow-root-translation", action="store_true")
    args = parser.parse_args()
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")

    mesh_path = args.mesh_path.expanduser().resolve()
    motion_path = args.motion.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    asset = load_mesh_asset(mesh_path)
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {key: np.array(archive[key], copy=True) for key in archive.files}
    if str(asset["skeleton_hash"]) != str(motion["skeleton_hash"]):
        raise ValueError("Mesh and motion skeleton hashes differ")
    if not np.array_equal(asset["joint_names"], motion["joint_names"]):
        raise ValueError("Mesh and motion joint names differ")
    if not np.array_equal(asset["parents"], motion["parents"]):
        raise ValueError("Mesh and motion hierarchies differ")

    root_translation = np.asarray(motion["pred_root_translation"], dtype=np.float64)
    has_root_motion = bool(motion["has_predicted_root_translation"]) or bool(
        np.max(np.abs(root_translation)) > 1e-8
    )
    if has_root_motion and not args.allow_root_translation:
        raise ValueError("Motion contains root translation; pass --allow-root-translation")
    camera_fields = (
        "visualization_camera_root_position",
        "visualization_camera_intrinsics",
        "visualization_camera_image_size",
    )
    present = [name in motion for name in camera_fields]
    if any(present) and not all(present):
        raise ValueError("Incomplete visualization camera fields")
    has_camera = all(present)
    if has_camera and has_root_motion:
        raise ValueError("Camera-root preview and predicted root translation cannot be combined")

    deltas = deformation_deltas(
        motion["pred_local_rotation_matrix"], asset["rest_translations"],
        asset["rest_rotations_quat"], asset["parents"], root_translation,
    )
    vertices = linear_blend_skinning(asset["vertices"], asset["weights"], deltas)
    if has_camera:
        camera_root = np.asarray(motion["visualization_camera_root_position"], dtype=np.float64)
        if camera_root.shape != (len(vertices), 3):
            raise ValueError("visualization camera root must be [T,3]")
        vertices = vertices + camera_root[:, None, :]
    display = opencv_camera_points_to_blender(vertices)

    resolution = args.resolution
    camera_pose = np.eye(4, dtype=np.float64)
    if has_camera:
        fx, fy, cx, cy = np.asarray(motion["visualization_camera_intrinsics"], dtype=np.float64)
        original_width, original_height = np.asarray(
            motion["visualization_camera_image_size"], dtype=np.float64
        )
        scale = resolution / original_width
        height = max(1, round(original_height * scale))
        camera = pyrender.IntrinsicsCamera(
            fx=fx * scale, fy=fy * scale, cx=cx * scale, cy=cy * scale,
            znear=0.01,
        )
        view = "opencv_perspective_camera"
    else:
        height = resolution
        low = display.min(axis=(0, 1))
        high = display.max(axis=(0, 1))
        center = (low + high) * 0.5
        extent = float(np.max(high - low))
        ortho_scale = 1.12 * float(np.max(np.maximum(high[:2] - low[:2], 1e-6)))
        camera = pyrender.OrthographicCamera(
            xmag=ortho_scale / 2.0, ymag=ortho_scale / 2.0,
            znear=0.01, zfar=max(100.0, 10.0 * extent),
        )
        camera_pose[:3, 3] = center + np.asarray([0.0, 0.0, max(2.5 * extent, 1.0)])
        view = (
            "root_motion_front_orthographic_camera_axes"
            if has_root_motion else "root_centered_orthographic_camera_axes"
        )

    faces = np.asarray(asset["faces"], dtype=np.int32)
    texture_refs = [str(value) for value in asset.get("texture_files", [])]
    textured = bool("uvs" in asset and texture_refs and texture_refs[0])
    if textured:
        texture_path = mesh_path.parent / texture_refs[0]
        texture = np.asarray(Image.open(texture_path).convert("RGB"), dtype=np.uint8)
        uvs = np.asarray(asset["uvs"], dtype=np.float32)
        if uvs.shape == (len(asset["vertices"]), 2):
            uvs = uvs[faces].reshape(-1, 2)
        if uvs.shape != (len(faces) * 3, 2):
            raise ValueError(f"Invalid Mobjaverse UV shape: {uvs.shape}")
    else:
        texture = uvs = None
    if args.light_energies is not None and len(args.light_energies) != 1:
        raise ValueError("mobj requires one light energy")
    intensity = (7.5 if textured else 1.5) if args.light_energies is None else args.light_energies[0]
    if not np.isfinite(intensity) or intensity < 0:
        raise ValueError("Mobjaverse light intensity must be finite and non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=height)
    material = new_texture = new_uvs = inverse_indices = unique_indices = None
    try:
        for frame, posed_vertices in enumerate(display):
            if textured:
                (tri_mesh, material, new_texture, new_uvs,
                 inverse_indices, unique_indices) = make_textured_mesh(
                    vertices=posed_vertices, faces=faces, texture=texture, uvs=uvs,
                    new_texture=new_texture, new_uvs=new_uvs,
                    inverse_indices=inverse_indices, unique_indices=unique_indices,
                    cached_material=material,
                )
            else:
                tri_mesh = trimesh.Trimesh(
                    vertices=posed_vertices, faces=faces,
                    vertex_colors=asset.get("vertex_colors"), process=False,
                )
            mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True, material=material)
            scene = pyrender.Scene()
            scene.add(mesh)
            camera_node = scene.add(camera, pose=camera_pose)
            light_node = scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                pose=camera_pose,
            )
            scene.main_camera_node = camera_node
            color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.OFFSCREEN)
            rgba = np.empty((height, resolution, 4), dtype=np.uint8)
            rgba[:, :, :3] = color
            rgba[:, :, 3] = np.where(depth > 0, 255, 0).astype(np.uint8)
            Image.fromarray(rgba, "RGBA").save(output_dir / f"frame_{frame:04d}.png")
            scene.remove_node(light_node)
            scene.remove_node(camera_node)
    finally:
        renderer.delete()

    report = {
        "mesh_path": str(mesh_path),
        "motion": str(motion_path),
        "output_dir": str(output_dir),
        "frames": len(display),
        "view": view,
        "lighting_preset": args.lighting_preset,
        "lighting": {
            "light_names": ["directional"],
            "configured_energies": [float(intensity)],
            "applied_energies": [float(intensity)],
        },
        "background": "transparent",
        "textured_materials": textured,
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
