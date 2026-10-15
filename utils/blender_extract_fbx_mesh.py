#!/usr/bin/env python3
"""Extract FBX mesh, skin weights, and materials inside Blender."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.mesh_skinning import fit_proper_similarity, rest_global_transforms
from utils.blender_materials import MaterialExporter
from utils.fbx_coordinates import fbx_source_hash, read_fbx_coordinates


def image_for_material(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    principled = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is not None:
        for link in principled.inputs["Base Color"].links:
            if link.from_node.type == "TEX_IMAGE" and link.from_node.image is not None:
                return link.from_node.image
    return None


def source_driver(name: str, bones, target_indices: dict[str, int]) -> int | None:
    bone = bones.get(name)
    while bone is not None:
        if bone.name in target_indices:
            return target_indices[bone.name]
        bone = bone.parent
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    static = np.load(args.static, allow_pickle=True).item()
    names = [str(name) for name in static["joint_names"]]
    parents = np.asarray(static["parents"], dtype=np.int32)
    translations = np.asarray(static["rest_translations"], dtype=np.float32)
    quaternions = np.asarray(static["rest_rotations_quat"], dtype=np.float32)
    target_rest = rest_global_transforms(translations, quaternions, parents)[:, :3, 3]
    target_indices = {name: index for index, name in enumerate(names)}

    try:
        bpy.ops.preferences.addon_disable(module="cycles")
    except Exception:
        pass
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_options = dict(use_anim=False, automatic_bone_orientation=False,
                          ignore_leaf_bones=False, primary_bone_axis='Y', secondary_bone_axis='X')
    explicit_basis = int(static.get('fbx_basis_conversion_version', 0)) == 2
    if explicit_basis:
        metadata, _, _ = read_fbx_coordinates(args.fbx)
        if fbx_source_hash(args.fbx) != static['source_fbx_sha256']:
            raise ValueError('FBX differs from the source used to generate static.npy')
        import_options.update(use_manual_orientation=True,
                              axis_up=metadata['blender_import_up'],
                              axis_forward=metadata['blender_import_forward'],
                              global_scale=1.0, bake_space_transform=False)
    bpy.ops.import_scene.fbx(filepath=str(args.fbx.resolve()), **import_options)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise ValueError(f"Expected one armature and at least one mesh in {args.fbx}")
    armature = armatures[0]
    armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    bones = armature.data.bones
    matched = [name for name in names if name in bones]
    if len(matched) < 3:
        raise ValueError(f"Too few FBX bones match the target skeleton: {matched}")
    source_rest = np.asarray(
        [armature.matrix_world @ bones[name].head_local for name in matched],
        dtype=np.float64,
    )
    if explicit_basis:
        with np.load(args.static.parent / 'bind_transforms.npz', allow_pickle=False) as bind:
            transform = bind['camera_from_world']
        scale = float(np.linalg.norm(transform[:3, 0]))
        rotation = transform[:3, :3] / scale
        if scale <= 0 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError('Invalid explicit FBX mesh basis transform')
        fitted = scale * (source_rest @ rotation.T) + transform[:3, 3]
        errors = np.linalg.norm(fitted - target_rest[[target_indices[name] for name in matched]], axis=1)
        fit = dict(scale=scale, rotation=rotation, translation=transform[:3, 3], errors=errors)
    else:
        fit = fit_proper_similarity(source_rest, target_rest[[target_indices[name] for name in matched]])
    max_error = float(np.max(fit["errors"]))
    if fit["scale"] <= 0 or max_error > 0.01:
        raise ValueError(f"FBX bind skeleton differs from target static: max error {max_error:.6f} m")

    all_vertices, all_faces, all_weights, all_uvs, all_materials = [], [], [], [], []
    all_normals, all_smooth = [], []
    uv_names = sorted({layer.name for obj in meshes for layer in obj.data.uv_layers})
    uv_layers = {name: [] for name in uv_names}
    exporter = MaterialExporter(args.output_dir)
    material_records = []
    texture_files = []
    vertex_offset = 0
    for obj in meshes:
        data = obj.data
        # Reading loop triangles preserves the FBX's corner normals and UV seams.
        # Destructive bmesh triangulation can discard its custom split normals.
        data.calc_loop_triangles()
        normal_transform = fit["rotation"] @ np.linalg.inv(np.asarray(obj.matrix_world)[:3, :3]).T
        corner_normals = np.asarray([normal.vector[:] for normal in data.corner_normals], dtype=np.float64)
        corner_normals = corner_normals @ normal_transform.T
        corner_normals /= np.maximum(np.linalg.norm(corner_normals, axis=1, keepdims=True), 1e-12)
        vertices = np.asarray([obj.matrix_world @ vertex.co for vertex in data.vertices], dtype=np.float64)
        vertices = fit["scale"] * (vertices @ fit["rotation"].T) + fit["translation"]
        weights = np.zeros((len(vertices), len(names)), dtype=np.float32)
        for vertex_index, vertex in enumerate(data.vertices):
            for group in vertex.groups:
                name = obj.vertex_groups[group.group].name
                driver = source_driver(name, bones, target_indices)
                if driver is not None:
                    weights[vertex_index, driver] += group.weight
        totals = weights.sum(axis=1)
        if np.any(totals <= 1e-8) and obj.parent_type == "BONE":
            driver = source_driver(obj.parent_bone, bones, target_indices)
            if driver is not None:
                weights[totals <= 1e-8, driver] = 1.0
                totals = weights.sum(axis=1)
        if np.any(totals <= 1e-8):
            raise ValueError(f"FBX mesh {obj.name} has vertices without target skin weights")
        weights /= totals[:, None]
        material_offset = len(texture_files)
        slots = list(data.materials)
        if not slots:
            slots = [None]
        for material in slots:
            material_records.append(exporter.material(material))
            base_image = exporter.image(image_for_material(material))
            texture_files.append(base_image["path"] if base_image else "")
        uv_layer = data.uv_layers.active
        mirrored = np.linalg.det(fit['rotation'] @ np.asarray(obj.matrix_world)[:3, :3]) < 0
        for triangle in data.loop_triangles:
            polygon = data.polygons[triangle.polygon_index]
            loops = list(triangle.loops)
            if mirrored:
                loops.reverse()
            all_faces.append([vertex_offset + data.loops[index].vertex_index for index in loops])
            all_materials.append(material_offset + min(int(polygon.material_index), len(slots) - 1))
            all_smooth.append(polygon.use_smooth)
            for loop_index in loops:
                all_normals.append(corner_normals[loop_index])
                all_uvs.append(tuple(uv_layer.data[loop_index].uv) if uv_layer is not None else (0.0, 0.0))
                for name in uv_names:
                    layer = data.uv_layers.get(name)
                    uv_layers[name].append(tuple(layer.data[loop_index].uv) if layer else (0.0, 0.0))
        all_vertices.append(vertices)
        all_weights.append(weights)
        vertex_offset += len(vertices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "mesh_asset.npz"
    staged = output.with_suffix(".npz.tmp")
    with staged.open("wb") as file:
        np.savez_compressed(
            file,
            vertices=np.concatenate(all_vertices).astype(np.float32),
            faces=np.asarray(all_faces, dtype=np.int32),
            weights=np.concatenate(all_weights).astype(np.float32),
            uvs=np.asarray(all_uvs, dtype=np.float32),
            face_material=np.asarray(all_materials, dtype=np.int32),
            texture_files=np.asarray(texture_files),
            materials_json=np.asarray(json.dumps({"version": 1, "materials": material_records})),
            corner_normals=np.asarray(all_normals, dtype=np.float32),
            face_smooth=np.asarray(all_smooth, dtype=np.bool_),
            uv_layer_names=np.asarray(uv_names, dtype=str),
            uv_layers=np.asarray([uv_layers[name] for name in uv_names], dtype=np.float32),
            parents=parents,
            rest_translations=translations,
            rest_rotations_quat=quaternions,
            joint_names=np.asarray(names),
            skeleton_hash=np.asarray(str(static["skeleton_hash"])),
        )
    os.replace(staged, output)
    print(f"{output} (FBX bind fit {max_error:.6f} m)", flush=True)


if __name__ == "__main__":
    main()
