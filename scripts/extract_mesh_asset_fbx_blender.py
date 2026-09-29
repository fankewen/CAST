#!/usr/bin/env python3
"""Extract a skinned render mesh directly from a character FBX in Blender."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

import bmesh
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.mesh_skinning import fit_proper_similarity, rest_global_transforms


def image_for_material(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        for link in principled.inputs["Base Color"].links:
            if link.from_node.type == "TEX_IMAGE" and link.from_node.image is not None:
                return link.from_node.image
    return next(
        (node.image for node in material.node_tree.nodes
         if node.type == "TEX_IMAGE" and node.image is not None),
        None,
    )


def copy_image(image, destination: Path, material_index: int) -> str:
    if image is None:
        return ""
    source = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
    destination.mkdir(parents=True, exist_ok=True)
    if source is not None and source.is_file():
        target = destination / f"{material_index:03d}_{source.name}"
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    elif image.has_data:
        target = destination / f"{material_index:03d}.png"
        image.save_render(str(target))
    else:
        return ""
    return (Path("textures") / target.name).as_posix()


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
    bpy.ops.import_scene.fbx(filepath=str(args.fbx.resolve()))
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
    fit = fit_proper_similarity(source_rest, target_rest[[target_indices[name] for name in matched]])
    max_error = float(np.max(fit["errors"]))
    if fit["scale"] <= 0 or max_error > 0.01:
        raise ValueError(f"FBX bind skeleton differs from target static: max error {max_error:.6f} m")

    all_vertices, all_faces, all_weights, all_uvs, all_materials = [], [], [], [], []
    texture_files = []
    vertex_offset = 0
    for obj in meshes:
        data = obj.data
        bm = bmesh.new()
        bm.from_mesh(data)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(data)
        bm.free()
        data.update()
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
        texture_files.extend(
            copy_image(image_for_material(material), args.output_dir / "textures", material_offset + index)
            for index, material in enumerate(slots)
        )
        uv_layer = data.uv_layers.active
        for polygon in data.polygons:
            if len(polygon.vertices) != 3:
                raise ValueError("FBX triangulation failed")
            all_faces.append([vertex_offset + int(v) for v in polygon.vertices])
            all_materials.append(material_offset + min(int(polygon.material_index), len(slots) - 1))
            for loop_index in polygon.loop_indices:
                all_uvs.append(tuple(uv_layer.data[loop_index].uv) if uv_layer is not None else (0.0, 0.0))
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
