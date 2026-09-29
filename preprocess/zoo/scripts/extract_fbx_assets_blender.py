"""Blender worker for one Truebone animation FBX and its character FBX.

The extraction follows MocapAnything's FBX export convention: BVH joint order
defines skinning-weight columns, including any synthetic Blender root nodes.
Run through prepare_zoo_fbx.py, not directly.
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

# Blender 5.1's FBX light importer writes a removed Cycles property. FBX
# lights are irrelevant to skeleton and mesh extraction, so disable Cycles
# for this Blender process before importing FBXs.
bpy.ops.preferences.addon_disable(module="cycles")


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_fbx(path):
    reset_scene()
    bpy.ops.import_scene.fbx(filepath=str(path))
    imported = list(bpy.context.selected_objects)
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(armatures) != 1:
        raise ValueError(f"Expected one armature in {path}; found {len(armatures)}")
    return armatures[0], meshes


def export_bvh(path, destination):
    armature, meshes = import_fbx(path)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    action = armature.animation_data.action if armature.animation_data else None
    if action:
        start, end = action.frame_range
        bpy.context.scene.frame_start = int(start)
        bpy.context.scene.frame_end = int(end)
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_anim.bvh(filepath=str(destination), root_transform_only=True)
    reset_scene()
    return bool(meshes)


def bvh_names(path):
    result = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith(("ROOT ", "JOINT ")):
                result.append(line.split(None, 1)[1])
            elif line == "MOTION":
                break
    return result


def bvh_signature(path):
    names, offsets = [], []
    pending = False
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.strip().split()
            if parts[:1] in (["ROOT"], ["JOINT"]):
                names.append(parts[1])
                pending = True
            elif parts[:1] == ["OFFSET"] and pending:
                offsets.append([float(value) for value in parts[1:4]])
                pending = False
            elif parts[:1] == ["MOTION"]:
                break
    return names, np.asarray(offsets)


def select_base(animation_bvh, candidates, scratch):
    names, offsets = bvh_signature(animation_bvh)
    name_match = None
    for index, candidate in enumerate(candidates):
        rest = scratch / f"candidate_{index:03d}.bvh"
        has_mesh = export_bvh(candidate, rest)
        if not has_mesh:
            continue
        rest_names, rest_offsets = bvh_signature(rest)
        if rest_names[:1] == ["Null"] and rest_names[1:] == names:
            rest_offsets = rest_offsets[1:]
        if rest_names == names or (rest_names[:1] == ["Null"] and rest_names[1:] == names):
            if name_match is None:
                name_match = (candidate, rest)
            if np.allclose(rest_offsets, offsets, atol=1e-3, rtol=0):
                return candidate, rest
    if name_match is not None:
        return name_match
    raise ValueError(f"No character FBX matches the animation skeleton: {animation_bvh}")


def extract_character(path, destination, names):
    armature, meshes = import_fbx(path)
    if not meshes:
        raise ValueError(f"No mesh objects in {path}")
    destination.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    np.save(destination / "blender_vertices.npy", np.asarray([v.co[:] for v in mesh.data.vertices]))

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode="OBJECT")

    index = {name: i for i, name in enumerate(names)}
    weights = np.zeros((len(mesh.data.vertices), len(names)), dtype=np.float32)
    for row, vertex in enumerate(mesh.data.vertices):
        for group in vertex.groups:
            name = mesh.vertex_groups[group.group].name
            if name in index:
                weights[row, index[name]] = group.weight
        total = weights[row].sum()
        if total > 1e-8:
            weights[row] /= total
    if not np.any(weights):
        raise ValueError(f"No FBX vertex groups match BVH joints in {path}")
    np.save(destination / "skinning_weights.npy", weights)

    for material in mesh.data.materials:
        if not material or not material.use_nodes or not material.node_tree:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            image = node.image
            source = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
            if source and source.is_file():
                target = destination / source.name
                if not target.exists():
                    shutil.copyfile(source, target)
            elif image.has_data:
                target = destination / f"{image.name}.png"
                if not target.exists():
                    image.filepath_raw = str(target)
                    image.file_format = "PNG"
                    image.save()

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.wm.obj_export(
        filepath=str(destination / "base_mesh.obj"),
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
    )
    # FBX materials often carry machine-local C:/ texture paths. The images
    # above were copied beside the OBJ, so make the MTL portable for rendering.
    mtl = destination / "base_mesh.mtl"
    if mtl.is_file():
        texture_names = {p.name for p in destination.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tga")}
        updated = []
        for line in mtl.read_text(encoding="utf-8", errors="replace").splitlines():
            drive = re.search(r"[A-Za-z]:[\\/]", line)
            if line.startswith("map_") and drive:
                filename = line[drive.start():].replace("\\", "/").rsplit("/", 1)[-1]
                if filename in texture_names:
                    line = line[:drive.start()] + filename
            updated.append(line)
        mtl.write_text("\n".join(updated) + "\n", encoding="utf-8")
    reset_scene()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--animation-fbx", type=Path, required=True)
    p.add_argument("--base-candidates", required=True)
    p.add_argument("--motion", required=True)
    p.add_argument("--species", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    candidates = [Path(path) for path in json.loads(args.base_candidates)]
    animation_bvh = args.output_root / "motions" / f"{args.motion}.bvh"
    export_bvh(args.animation_fbx, animation_bvh)
    character = args.output_root / "characters_by_motion" / args.motion
    character.mkdir(parents=True, exist_ok=True)
    rest = character / "rest.bvh"
    with tempfile.TemporaryDirectory(prefix="cast-zoo-base-") as temporary:
        base, candidate_rest = select_base(animation_bvh, candidates, Path(temporary))
        shutil.copyfile(candidate_rest, rest)
    names = bvh_names(animation_bvh)
    extract_character(base, character, names)
    selection = args.output_root / "fbx_selection" / f"{args.motion}.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text(
        json.dumps({"animation_fbx": str(args.animation_fbx), "base_fbx": str(base)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {args.motion} with {len(names)} character joints", flush=True)


if __name__ == "__main__":
    main()
