#!/usr/bin/env python3
"""Convert a skinned FBX into CAST skeleton, bind transforms, and render mesh.

Usage: python -m utils.convert_fbx_to_cast --source character.fbx
       --blender /path/to/blender --output-dir results/character_asset

The source's root, hierarchy, bone count, and character orientation are kept.
FBX GlobalSettings supplies the world axes, handedness and units. Both world
and bone-local bases are expressed as +X right, +Y down, +Z forward, in meters.
BindPose/skin-cluster matrices supply rest; animation frames are never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from utils.mesh_skinning import COORDINATE_SYSTEM, rest_global_transforms, validate_parents
from utils.fbx_coordinates import fbx_source_hash, read_fbx_coordinates


def rigid_rest(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite bone rest transform")
    u, scales, vt = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) <= 0:
        raise ValueError("Reflected bone transforms are unsupported")
    if scales.min() <= 1e-12 or scales.max() / scales.min() > 1.001:
        raise ValueError("Nonuniform bone scaling is unsupported")
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = matrix[:3, 3]
    return result


def extract_skeleton(args) -> None:
    import bpy
    from mathutils import Matrix

    try:
        bpy.ops.preferences.addon_disable(module="cycles")
    except Exception:
        pass
    bpy.ops.wm.read_factory_settings(use_empty=True)
    metadata, bind_by_name, bind_sources = read_fbx_coordinates(args.source)
    bpy.ops.import_scene.fbx(filepath=str(args.source), use_anim=False,
                             use_manual_orientation=True,
                             axis_up=metadata['blender_import_up'],
                             axis_forward=metadata['blender_import_forward'],
                             automatic_bone_orientation=False, ignore_leaf_bones=False,
                             primary_bone_axis='Y', secondary_bone_axis='X',
                             global_scale=1.0, bake_space_transform=False)
    rigs = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(rigs) != 1:
        raise ValueError(f"Expected exactly one armature, found {len(rigs)}")
    rig = rigs[0]
    rig.data.pose_position = "REST"
    bpy.context.view_layer.update()
    roots = [bone for bone in rig.data.bones if bone.parent is None]
    if len(roots) != 1:
        raise ValueError(f"Inference needs a single-root skeleton; source has {len(roots)} roots")
    bones = []

    def visit(bone):
        bones.append(bone)
        for child in bone.children:
            visit(child)

    visit(roots[0])
    names = [bone.name for bone in bones]
    indices = {name: j for j, name in enumerate(names)}
    parents = np.array([-1 if bone.parent is None else indices[bone.parent.name] for bone in bones], dtype=np.int32)
    validate_parents(parents)
    world_rest = np.stack([rigid_rest(rig.matrix_world @ bone.matrix_local) for bone in bones])
    importer = np.asarray(metadata['source_to_blender'], dtype=np.float64)
    inverse_importer = np.linalg.inv(importer)
    source_rest, rest_sources = [], []
    bind_position_error = 0.0
    for j, name in enumerate(names):
        if name in bind_by_name:
            matrix = bind_by_name[name]
            error = float(np.linalg.norm((importer @ matrix)[:3, 3] - world_rest[j, :3, 3]))
            bind_position_error = max(bind_position_error, error)
            # Blender's float32 bone hierarchy accumulates sub-millimeter
            # rounding error; the output still uses the original FBX matrix.
            if error > 1e-3:
                raise ValueError(f'FBX bind and imported rest disagree for {name}: {error:.8f} m')
            rest_sources.append('+'.join(bind_sources[name]))
        else:
            # Unskinned helper bones may have no PoseNode/TransformLink. Blender
            # evaluates their declared rest transforms with use_anim=False.
            matrix = inverse_importer @ world_rest[j]
            rest_sources.append('FBX declared rest transform (animation disabled)')
        source_rest.append(rigid_rest(matrix))
    source_rest = np.stack(source_rest)
    basis = np.asarray(metadata['source_to_opencv_basis'], dtype=np.float64)
    meter_scale = metadata['meters_per_source_unit']
    camera_rest = source_rest.copy()
    camera_rest[:, :3, :3] = basis @ source_rest[:, :3, :3] @ basis.T
    camera_rest[:, :3, 3] = meter_scale * (source_rest[:, :3, 3] @ basis.T)
    origin = camera_rest[0, :3, 3].copy()
    camera_rest[:, :3, 3] -= origin
    # This affine maps imported mesh points/normals, while the right-hand basis
    # change above maps each bone's local axes. Do not discard the latter.
    camera = np.eye(4)
    camera[:3, :3] = meter_scale * basis @ inverse_importer[:3, :3]
    camera[:3, 3] = -origin
    local = np.stack([camera_rest[j] if parent < 0 else np.linalg.inv(camera_rest[parent]) @ camera_rest[j]
                      for j, parent in enumerate(parents)])
    translations = local[:, :3, 3].astype(np.float32)
    quaternions = []
    for matrix in local:
        quat = Matrix(matrix[:3, :3].tolist()).to_quaternion()
        quaternions.append([quat.x, quat.y, quat.z, quat.w])
    quaternions = np.asarray(quaternions, dtype=np.float32)
    reconstructed = rest_global_transforms(translations, quaternions, parents)
    error = float(np.max(np.abs(reconstructed - camera_rest)))
    if not np.allclose(reconstructed, camera_rest, atol=2e-5, rtol=1e-6):
        raise ValueError(f"Rest transform reconstruction failed: {error}")
    diagonal = float(np.linalg.norm(np.ptp(camera_rest[:, :3, 3], axis=0)))
    if diagonal <= 1e-8:
        raise ValueError("Skeleton has zero rest extent")
    skeleton_hash = hashlib.sha256(json.dumps(names).encode() + parents.tobytes()
                                   + camera_rest.astype(np.float32).tobytes()).hexdigest()
    static = dict(skeleton_id=args.skeleton_id, joints=len(names), joint_names=np.asarray(names),
                  parents=parents, rest_translations=translations, rest_rotations_quat=quaternions,
                  metric_scale=np.float32(2 / diagonal), coordinate_system=COORDINATE_SYSTEM,
                  units="meters", skeleton_hash=skeleton_hash, valid_joint_mask=np.ones(len(names), bool),
                  source_fbx=str(args.source), fbx_basis_conversion_version=2,
                  source_fbx_sha256=fbx_source_hash(args.source),
                  source_coordinate_metadata=metadata)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "static.npy", static)
    np.savez(output / "bind_transforms.npz", world_rest=world_rest, camera_from_world=camera,
             parents=parents, joint_names=np.asarray(names), rest_local=local,
             source_rest=source_rest, opencv_rest=camera_rest,
             source_to_blender=importer, bone_local_target_to_source=basis.T)
    report = dict(source_fbx=str(args.source), skeleton_id=args.skeleton_id, root_bone=names[0],
                  joints=len(names), joint_names=names, parents=parents.tolist(),
                  coordinate_system=COORDINATE_SYSTEM, units="meters", camera_from_world=camera.tolist(),
                  metric_scale=float(static["metric_scale"]), skeleton_hash=skeleton_hash,
                  rest_reconstruction_max_error=error, source_coordinate_metadata=metadata,
                  rest_sources=rest_sources, bind_import_position_max_error_m=bind_position_error,
                  fbx_basis_conversion_version=2)
    (output / "skeleton.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Exported original root {names[0]!r}, all {len(names)} bones: {output / 'static.npy'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original skinned FBX")
    parser.add_argument("--blender", type=Path, help="Required for normal Python invocation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skeleton-id", help="Defaults to the FBX filename stem")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    args.source = args.source.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.skeleton_id = args.skeleton_id or args.source.stem
    if not args.source.is_file() or args.source.suffix.lower() != ".fbx":
        parser.error(f"--source must be an existing FBX: {args.source}")
    if args.worker:
        extract_skeleton(args)
        return
    if args.blender is None or not args.blender.expanduser().resolve().is_file():
        parser.error("--blender must name an existing Blender executable")
    blender = str(args.blender.expanduser().resolve())
    subprocess.run([blender, "-b", "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--",
                    "--worker", "--source", str(args.source), "--output-dir", str(args.output_dir),
                    "--skeleton-id", args.skeleton_id], check=True)
    subprocess.run([blender, "-b", "--python-exit-code", "1", "--python",
                    str(Path(__file__).with_name("blender_extract_fbx_mesh.py")), "--",
                    "--fbx", str(args.source), "--static", str(args.output_dir / "static.npy"),
                    "--output-dir", str(args.output_dir)], check=True)
    print(f"Inference assets ready: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
