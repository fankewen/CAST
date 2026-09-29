#!/usr/bin/env python3
"""Prepare one inference mesh from an original character FBX or Mobjaverse NPZ."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from utils.mesh_skinning import fit_proper_similarity, rest_global_transforms


def target_skeleton(static_path: Path) -> dict:
    static = np.load(static_path, allow_pickle=True).item()
    return {
        "parents": np.asarray(static["parents"], dtype=np.int32),
        "rest_translations": np.asarray(static["rest_translations"], dtype=np.float32),
        "rest_rotations_quat": np.asarray(static["rest_rotations_quat"], dtype=np.float32),
        "joint_names": np.asarray([str(name) for name in static["joint_names"]]),
        "skeleton_hash": np.asarray(str(static["skeleton_hash"])),
    }


def save_asset(output_dir: Path, static: dict, vertices, faces, weights, **extra) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "mesh_asset.npz"
    staged = output.with_suffix(".npz.tmp")
    with staged.open("wb") as file:
        np.savez_compressed(
            file,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int32),
            weights=np.asarray(weights, dtype=np.float32),
            **static,
            **extra,
        )
    os.replace(staged, output)
    print(output, flush=True)


def build_mobj(source: Path, static: dict, output_dir: Path) -> None:
    def optional_array(raw, key):
        if key not in raw:
            return None
        value = raw[key]
        if value.shape == () and value.dtype == object and value.item() is None:
            return None
        return np.asarray(value)

    with np.load(source, allow_pickle=True) as raw:
        vertices = np.asarray(raw["vertices"], dtype=np.float64)
        faces = np.asarray(raw["faces"], dtype=np.int32)
        weights = np.asarray(raw["skin"], dtype=np.float64)
        source_rest = np.asarray(raw["matrix_local"], dtype=np.float64)[:, :3, 3]
        uvs = optional_array(raw, "uvs")
        texture = optional_array(raw, "texture")
        vertex_colors = optional_array(raw, "vertex_colors")
    if source_rest.shape != (len(static["parents"]), 3):
        raise ValueError("Mobjaverse source and target joint counts differ")
    span = float(np.max(np.ptp(vertices, axis=0)))
    if span <= 0:
        raise ValueError("Mobjaverse mesh has zero extent")
    scale = 2.0 / span
    normalized = (vertices - vertices.min(axis=0)) / span
    bias = 0.5 - (normalized.min(axis=0) + normalized.max(axis=0)) * 0.5
    offset = -vertices.min(axis=0) * scale + bias * 2.0 - 1.0
    source_rest = source_rest * scale + offset
    vertices = vertices * scale + offset
    target_rest = rest_global_transforms(
        static["rest_translations"], static["rest_rotations_quat"], static["parents"]
    )[:, :3, 3]
    fit = fit_proper_similarity(source_rest, target_rest)
    if fit["scale"] <= 0 or float(np.max(fit["errors"])) > 0.003:
        raise ValueError(f"Mobjaverse bind skeleton differs from cache: {np.max(fit['errors']):.6f} m")
    vertices = fit["scale"] * (vertices @ fit["rotation"].T) + fit["translation"]
    sums = weights.sum(axis=1, keepdims=True)
    if weights.shape != (len(vertices), len(static["parents"])) or np.any(sums <= 0):
        raise ValueError("Mobjaverse skin weights do not match mesh and skeleton")
    weights /= sums
    extra = {}
    if vertex_colors is not None and vertex_colors.size:
        extra["vertex_colors"] = vertex_colors
    if uvs is not None and uvs.size:
        # The source UVs use the opposite V origin from Blender's image nodes.
        # The official Mobj renderer applies the same flip before texturing.
        blender_uvs = np.asarray(uvs, dtype=np.float32).copy()
        blender_uvs[:, 1] = 1.0 - blender_uvs[:, 1]
        extra["uvs"] = blender_uvs
    if texture is not None and texture.size:
        output_dir.mkdir(parents=True, exist_ok=True)
        pixels = np.clip(texture * 255 if texture.max() <= 1.5 else texture, 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(output_dir / "texture.png")
        extra["texture_files"] = np.asarray(["texture.png"])
    save_asset(output_dir, static, vertices, faces, weights, **extra)


def build_fbx(source: Path, static_path: Path, output_dir: Path, blender: Path) -> None:
    if not blender.is_file():
        raise FileNotFoundError(f"Blender executable: {blender}")
    worker = Path(__file__).with_name("extract_mesh_asset_fbx_blender.py")
    subprocess.run(
        [str(blender), "-b", "--python-exit-code", "1", "--python", str(worker), "--",
         "--fbx", str(source), "--static", str(static_path), "--output-dir", str(output_dir)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original character FBX or Mobjaverse raw_data.npz")
    parser.add_argument("--static", type=Path, required=True, help="Target cache static.npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blender", type=Path, help="Required for FBX sources")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    static_path = args.static.expanduser().resolve()
    if not source.is_file() or not static_path.is_file():
        raise FileNotFoundError(f"Missing source or static: {source}, {static_path}")
    if source.suffix.lower() == ".fbx":
        if args.blender is None:
            parser.error("--blender is required for an FBX source")
        build_fbx(source, static_path, args.output_dir.expanduser().resolve(), args.blender.expanduser().resolve())
    elif source.name == "raw_data.npz":
        build_mobj(source, target_skeleton(static_path), args.output_dir.expanduser().resolve())
    else:
        parser.error("--source must be an original character FBX or Mobjaverse raw_data.npz")


if __name__ == "__main__":
    main()
