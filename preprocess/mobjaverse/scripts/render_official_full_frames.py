#!/usr/bin/env python3
"""Render every source animation frame from the official Mobjaverse NPZ files.

The official dataset checkout is imported read-only. Mesh deformation and textured
mesh construction call the official implementation directly:

    Asset.vertices_with_pose(..., dqs=False)
    render.make_textured_mesh(...)

This wrapper only adds resumable batched I/O, RGB/mask tar packaging, and a
single camera fitted against the union of the complete animation. It never
edits files inside the official checkout and never truncates or pads time.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tarfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Must be selected before importing pyrender / OpenGL through official render.py.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image

@dataclass(frozen=True)
class CameraSpec:
    name: str
    theta_degrees: float
    phi_degrees: float
    target: list[float]
    distance: float
    yfov_radians: float
    camera_to_world: list[list[float]]
    fit_fraction: float
    union_min: list[float]
    union_max: list[float]
    union_extent: list[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--mask-compress-level", type=int, default=1)
    p.add_argument("--lens", type=float, default=32.0)
    p.add_argument("--sensor-size", type=float, default=36.0)
    p.add_argument(
        "--camera-mode",
        choices=("full-motion-axis-fit", "official-fixed"),
        default="full-motion-axis-fit",
    )
    p.add_argument(
        "--fit-fraction",
        type=float,
        default=0.80,
        help="Maximum fraction of image width/height occupied by the union AABB.",
    )
    p.add_argument("--official-distance", type=float, default=3.0)
    p.add_argument("--official-theta", type=float, default=0.0)
    p.add_argument("--official-phi", type=float, default=0.0)
    p.add_argument("--asset-id", action="append", default=[])
    p.add_argument("--asset-list", type=Path)
    p.add_argument("--max-assets", type=int)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--egl-device-id", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--zero-skin-policy",
        choices=("reject", "rest-static"),
        default="reject",
        help="Never claim zero-skin source assets contain renderable animation.",
    )
    return p.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def unwrap_npz_value(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.shape == () and x.dtype == object:
        return x.item()
    return x


def load_official_modules(source_root: Path):
    sys.path.insert(0, str(source_root))
    import render as official_render  # type: ignore
    from src.rig_package.info.asset import Asset  # type: ignore

    return official_render, Asset


def canonical_asset_id(text: str) -> str:
    text = text.strip().replace("\\", "/")
    if not text:
        raise ValueError("empty asset id")
    text = text.rstrip("/").split("/")[-1]
    if not text.isdigit():
        raise ValueError(f"invalid asset id: {text!r}")
    return text.zfill(6)


def dataset_asset_ids(source_root: Path) -> list[str]:
    ids: set[str] = set()
    for path in sorted((source_root / "datalist").rglob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                ids.add(canonical_asset_id(line))
    if not ids:
        raise RuntimeError(f"No assets found below {source_root / 'datalist'}")
    return sorted(ids)


def selected_asset_ids(args: argparse.Namespace) -> list[str]:
    if args.asset_id:
        ids = [canonical_asset_id(x) for x in args.asset_id]
    elif args.asset_list:
        ids = [
            canonical_asset_id(x)
            for x in args.asset_list.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
    else:
        ids = dataset_asset_ids(args.source_root)
    ids = sorted(set(ids))
    ids = [x for i, x in enumerate(ids) if i % args.num_shards == args.shard_index]
    if args.max_assets is not None:
        ids = ids[: args.max_assets]
    return ids


def load_asset(npz_path: Path, Asset):
    with np.load(npz_path, allow_pickle=True) as raw:
        data = {k: unwrap_npz_value(raw[k]) for k in raw.files}
    return Asset(**data)


def prepare_asset(asset) -> tuple[int, bool]:
    if asset.matrix_basis is None:
        raise ValueError("matrix_basis is missing")
    if asset.vertices is None or asset.faces is None:
        raise ValueError("vertices/faces are missing")
    if asset.skin is None:
        raise ValueError("skin is missing")
    # Match the official render.py main() exactly.
    asset.matrix_basis[:, 0, :3, 3] = 0
    asset.normalize_vertices((-1.0, 1.0))
    frame_count = int(asset.matrix_basis.shape[0])
    zero_skin = bool(float(np.abs(asset.skin).sum()) <= 1e-12)
    return frame_count, zero_skin


def deformed_vertices(asset, pose: np.ndarray, zero_skin_policy: str) -> np.ndarray:
    if float(np.abs(asset.skin).sum()) <= 1e-12 and zero_skin_policy == "rest-static":
        # Explicit source-data fallback. It is recorded in complete.json and
        # must not be treated as a valid animated sample by cache construction.
        return np.asarray(asset.vertices)
    # Official Mobjaverse deformation path: LBS, never DQS.
    return asset.vertices_with_pose(matrix_basis=pose, inplace=False, dqs=False)


def union_bounds(asset, zero_skin_policy: str) -> tuple[np.ndarray, np.ndarray, float]:
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    first: np.ndarray | None = None
    max_motion = 0.0
    for pose in asset.matrix_basis:
        vertices = np.asarray(deformed_vertices(asset, pose, zero_skin_policy))
        if not np.isfinite(vertices).all():
            raise ValueError("non-finite deformed vertices")
        if vertices.size == 0:
            raise ValueError("empty deformed vertices")
        lo = np.minimum(lo, vertices.min(axis=0))
        hi = np.maximum(hi, vertices.max(axis=0))
        if first is None:
            first = vertices.copy()
        elif first.shape == vertices.shape:
            max_motion = max(max_motion, float(np.max(np.linalg.norm(vertices - first, axis=1))))
    return lo, hi, max_motion


def aabb_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ],
        dtype=np.float64,
    )


def choose_axis_view(extent: np.ndarray) -> tuple[str, float, float]:
    # Looking along the smallest spatial extent maximizes the projected area.
    # Tie order preserves the official -Y view for conventionally oriented data.
    ranked = sorted(((float(extent[1]), 0), (float(extent[0]), 1), (float(extent[2]), 2)))
    axis = ranked[0][1]
    if axis == 0:
        return "minus_y", 0.0, 0.0
    if axis == 1:
        return "plus_x", 0.0, 90.0
    return "plus_z", 90.0, 0.0


def build_camera(
    official_render,
    lo: np.ndarray,
    hi: np.ndarray,
    camera_mode: str,
    fit_fraction: float,
    lens: float,
    sensor_size: float,
    official_distance: float,
    official_theta: float,
    official_phi: float,
) -> CameraSpec:
    if not (0.1 <= fit_fraction < 1.0):
        raise ValueError("fit_fraction must be in [0.1, 1.0)")
    center = (lo + hi) / 2.0
    extent = hi - lo
    yfov = 2.0 * math.atan(sensor_size / (2.0 * lens))
    if camera_mode == "official-fixed":
        name = "official_fixed"
        theta = official_theta
        phi = official_phi
        distance = official_distance
        target = np.zeros(3, dtype=np.float64)
    else:
        name, theta, phi = choose_axis_view(extent)
        target = center
        unit_scene = official_render.Scene(
            lens=lens,
            dis=1.0,
            theta=math.radians(theta),
            phi=math.radians(phi),
            sensor_size=sensor_size,
        )
        rotation = np.asarray(unit_scene.camera[:3, :3], dtype=np.float64)
        q = (aabb_corners(lo, hi) - target) @ rotation
        tan_half = math.tan(yfov / 2.0) * fit_fraction
        required_x = q[:, 2] + np.abs(q[:, 0]) / tan_half
        required_y = q[:, 2] + np.abs(q[:, 1]) / tan_half
        required_near = q[:, 2] + 0.10
        distance = float(max(0.25, required_x.max(), required_y.max(), required_near.max()))
    scene = official_render.Scene(
        lens=lens,
        dis=distance,
        theta=math.radians(theta),
        phi=math.radians(phi),
        sensor_size=sensor_size,
    )
    camera_to_world = np.asarray(scene.camera, dtype=np.float64)
    camera_to_world[:3, 3] += target
    return CameraSpec(
        name=name,
        theta_degrees=float(theta),
        phi_degrees=float(phi),
        target=target.astype(float).tolist(),
        distance=float(distance),
        yfov_radians=float(yfov),
        camera_to_world=camera_to_world.astype(float).tolist(),
        fit_fraction=float(fit_fraction),
        union_min=lo.astype(float).tolist(),
        union_max=hi.astype(float).tolist(),
        union_extent=extent.astype(float).tolist(),
    )


def encode_rgb(color: np.ndarray, quality: int) -> bytes:
    out = io.BytesIO()
    Image.fromarray(np.asarray(color, dtype=np.uint8), mode="RGB").save(
        out,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
    )
    return out.getvalue()


def encode_mask(depth: np.ndarray, compress_level: int) -> bytes:
    mask = np.asarray(depth > 0, dtype=np.uint8) * 255
    out = io.BytesIO()
    Image.fromarray(mask, mode="L").save(
        out,
        format="PNG",
        compress_level=compress_level,
        optimize=False,
    )
    return out.getvalue()


def add_tar_bytes(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    tf.addfile(info, io.BytesIO(payload))


def output_paths(output_root: Path, asset_id: str) -> tuple[Path, Path, Path]:
    asset_dir = output_root / "assets" / asset_id[:2] / asset_id
    return asset_dir, asset_dir / "view_00.tar", asset_dir / "complete.json"


def is_complete(complete_path: Path, tar_path: Path) -> bool:
    if not complete_path.is_file() or not tar_path.is_file():
        return False
    try:
        value = json.loads(complete_path.read_text(encoding="utf-8"))
        return (
            value.get("status") == "complete"
            and int(value.get("source_frames", -1)) == int(value.get("rendered_frames", -2))
            and int(value.get("tar_bytes", -1)) == tar_path.stat().st_size
        )
    except Exception:
        return False


def render_one(
    args: argparse.Namespace,
    official_render,
    Asset,
    asset_id: str,
) -> dict[str, Any]:
    started = time.time()
    source_npz = args.source_root / "mobjaverse" / asset_id / "raw_data.npz"
    asset_dir, tar_path, complete_path = output_paths(args.output_root, asset_id)
    if not args.overwrite and is_complete(complete_path, tar_path):
        return {"asset_id": asset_id, "status": "skipped_complete"}
    asset_dir.mkdir(parents=True, exist_ok=True)
    # A failed overwrite must not leave a stale completion marker behind.
    if complete_path.exists():
        complete_path.unlink()
    error_path = asset_dir / "error.json"
    if error_path.exists():
        error_path.unlink()
    tmp_tar = tar_path.with_name(f".{tar_path.name}.tmp.{os.getpid()}")
    if tmp_tar.exists():
        tmp_tar.unlink()

    asset = load_asset(source_npz, Asset)
    frame_count, zero_skin = prepare_asset(asset)
    if frame_count <= 0:
        raise ValueError("animation has zero frames")
    if zero_skin and args.zero_skin_policy == "reject":
        raise ValueError("source skin weights are all zero; official LBS cannot animate this mesh")

    lo, hi, max_motion = union_bounds(asset, args.zero_skin_policy)
    camera = build_camera(
        official_render=official_render,
        lo=lo,
        hi=hi,
        camera_mode=args.camera_mode,
        fit_fraction=args.fit_fraction,
        lens=args.lens,
        sensor_size=args.sensor_size,
        official_distance=args.official_distance,
        official_theta=args.official_theta,
        official_phi=args.official_phi,
    )

    import pyrender
    import trimesh

    resolution = int(args.resolution)
    renderer = pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=resolution)
    texture = asset.texture
    uvs = asset.uvs.copy() if asset.uvs is not None else None
    if uvs is not None:
        uvs[:, 1] = 1 - uvs[:, 1]
    if texture is not None and (texture.shape[0] == 0 or texture.shape[1] == 0):
        texture = None
    vertex_colors = asset.vertex_colors
    intensity = 1.5 if texture is None else 7.5
    material = None
    new_texture = None
    new_uvs = None
    inverse_indices = None
    unique_indices = None
    try:
        with tarfile.open(tmp_tar, mode="w") as tf:
            for frame, pose in enumerate(asset.matrix_basis):
                vertices = deformed_vertices(asset, pose, args.zero_skin_policy)
                if texture is not None and uvs is not None:
                    (
                        tri_mesh,
                        material,
                        new_texture,
                        new_uvs,
                        inverse_indices,
                        unique_indices,
                    ) = official_render.make_textured_mesh(
                        vertices=vertices,
                        faces=asset.faces,
                        texture=texture,
                        uvs=uvs,
                        new_uvs=new_uvs,
                        new_texture=new_texture,
                        inverse_indices=inverse_indices,
                        unique_indices=unique_indices,
                        cached_material=material,
                    )
                else:
                    tri_mesh = trimesh.Trimesh(
                        vertices=vertices,
                        faces=asset.faces,
                        vertex_colors=vertex_colors,
                        process=False,
                    )
                render_mesh = pyrender.Mesh.from_trimesh(
                    tri_mesh, smooth=True, material=material
                )
                scene = pyrender.Scene()
                scene.add(render_mesh)
                camera_node = scene.add(
                    pyrender.PerspectiveCamera(yfov=camera.yfov_radians),
                    pose=np.asarray(camera.camera_to_world, dtype=np.float64),
                )
                light_node = scene.add(
                    pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                    pose=np.asarray(camera.camera_to_world, dtype=np.float64),
                )
                scene.main_camera_node = camera_node
                color, depth = renderer.render(
                    scene, flags=pyrender.constants.RenderFlags.OFFSCREEN
                )
                rgb_bytes = encode_rgb(color, args.jpeg_quality)
                mask_bytes = encode_mask(depth, args.mask_compress_level)
                stem = f"{asset_id}.view00.{frame:06d}"
                add_tar_bytes(tf, f"{stem}.jpg", rgb_bytes)
                add_tar_bytes(tf, f"{stem}.png", mask_bytes)
                # Explicitly release per-frame GL scene references.
                scene.remove_node(light_node)
                scene.remove_node(camera_node)
        os.replace(tmp_tar, tar_path)
    finally:
        renderer.delete()
        if tmp_tar.exists():
            tmp_tar.unlink()

    metadata: dict[str, Any] = {
        "status": "complete",
        "asset_id": asset_id,
        "source_npz": str(source_npz),
        "source_frames": frame_count,
        "rendered_frames": frame_count,
        "temporal_policy": {
            "truncate": False,
            "pad": False,
            "repeat_tail": False,
            "resample": False,
        },
        "official_source": {
            "root": str(args.source_root),
            "deformation": "Asset.vertices_with_pose(matrix_basis=pose, inplace=False, dqs=False)",
            "root_translation_removed": True,
            "normalization": [-1.0, 1.0],
        },
        "camera": asdict(camera),
        "render": {
            "resolution": resolution,
            "rgb": {"format": "jpeg", "quality": int(args.jpeg_quality), "subsampling": 0},
            "mask": {
                "format": "png_l8",
                "definition": "depth > 0",
                "compress_level": int(args.mask_compress_level),
            },
            "view_count": 1,
        },
        "source_quality": {
            "zero_skin": zero_skin,
            "zero_skin_policy": args.zero_skin_policy,
            "max_vertex_displacement_from_first_frame": max_motion,
        },
        "tar_path": str(tar_path),
        "tar_bytes": tar_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(complete_path, metadata)
    return metadata


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard configuration")
    if args.resolution < 1 or args.lens <= 0 or args.sensor_size <= 0:
        raise ValueError("resolution, lens and sensor size must be positive")
    if not 1 <= args.jpeg_quality <= 100 or not 0 <= args.mask_compress_level <= 9:
        raise ValueError("invalid image encoding settings")
    os.environ["EGL_DEVICE_ID"] = str(args.egl_device_id)
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    official_render, Asset = load_official_modules(args.source_root)
    ids = selected_asset_ids(args)
    if not ids:
        raise ValueError("No assets selected")
    run_config = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "selected_assets": len(ids),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "egl_device_id": args.egl_device_id,
    }
    atomic_json(args.output_root / f"run_config_shard_{args.shard_index:03d}.json", run_config)

    summary: dict[str, Any] = {
        **run_config,
        "completed": 0,
        "skipped_complete": 0,
        "failed": 0,
        "source_frames_completed": 0,
        "failures": [],
    }
    summary_path = args.output_root / f"summary_shard_{args.shard_index:03d}.json"
    for index, asset_id in enumerate(ids, 1):
        try:
            result = render_one(args, official_render, Asset, asset_id)
            status = result["status"]
            if status == "skipped_complete":
                summary["skipped_complete"] += 1
            else:
                summary["completed"] += 1
                summary["source_frames_completed"] += int(result["source_frames"])
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(ids),
                        "asset_id": asset_id,
                        "status": status,
                        "frames": result.get("source_frames"),
                        "elapsed_seconds": result.get("elapsed_seconds"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            summary["failed"] += 1
            failure = {
                "asset_id": asset_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            summary["failures"].append(failure)
            asset_dir, _, _ = output_paths(args.output_root, asset_id)
            atomic_json(asset_dir / "error.json", failure)
            print(json.dumps({"asset_id": asset_id, "status": "failed", "error": repr(exc)}), flush=True)
        atomic_json(summary_path, summary)
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(summary_path, summary)
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
