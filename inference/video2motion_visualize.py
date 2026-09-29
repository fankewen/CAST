#!/usr/bin/env python3
"""Convert an existing unlabeled Video2Motion prediction into a driven-mesh video."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.cache_formats import skeleton_id_from_static
from inference.video2motion_unlabeled import center_crop_box, iter_video_rgb, video_metadata
from utils.rotation import rot6d_to_rotmat_tensor
from utils.motion_visualization import (
    local_rotation_fk,
    motion_output_dir,
    video_output,
    render_unlabeled_video,
    write_unlabeled_motion_npz,
)

def required_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def main() -> None:
    parser = argparse.ArgumentParser(description="Render an existing Video2Motion prediction")
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=512)
    parser.add_argument("--center-crop-scale", type=float, default=0.62)
    parser.add_argument("--lighting-preset", choices=("zoo", "mobj", "mixamo"), default="zoo")
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--world-strength", type=float)
    parser.add_argument("--light-energies", type=float, nargs="+")
    parser.add_argument("--show-skeleton-overlay", action="store_true")
    parser.add_argument("--keep-mesh-frames", action="store_true", help=argparse.SUPPRESS)  # No-op: renders always clean their frames.
    args = parser.parse_args()
    input_video = required_file(args.input_video)
    static_path = required_file(args.static)
    mesh_path = required_file(args.mesh_path)
    checkpoint = required_file(args.checkpoint)
    prediction_dir = args.prediction_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not prediction_dir.is_dir():
        raise NotADirectoryError(prediction_dir)
    prefix = input_video.stem
    static = np.load(static_path, allow_pickle=True).item()
    skeleton_id = skeleton_id_from_static(static)
    nested_prediction = motion_output_dir(prediction_dir, skeleton_id, prefix)
    if (nested_prediction / f"{prefix}_root_global_rot6d.npy").is_file():
        prediction_dir = nested_prediction
    output_dir = motion_output_dir(output_dir, skeleton_id, prefix)
    root = np.asarray(np.load(required_file(prediction_dir / f"{prefix}_root_global_rot6d.npy"), allow_pickle=False), dtype=np.float32)
    nonroot = np.asarray(np.load(required_file(prediction_dir / f"{prefix}_nonroot_local_rot6d.npy"), allow_pickle=False), dtype=np.float32)
    metadata = json.loads(required_file(prediction_dir / f"{prefix}_metadata.json").read_text(encoding="utf-8"))
    if root.ndim != 2 or root.shape[1] != 6:
        raise ValueError(f"Root prediction must be [T,6], got {root.shape}")
    if nonroot.ndim != 3 or nonroot.shape[0] != root.shape[0] or nonroot.shape[2] != 6:
        raise ValueError(f"Non-root prediction must be [T,J-1,6], got {nonroot.shape}")
    static = np.load(static_path, allow_pickle=True).item()
    rot6d = np.concatenate((root[:, None], nonroot), axis=1)
    parents = np.asarray(static["parents"], dtype=np.int64)
    if rot6d.shape[1] != len(parents):
        raise ValueError(f"Prediction/static joint mismatch: {rot6d.shape[1]} != {len(parents)}")
    rotation = rot6d_to_rotmat_tensor(torch.from_numpy(rot6d)).numpy()
    fk = local_rotation_fk(rotation, np.asarray(static["rest_translations"], dtype=np.float32), parents).astype(np.float32)
    decoded = video_metadata(input_video)
    if int(decoded["source_frame_count"]) < len(fk):
        raise ValueError("Input video has fewer frames than prediction")
    fps = float(metadata.get("output_fps", decoded["output_fps"]))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid output FPS: {fps}")
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_path = output_dir / f"{prefix}_predicted_motion.npz"
    fk_path = output_dir / f"{prefix}_pred_fk_camera_root_centered.npy"
    output_video = output_dir / "visualization.mp4"
    np.save(fk_path, fk)
    write_unlabeled_motion_npz(motion_path, checkpoint_path=checkpoint, skeleton_id=skeleton_id_from_static(static), sequence=prefix, static=static, fps=fps, rotation_matrix=rotation, fk_position=fk)
    crop_box = center_crop_box((int(decoded["width"]), int(decoded["height"])), args.center_crop_scale)
    with video_output(output_dir) as (staged_video, mesh_frame_dir):
        lighting_args = []
        if args.light_energies is not None:
            lighting_args += ["--light-energies", *(str(value) for value in args.light_energies)]
        if args.lighting_preset == "mobj":
            if args.world_strength is not None or args.exposure != 0:
                raise ValueError("Mobjaverse uses pyrender; adjust --light-energies")
        else:
            lighting_args += ["--exposure", str(args.exposure)]
            if args.world_strength is not None:
                lighting_args += ["--world-strength", str(args.world_strength)]
        render_report = render_unlabeled_video(rgb_frames=iter_video_rgb(input_video, len(fk)), fk_position=fk, static=static, mesh_path=mesh_path, motion_path=motion_path, mesh_frame_dir=mesh_frame_dir, crop_box=crop_box, output_path=staged_video, fps=fps, panel_size=args.panel_size, show_skeleton_overlay=args.show_skeleton_overlay, lighting_preset=args.lighting_preset, lighting_args=lighting_args, background=args.background)
    report = {"input_video": str(input_video), "prediction_dir": str(prediction_dir), "static": str(static_path), "mesh_path": str(mesh_path), "checkpoint": str(checkpoint), "output_video": str(output_video), "motion": str(motion_path), "fk_position": str(fk_path), "render": render_report}
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
