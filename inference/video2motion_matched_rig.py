"""Matched-rig evaluation over complete labeled validation motions."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cache import _backend_cfg, _normalize_splits, _source_configs
from data.cache_reader import build_cache_dataset
from utils.batching import (collate_single_anyspecies_padded, iter_worker_parallel_batches)
from utils.model_inputs import model_forward_inputs
from utils.motion_metrics import topocap_protocol_metrics
from utils.config_utils import instantiate_from_config, load_yaml_config
from utils.config_validation import require_bool, require_positive_int
from utils.render_skeleton_rgb_pose_compare import render_frame_sources
from utils.motion_visualization import (
    local_rotation_fk,
    motion_output_dir,
    video_output,
)
from utils.rotation import (
    project_rotmat_to_so3,
    rot6d_to_rotmat_tensor,
    rotmat_to_rot6d_tensor,
)


def _validation_datasets(
    data_cfg: Mapping,
    *,
    window: int,
    stride: int,
    selected_names: set[str] | None,
):
    """Build the configured validation splits as deterministic sliding windows."""
    datasets = {}
    available = set()
    for source_name, source_cfg in _source_configs(data_cfg).items():
        backend_cfg = _backend_cfg(source_cfg)
        raw_validation = source_cfg.get("validation", {source_name: {}})
        if not isinstance(raw_validation, Mapping) or not raw_validation:
            raise ValueError(
                f"data.sources.{source_name}.validation must be a non-empty mapping"
            )
        for raw_name, raw_cfg in raw_validation.items():
            validation_name = str(raw_name)
            available.add(validation_name)
            if selected_names is not None and validation_name not in selected_names:
                continue
            if raw_cfg is None:
                raw_cfg = {}
            if not isinstance(raw_cfg, Mapping):
                raise TypeError(
                    f"data.sources.{source_name}.validation.{validation_name} "
                    "must be a mapping"
                )
            raw_cfg = dict(raw_cfg)
            source_splits = _normalize_splits(
                raw_cfg.pop("source_splits", None),
                f"data.sources.{source_name}.validation.{validation_name}.source_splits",
            )
            validation_cfg = {**backend_cfg, **raw_cfg, "eval_stride": stride}
            datasets[validation_name] = build_cache_dataset(
                validation_cfg,
                "test",
                window,
                first_window_eval=False,
                complete_motion_eval=False,
                source_splits=source_splits,
                eval_ratio=1.0 if source_splits is not None else None,
            )
    if selected_names is not None:
        unknown = sorted(selected_names - available)
        if unknown:
            raise KeyError(
                f"Unknown validation sources {unknown}; available={sorted(available)}"
            )
    if not datasets:
        raise ValueError("No validation datasets were selected")
    return datasets


def _new_motion(batch, index: int, joint_count: int):
    motion_frames = int(batch["motion_frames"][index])
    if motion_frames < 1:
        raise ValueError(f"Invalid motion length {motion_frames}")
    parents = batch["parent_a"][index, :joint_count].detach().cpu().numpy().astype(np.int64)
    offsets = batch["offset_a"][index, :joint_count].detach().cpu().numpy().astype(np.float32)
    return {
        "rel": str(batch["rel"][index]),
        "skeleton_id": str(batch["species"][index]),
        "motion_frames": motion_frames,
        "motion_offset": int(batch["motion_offset"][index]),
        "joint_count": joint_count,
        "parents": parents,
        "offsets": offsets,
        "rot_sum": np.zeros((motion_frames, joint_count, 3, 3), dtype=np.float64),
        "gt_rot_sum": np.zeros((motion_frames, joint_count, 3, 3), dtype=np.float64),
        "gt_position": np.zeros((motion_frames, joint_count, 3), dtype=np.float32),
        "has_gt_position": "position" in batch,
        "count": np.zeros((motion_frames,), dtype=np.float64),
        "frame_sources": [None] * motion_frames,
        "camera_root_position": np.zeros((motion_frames, 3), dtype=np.float32),
        "rest_position": batch["ref_position"][index, :joint_count].detach().cpu().numpy().astype(np.float32),
        "camera_intrinsics": batch["camera_intrinsics"][index].detach().cpu().numpy().astype(np.float32),
        "window_starts": [],
    }


def _append_window(storage, batch, output, index: int) -> None:
    rel = str(batch["rel"][index])
    if rel != storage["rel"]:
        raise ValueError(f"Motion accumulation mismatch: {rel} != {storage['rel']}")
    joint_count = int(batch["J_valid"][index])
    valid_mask = batch["frame_valid_mask"][index].bool()
    frame_count = int(valid_mask.sum())
    start = int(batch["window_start"][index])
    end = min(start + frame_count, storage["motion_frames"])
    if joint_count != storage["joint_count"] or not 0 <= start < end:
        raise ValueError(f"Invalid window [{start},{end}) for {rel}")
    pred = rot6d_to_rotmat_tensor(output["pred_rot6d"][index, valid_mask, :joint_count].float()).cpu().numpy()[: end-start]
    gt = rot6d_to_rotmat_tensor(batch["rot6d_a"][index, valid_mask, :joint_count].float()).cpu().numpy()[: end-start]
    storage["rot_sum"][start:end] += pred.astype(np.float64)
    storage["gt_rot_sum"][start:end] += gt.astype(np.float64)
    if "position" in batch:
        storage["gt_position"][start:end] = (
            batch["position"][index, valid_mask, :joint_count]
            .detach().cpu().numpy()[: end-start]
        )
    storage["count"][start:end] += 1.0
    members = batch["rgb_member_names"][index][: end-start]
    tar_path = str(batch["rgb_tar_path"][index])
    for offset, member in enumerate(members):
        if member is not None:
            storage["frame_sources"][start + offset] = (tar_path, str(member))
    storage["camera_root_position"][start:end] = batch["camera_root_position"][index, valid_mask].detach().cpu().numpy()[: end-start]
    storage["window_starts"].append(start)


def _complete_motion_metrics(storage, pred_matrix, gt_matrix):
    """Compute metrics once on the fully assembled animation."""
    pred_position = local_rotation_fk(pred_matrix, storage["offsets"], storage["parents"])
    if storage.get("has_gt_position", False):
        gt_position = storage["gt_position"]
    else:
        gt_position = local_rotation_fk(gt_matrix, storage["offsets"], storage["parents"])
    return topocap_protocol_metrics(
        pred_position,
        gt_position,
        storage["rest_position"],
    )


def _save_motion(output_dir, storage, validation_name, cfg, visualize):
    missing = np.flatnonzero(storage["count"] == 0)
    if len(missing):
        raise RuntimeError(f"Sliding windows did not cover {storage['rel']}: {missing[:20].tolist()}")
    pred_matrix = project_rotmat_to_so3(torch.from_numpy(storage["rot_sum"] / storage["count"][:, None, None, None]).float()).numpy()
    gt_matrix = project_rotmat_to_so3(torch.from_numpy(storage["gt_rot_sum"] / storage["count"][:, None, None, None]).float()).numpy()
    motion_metrics = _complete_motion_metrics(storage, pred_matrix, gt_matrix)
    rot6d = rotmat_to_rot6d_tensor(torch.from_numpy(pred_matrix)).numpy()
    target = motion_output_dir(Path(output_dir) / validation_name, storage["skeleton_id"], storage["rel"])
    np.save(target / "root_global_rot6d.npy", rot6d[:, 0])
    np.save(target / "nonroot_local_rot6d.npy", rot6d[:, 1:])
    np.save(target / "frame_indices.npy", np.arange(storage["motion_frames"], dtype=np.int64) + storage["motion_offset"])
    np.save(target / "window_starts.npy", np.asarray(storage["window_starts"], dtype=np.int64))
    if visualize:
        if any(source is None for source in storage["frame_sources"]):
            raise RuntimeError(f"Missing RGB frame source for {storage['rel']}")
        visual_cfg = cfg["eval"].get("visualization", {})
        gt_position = storage["gt_position"] if storage.get("has_gt_position", False) else local_rotation_fk(gt_matrix, storage["offsets"], storage["parents"])
        visual_rotation = pred_matrix.copy()
        if require_bool(visual_cfg.get("use_gt_root_motion", False), "eval.visualization.use_gt_root_motion"):
            visual_rotation[:, 0] = gt_matrix[:, 0]
        pred_position = local_rotation_fk(visual_rotation, storage["offsets"], storage["parents"])
        with video_output(target) as (output_video, _):
            render_frame_sources(
                storage["frame_sources"], gt_position, pred_position,
                storage["parents"], [f"joint_{i}" for i in range(storage["joint_count"])],
                output_video, fps=float(visual_cfg.get("fps", 30.0)),
                elev=float(visual_cfg.get("elev", 18.0)), azim=float(visual_cfg.get("azim", 60.0)),
                clip=storage["rel"], camera_root_position=storage["camera_root_position"],
                camera_intrinsics=storage["camera_intrinsics"],
            )
    print(f"[{validation_name}] saved {storage['rel']}: {storage['motion_frames']} frames", flush=True)
    return motion_metrics


@torch.no_grad()
def run(
    config_path,
    checkpoint_path,
    output_dir,
    *,
    stride: int = 20,
    validation_sources: set[str] | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device_name: str | None = None,
    visualize_override: bool | None = None,
    metrics_output_path: str | None = None,
    input_stride: int = 1,
) -> None:
    cfg = load_yaml_config(config_path)
    visual_cfg = cfg.get("eval", {}).get("visualization", {})
    visualize = (
        require_bool(visual_cfg.get("enabled", False), "eval.visualization.enabled")
        if visualize_override is None else bool(visualize_override)
    )
    window = require_positive_int(cfg.get("eval", {}).get("window", 81), "eval.window")
    stride = require_positive_int(stride, "stride")
    input_stride = require_positive_int(input_stride, "input_stride")
    if stride > window:
        raise ValueError(f"stride must be in [1,{window}], got {stride}")
    device = torch.device(
        device_name
        if device_name is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    datasets = _validation_datasets(
        cfg["data"],
        window=window,
        stride=stride,
        selected_names=validation_sources,
    )
    eval_batch_size = require_positive_int(
        cfg.get("eval", {}).get("batch_size", 1)
        if batch_size is None
        else batch_size,
        "eval.batch_size",
    )
    worker_count = int(
        cfg.get("eval", {}).get("num_workers", 0)
        if num_workers is None
        else num_workers
    )
    if worker_count < 0:
        raise ValueError("num_workers must be non-negative")
    prefetch_factor = require_positive_int(
        cfg.get("eval", {}).get("prefetch_factor", 2),
        "eval.prefetch_factor",
    )
    max_joints = require_positive_int(
        cfg["data"].get("max_joints"), "data.max_joints"
    )
    dynamic_max_joints = require_bool(
        cfg["data"].get("dynamic_batch_max_joints", False),
        "data.dynamic_batch_max_joints",
    )
    collate_single = functools.partial(
        collate_single_anyspecies_padded,
        max_joints=max_joints,
        dynamic_max_joints=dynamic_max_joints,
    )
    model = instantiate_from_config(cfg["model"]).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(checkpoint["model_state"])
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for validation_name, dataset in datasets.items():
        print(
            f"[{validation_name}] full validation inference: "
            f"window={window}, stride={stride}, samples={len(dataset)}",
            flush=True,
        )
        species_metrics = {}
        animation_metrics = []
        loader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=device.type == "cuda",
            persistent_workers=worker_count > 0,
            prefetch_factor=prefetch_factor if worker_count > 0 else None,
            collate_fn=collate_single,
        )
        current = None
        for batch in iter_worker_parallel_batches(
            loader, eval_batch_size, drop_last=False
        ):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            if input_stride > 1:
                # Sparse visual observations for completion: one observed
                # frame every N frames, while the model still predicts every
                # frame in the complete animation. Align to absolute motion
                # indices so adjacent windows use the same observation grid.
                local = torch.arange(
                    batch["frame_valid_mask"].shape[1], device=device
                )[None, :]
                window_start = torch.as_tensor(
                    batch["window_start"], device=device, dtype=torch.long
                )
                absolute = window_start[:, None] + local
                observed = absolute.remainder(input_stride).eq(0)
                observed &= batch["frame_valid_mask"].bool()
                batch["frame_observed_mask"] = observed
            output = model(
                model_forward_inputs(batch, cfg),
                return_reconstructed_positions=False,
            )
            for index, rel in enumerate(batch["rel"]):
                rel = str(rel)
                if current is not None and current["rel"] != rel:
                    motion_metrics = _save_motion(output_dir, current, validation_name, cfg, visualize)
                    species_metrics.setdefault(current["skeleton_id"], []).append(motion_metrics)
                    animation_metrics.append({
                        "animation": current["rel"],
                        "species": current["skeleton_id"],
                        "frames": current["motion_frames"],
                        "metrics": motion_metrics,
                    })
                    current = None
                joint_count = int(batch["J_valid"][index])
                if current is None:
                    current = _new_motion(batch, index, joint_count)
                _append_window(current, batch, output, index)
        if current is not None:
            motion_metrics = _save_motion(output_dir, current, validation_name, cfg, visualize)
            species_metrics.setdefault(current["skeleton_id"], []).append(motion_metrics)
            animation_metrics.append({
                "animation": current["rel"],
                "species": current["skeleton_id"],
                "frames": current["motion_frames"],
                "metrics": motion_metrics,
            })

        per_species = {}
        for species, motion_list in species_metrics.items():
            keys = sorted({key for metrics in motion_list for key in metrics})
            per_species[species] = {
                key: float(np.mean([metrics[key] for metrics in motion_list if key in metrics]))
                for key in keys
            }
        species_names = sorted(per_species)
        metric_names = sorted({key for metrics in per_species.values() for key in metrics})
        overall = {
            key: float(np.mean([per_species[species][key] for species in species_names if key in per_species[species]]))
            for key in metric_names
        }
        all_summaries[validation_name] = {
            "aggregation": "complete_animation_then_equal_species",
            "evaluated_animation_count": int(sum(len(v) for v in species_metrics.values())),
            "evaluated_species_count": len(species_names),
            "per_animation": animation_metrics,
            "per_species": per_species,
            "metrics": overall,
        }
        print(
            json.dumps(
                {"validation_source": validation_name, "metrics": overall},
                sort_keys=True,
            ),
            flush=True,
        )

    metrics_path = Path(metrics_output_path).expanduser().resolve() if metrics_output_path else output_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregation": "complete_animation_then_equal_species",
        "validation_sources": all_summaries,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved complete-animation metrics: {metrics_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run complete matched-rig validation inference with overlapping windows; "
            "the default 81-frame window advances by 20 frames"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument(
        "--validation-source",
        action="append",
        dest="validation_sources",
        help="Validation name to run; repeat the option to select several. Default: all",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--metrics-output", help="JSON path for complete-animation metrics (default: OUTPUT_DIR/metrics.json)")
    parser.add_argument("--input-stride", type=int, default=1, help="Observe one input frame every N frames; N=1 uses all frames")
    args = parser.parse_args()
    run(
        args.config,
        args.checkpoint,
        args.output_dir,
        stride=args.stride,
        validation_sources=(
            None
            if args.validation_sources is None
            else set(args.validation_sources)
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        visualize_override=args.visualize,
        metrics_output_path=args.metrics_output,
        input_stride=args.input_stride,
    )


if __name__ == "__main__":
    main()
