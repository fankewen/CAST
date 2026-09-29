"""Evaluate source-video to target-skeleton motion on Mixamo directed pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.cache_formats import skeleton_id_from_static
from inference.video2motion_unlabeled import (
    inference_window_starts,
    iter_feature_windows,
    iter_mask_bbox_crops,
    static_conditioning,
)
from utils.config_utils import instantiate_from_config, load_yaml_config
from utils.mesh_skinning import rest_global_transforms
from utils.model_inputs import normalize_model_skeleton_inputs
from utils.motion_metrics import topocap_protocol_metrics
from utils.motion_visualization import (
    local_rotation_fk,
    motion_output_dir,
    write_unlabeled_motion_npz,
)
from utils.rotation import (
    project_rotmat_to_so3,
    rot6d_to_rotmat_tensor,
    rotmat_to_rot6d_tensor,
)


def _pairs(cache_root: Path) -> list[dict]:
    path = cache_root / "pairs.jsonl"
    with path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows:
        raise ValueError(f"No cross-skeleton pairs in {path}")
    return rows


def _frames(directory: Path, count: int, mode: str):
    for index in range(count):
        with Image.open(directory / f"{index:06d}.png") as image:
            yield np.asarray(image.convert(mode))


def _reference_positions(path: Path, frames: int, joints: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as reference:
        transforms = np.asarray(reference["global_transforms"], dtype=np.float32)
    if transforms.shape != (frames, joints, 4, 4):
        raise ValueError(f"{path} has unexpected global_transforms shape {transforms.shape}")
    positions = transforms[:, :, :3, 3]
    return positions - positions[:, :1]


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    keys = rows[0]["metrics"]
    return {
        key: float(np.mean([row["metrics"][key] for row in rows]))
        for key in keys
    }


@torch.no_grad()
def run(
    cache_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    stride: int,
    device_name: str,
) -> None:
    cfg = load_yaml_config(config_path)
    window = int(cfg.get("eval", {}).get("window", cfg["train"]["seq_len"]))
    if not 1 <= stride <= window:
        raise ValueError(f"stride must be in [1, {window}], got {stride}")
    max_joints = int(cfg["data"]["max_joints"])
    image_size = int(cfg["data"]["image_size"])
    bbox_margin = int(cfg["data"].get("bbox_margin", 5))
    device = torch.device(device_name)
    model = instantiate_from_config(cfg["model"]).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(checkpoint["model_state"])

    pair_results = []
    pairs = _pairs(cache_root)
    for row in pairs:
        pair_id = str(row["pair_id"])
        source = str(row["source"])
        target_id = str(row["target"])
        frames = int(row["frames"])
        static_path = cache_root / row["target_static"]
        static = np.load(static_path, allow_pickle=True).item()
        if skeleton_id_from_static(static) != target_id:
            raise ValueError(f"{pair_id}: target_static does not match target")
        parents = np.asarray(static["parents"], dtype=np.int64)
        offsets = np.asarray(static["rest_translations"], dtype=np.float32)
        rgb = _frames(cache_root / row["source_rgb"], frames, "RGB")
        masks = _frames(cache_root / row["source_mask"], frames, "L")
        features = iter_mask_bbox_crops(
            rgb, masks, size=image_size, bbox_margin=bbox_margin
        )

        rotation_sum = np.zeros((frames, len(parents), 3, 3), dtype=np.float64)
        counts = np.zeros(frames, dtype=np.float64)
        overlap = window - stride
        for start, valid, image_window in iter_feature_windows(
            features, frames, window, overlap
        ):
            batch, joints = static_conditioning(
                static_path, image_window, device,
                max_joints=max_joints, valid_length=valid,
            )
            output = model(
                normalize_model_skeleton_inputs(batch),
                return_reconstructed_positions=False,
            )
            rotation = rot6d_to_rotmat_tensor(
                output["pred_rot6d"][0, :valid, :joints].float().cpu()
            ).numpy()
            rotation_sum[start:start + valid] += rotation
            counts[start:start + valid] += 1
        if np.any(counts == 0):
            raise RuntimeError(f"Sliding windows did not cover {pair_id}")
        rotation = project_rotmat_to_so3(
            torch.from_numpy(rotation_sum / counts[:, None, None, None]).float()
        ).numpy()
        predicted = local_rotation_fk(rotation, offsets, parents)
        reference = _reference_positions(
            cache_root / row["target_reference"], frames, len(parents)
        )
        rest = rest_global_transforms(
            offsets, static["rest_rotations_quat"], parents
        )[:, :3, 3]
        metrics = topocap_protocol_metrics(predicted, reference, rest)

        pair_dir = motion_output_dir(output_dir, target_id, pair_id)
        rot6d = rotmat_to_rot6d_tensor(torch.from_numpy(rotation)).numpy()
        np.save(pair_dir / "root_global_rot6d.npy", rot6d[:, 0])
        np.save(pair_dir / "nonroot_local_rot6d.npy", rot6d[:, 1:])
        write_unlabeled_motion_npz(
            pair_dir / "predicted_motion.npz",
            checkpoint_path=checkpoint_path,
            skeleton_id=target_id,
            sequence=pair_id,
            static=static,
            fps=float(row["fps"]),
            rotation_matrix=rotation,
            fk_position=predicted,
        )
        pair_results.append({
            "pair_id": pair_id,
            "source": source,
            "target": target_id,
            "clip": str(row["clip"]),
            "frames": frames,
            "metrics": metrics,
        })
        print(f"[{len(pair_results)}/{len(pairs)}] {pair_id}: {metrics}", flush=True)

    by_target = {}
    for result in pair_results:
        by_target.setdefault(result["target"], []).append(result)
    per_target = {target: _mean_metrics(rows) for target, rows in by_target.items()}
    overall = {
        key: float(np.mean([values[key] for values in per_target.values()]))
        for key in next(iter(per_target.values()))
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({
            "aggregation": "complete_pair_then_equal_target",
            "evaluated_pair_count": len(pair_results),
            "per_pair": pair_results,
            "per_target": per_target,
            "metrics": overall,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved cross-skeleton metrics: {output_dir / 'metrics.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(
        args.cache_root, args.config, args.checkpoint, args.output_dir,
        stride=args.stride, device_name=args.device,
    )


if __name__ == "__main__":
    main()
