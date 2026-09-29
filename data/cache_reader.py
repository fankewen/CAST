import json
import io
import os
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, IterableDataset, get_worker_info

from models.dinov3_infer import build_transform
from utils.config_validation import require_bool, require_int, require_positive_int
from data.cache_formats import CACHE_TAR_INVENTORY_FILENAME


# Canonical coordinate-system tag stored by every supported camera cache.
CACHE_CAMERA_COORDINATE_SYSTEM = "opencv_right_handed_ydown_zforward_meters"


def cache_rgb_member_names(
    member_clip_name: str,
    local_frames: Sequence[int],
    image_ext: str,
    pad_length: int = 0,
) -> List[Optional[str]]:
    """Build physical tar member names, independent of unique motion labels."""
    return [
        f"{member_clip_name}.{int(local_frame):06d}{image_ext}"
        for local_frame in local_frames
    ] + [None] * pad_length


def mask_bbox_with_margin(mask: np.ndarray, margin: int = 5) -> tuple[int, int, int, int]:
    """Return an image-clipped, PIL-style bbox around nonzero mask pixels.

    The returned coordinates are ``(left, top, right, bottom)`` with exclusive
    right/bottom edges.  The foreground bbox is expanded by exactly ``margin``
    pixels on every available side and clipped to the image boundary.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.any(mask != 0, axis=-1)
    elif mask.ndim == 2:
        mask = mask != 0
    else:
        raise ValueError(f"Cache mask must have shape [H,W] or [H,W,C], got {mask.shape}")
    margin = require_int(margin, "bbox_margin")
    if margin < 0:
        raise ValueError(f"bbox_margin must be non-negative, got {margin}")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cache foreground mask is empty")
    height, width = mask.shape
    left = max(int(xs.min()) - margin, 0)
    top = max(int(ys.min()) - margin, 0)
    right = min(int(xs.max()) + margin + 1, width)
    bottom = min(int(ys.max()) + margin + 1, height)
    return left, top, right, bottom


def crop_rgb_by_mask(rgb: Image.Image, mask: Image.Image, margin: int = 5) -> Image.Image:
    """Crop RGB to the expanded mask bbox, or keep the full frame if empty.

    A small number of source-cache masks are valid PNG files containing no
    foreground pixels.  Such a mask has no mathematically defined bbox.
    Returning the full RGB frame is the only deterministic fallback that
    preserves frame/label alignment without inventing a false foreground box.
    Non-empty masks always follow the exact margin contract.
    """
    rgb = rgb.convert("RGB")
    mask_array = np.asarray(mask.convert("L"))
    if rgb.size != mask.size:
        raise ValueError(f"RGB/mask size mismatch: rgb={rgb.size}, mask={mask.size}")
    if not np.any(mask_array):
        return rgb.copy()
    return rgb.crop(mask_bbox_with_margin(mask_array, margin))


def crop_box_from_mask(
    rgb: Image.Image, mask: Image.Image, margin: int = 5
) -> tuple[int, int, int, int]:
    """Return the fixed preprocessing crop box for an RGB/mask pair.

    Empty masks retain the complete image, matching ``crop_rgb_by_mask``.
    This box is known preprocessing metadata (from the input segmentation),
    never a model prediction target.
    """
    if rgb.size != mask.size:
        raise ValueError(f"RGB/mask size mismatch: rgb={rgb.size}, mask={mask.size}")
    mask_array = np.asarray(mask.convert("L"))
    if not np.any(mask_array):
        width, height = rgb.size
        return 0, 0, width, height
    return mask_bbox_with_margin(mask_array, margin)


def discover_clip_tar_paths(
    data_dir: str,
    clip_names: Sequence[str],
    require_all: bool = True,
) -> Dict[str, str]:
    """Recover missing motion→tar metadata by scanning WebDataset headers."""
    unresolved = set(str(name) for name in clip_names)
    discovered: Dict[str, str] = {}
    if not unresolved:
        return discovered
    for tar_path in sorted(Path(data_dir).glob("images/*.tar")):
        with tarfile.open(tar_path, "r") as archive:
            for member in archive:
                basename = Path(member.name).name
                if not basename.lower().endswith((".jpg", ".jpeg")):
                    continue
                frame_stem = Path(basename).stem
                if "." not in frame_stem:
                    continue
                clip_name = frame_stem.rsplit(".", 1)[0]
                if clip_name in unresolved:
                    discovered[clip_name] = os.path.relpath(tar_path, data_dir)
                    unresolved.remove(clip_name)
                    if not unresolved:
                        return discovered
    if unresolved and require_all:
        raise FileNotFoundError(
            f"Could not locate RGB tar members for clips: {sorted(unresolved)[:10]}"
        )
    return discovered


def scan_clip_tar_inventory(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """Inventory the actual paired RGB/mask frames stored in image tar shards.

Some early caches have optimistic ``meta.json`` entries: a motion
    may claim N images even though its shard contains fewer (or none), and some
    motions omit the tar reference altogether.  Training must follow the tar
    members that can really be decoded, not those metadata claims.

    The current cache contract keeps every clip in one tar.  We enforce that
    here because ``_load_cropped_frame`` opens one archive per item.  Only a
    contiguous RGB+mask prefix starting at frame zero is usable: retaining a
    later isolated frame would break the label/frame alignment.
    """
    raw: Dict[tuple[str, str], Dict[str, Any]] = {}
    for tar_path in sorted(Path(data_dir).glob("images/*.tar")):
        relative_tar = os.path.relpath(tar_path, data_dir)
        with tarfile.open(tar_path, "r") as archive:
            for member in archive:
                basename = Path(member.name).name
                extension = Path(basename).suffix.lower()
                if extension not in (".jpg", ".jpeg", ".png"):
                    continue
                frame_stem = Path(basename).stem
                if "." not in frame_stem:
                    continue
                clip_name, frame_text = frame_stem.rsplit(".", 1)
                if not frame_text.isdigit():
                    continue
                entry = raw.setdefault(
                    (relative_tar, clip_name),
                    {"tar": relative_tar, "rgb": set(), "mask": set(), "rgb_ext": None},
                )
                frame_index = int(frame_text)
                if extension == ".png":
                    entry["mask"].add(frame_index)
                else:
                    entry["rgb"].add(frame_index)
                    if entry["rgb_ext"] not in (None, extension):
                        raise ValueError(
                            f"Cache clip {clip_name} mixes RGB extensions "
                            f"{entry['rgb_ext']} and {extension}"
                        )
                    entry["rgb_ext"] = extension

    clip_counts = Counter(clip_name for _, clip_name in raw)
    inventory: Dict[str, Dict[str, Any]] = {}
    for (relative_tar, clip_name), entry in raw.items():
        paired = entry["rgb"] & entry["mask"]
        usable_frames = 0
        while usable_frames in paired:
            usable_frames += 1
        # A cache can contain multiple camera views with identical clip/member
        # names in different shards. Use the bare clip name only when it is
        # globally unique; otherwise qualify it by its tar path.
        inventory_key = (
            clip_name
            if clip_counts[clip_name] == 1
            else f"{relative_tar}::{clip_name}"
        )
        inventory[inventory_key] = {
            "tar": entry["tar"],
            "frames": usable_frames,
            "rgb_ext": entry["rgb_ext"] or ".jpg",
            "rgb_members": len(entry["rgb"]),
            "mask_members": len(entry["mask"]),
        }
    return inventory


def _tar_shard_signature(data_dir: str) -> List[Dict[str, Any]]:
    """Return a cheap signature used to invalidate the persistent inventory."""
    signature = []
    for tar_path in sorted(Path(data_dir).glob("images/*.tar")):
        stat = tar_path.stat()
        signature.append(
            {
                "path": os.path.relpath(tar_path, data_dir),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return signature


def _read_inventory_clips(path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    """Return the clips mapping of an inventory file, or None if unreadable."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, ValueError, TypeError):
        return None
    clips = payload.get("clips") if isinstance(payload, dict) else None
    return clips if isinstance(clips, dict) else None


def _valid_inventory_payload(payload: Any, signature: List[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("tar_signature") == signature
        and isinstance(payload.get("clips"), dict)
    )


def load_or_build_clip_tar_inventory(
    data_dir: str,
    inventory_mode: str = "scan",
) -> tuple[Dict[str, Dict[str, Any]], bool]:
    """Load the cache inventory, building and persisting it when absent.

    A cached payload is accepted when its tar signature and clips mapping are
    valid; the file name alone is never treated as sufficient.
    """
    import fcntl

    mode = str(inventory_mode).strip().lower()
    if mode not in {"scan", "metadata", "cached"}:
        raise ValueError(
            "inventory_mode must be scan, metadata, or cached, "
            f"got {inventory_mode!r}"
        )
    data_path = Path(data_dir)
    cache_path = data_path / CACHE_TAR_INVENTORY_FILENAME
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    data_path.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if mode in {"metadata", "cached"}:
            existing = _read_inventory_clips(cache_path)
            if existing is not None:
                return existing, False
        signature = _tar_shard_signature(data_dir)
        if cache_path.is_file():
            try:
                with cache_path.open("r", encoding="utf-8") as file:
                    cached = json.load(file)
            except (OSError, ValueError, TypeError):
                cached = None
            if _valid_inventory_payload(cached, signature):
                return cached["clips"], False

        inventory = scan_clip_tar_inventory(data_dir)
        final_signature = _tar_shard_signature(data_dir)
        if final_signature != signature:
            raise RuntimeError(
                f"Cache tar shards changed while inventory was being scanned: {data_dir}"
            )
        payload = {
            "tar_signature": final_signature,
            "clips": inventory,
        }
        temporary = cache_path.with_suffix(
            cache_path.suffix + f".tmp.{os.getpid()}"
        )
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        os.replace(temporary, cache_path)
        return inventory, True


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    x, y, z, w = [quat[..., i] for i in range(4)]
    norm = np.maximum(x * x + y * y + z * z + w * w, 1e-12)
    scale = 2.0 / norm

    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale

    matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1.0 - (yy + zz)
    matrix[..., 0, 1] = xy - wz
    matrix[..., 0, 2] = xz + wy
    matrix[..., 1, 0] = xy + wz
    matrix[..., 1, 1] = 1.0 - (xx + zz)
    matrix[..., 1, 2] = yz - wx
    matrix[..., 2, 0] = xz - wy
    matrix[..., 2, 1] = yz + wx
    matrix[..., 2, 2] = 1.0 - (xx + yy)
    return matrix


def _matrix_to_rotation_6d_columns(matrix: np.ndarray) -> np.ndarray:
    """Encode rotation matrices with the camera cache's 6D convention.

Cache ``rot6d.npy`` stores ``concat(R[:, 0], R[:, 1])``: the first two
    *columns* of each local rotation matrix, with each complete column kept
    contiguous.  This is deliberately not ``R[:2, :].reshape(6)``, which
    stores the first two rows and decodes to a different rotation under
    :func:`utils.rotation.rot6d_to_rotmat_tensor`.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices ending in [3, 3], got {matrix.shape}")
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1).astype(
        np.float32,
        copy=False,
    )


class CacheVideoSegmentDataset(Dataset):
    """
    Dataset for the camera-cache layout.

    Each item is sampled from one motion entry in meta.json, so windows never
    cross video/animation segment boundaries. The reference pose is always the
    character's reset pose from static.npy; animated GT is target-only data.
    """

    def __init__(
        self,
        skeleton_id: str,
        data_dir: str,
        window: int,
        split_mode: str,
        image_embed_tokens: int = 257,
        image_embed_dim: int = 1024,
        image_embed_output_dtype: str = "float32",
        include_cls_token: bool = True,
        allow_missing_image_embed: bool = False,
        eval_ratio: float = 0.05,
        split_seed: int = 42,
        split_file: Optional[str] = None,
        eval_stride: Optional[int] = None,
        validate_feature_cache: bool = True,
        full_sequence: bool = False,
        static_pos_joints: Optional[Sequence[int]] = None,
        static_rot_joints: Optional[Sequence[int]] = None,
        mmap: bool = True,
        max_joints: Optional[int] = None,
        image_size: int = 256,
        full_image_size: Optional[int] = None,
        bbox_margin: int = 5,
        complete_motion_eval: bool = False,
        first_window_eval: bool = False,
        motion_source_splits: Optional[Sequence[str]] = None,
        inventory_mode: str = "scan",
        verbose: bool = False,
    ):
        if split_mode not in ("train", "test"):
            raise ValueError(f"split_mode must be 'train' or 'test', got {split_mode}")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Cache data_dir not found: {data_dir}")

        self.skeleton_id = skeleton_id
        self.data_dir = data_dir
        # Precomputed DINO feature directory; the online RGB path does not
        # read it.
        self.window = require_positive_int(window, "window")
        self.split_mode = split_mode
        self.inventory_mode = str(inventory_mode).strip().lower()
        self.verbose = require_bool(verbose, "verbose")
        # Obj rigs are partitioned at rig level, but one rig hosts motions
        # from several source splits (``motions[].source_split``).  Keeping
        # every motion of a selected rig would fold the training motions of
        # a seen skeleton into its validation split.
        self.motion_source_splits = (
            None
            if motion_source_splits is None
            else tuple(str(value) for value in motion_source_splits)
        )
        self.image_embed_tokens = require_positive_int(
            image_embed_tokens, "image_embed_tokens"
        )
        self.image_embed_dim = require_positive_int(
            image_embed_dim, "image_embed_dim"
        )
        self.image_embed_output_dtype = np.dtype(image_embed_output_dtype)
        self.include_cls_token = require_bool(
            include_cls_token, "include_cls_token"
        )
        self.allow_missing_image_embed = require_bool(
            allow_missing_image_embed, "allow_missing_image_embed"
        )
        # Only None selects the default.  An explicit zero is invalid rather
        # than a request that should be silently rewritten to ``window``.
        self.eval_stride = require_positive_int(
            self.window if eval_stride is None else eval_stride,
            "eval_stride",
        )
        if self.eval_stride <= 0 or self.eval_stride > self.window:
            raise ValueError(f"eval_stride must be in [1, window], got {self.eval_stride}")
        self.validate_feature_cache = require_bool(
            validate_feature_cache, "validate_feature_cache"
        )
        # Offline inference can request deterministic full coverage for either
        # split. Training keeps one random window per train clip.
        self.full_sequence = require_bool(full_sequence, "full_sequence")
        self.complete_motion_eval = require_bool(
            complete_motion_eval, "complete_motion_eval"
        )
        self.first_window_eval = require_bool(
            first_window_eval, "first_window_eval"
        )
        if self.complete_motion_eval and self.first_window_eval:
            raise ValueError(
                "complete_motion_eval and first_window_eval are mutually exclusive"
            )
        self.image_size = require_positive_int(image_size, "image_size")
        self.full_image_size = (
            require_positive_int(full_image_size, "full_image_size")
            if full_image_size is not None
            else None
        )
        self.bbox_margin = require_int(bbox_margin, "bbox_margin")
        if self.bbox_margin < 0:
            raise ValueError(f"bbox_margin must be non-negative, got {self.bbox_margin}")
        self.image_transform = build_transform(self.image_size)
        self._tar_files: Dict[str, tarfile.TarFile] = {}

        with open(os.path.join(data_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)
        self.static = np.load(os.path.join(data_dir, "static.npy"), allow_pickle=True).item()
        self.duplicate_clip_names = {
            clip_name
            for clip_name, count in Counter(
                str(motion["clip_name"])
                for motion in self.meta.get("motions", [])
            ).items()
            if count > 1
        }

        self.num_joints = int(self.meta.get("joints", self.static["joints"]))
        self.joint_names = [str(x) for x in self.meta.get("joint_names", self.static["joint_names"])]
        static_joint_names = [str(x) for x in self.static["joint_names"]]
        self.parents = np.asarray(self.static["parents"], dtype=np.int64)[: self.num_joints]
        if len(self.joint_names) != self.num_joints or self.parents.shape[0] != self.num_joints:
            raise ValueError(f"{skeleton_id} invalid joint count in meta/static")
        if max_joints is not None and self.num_joints > require_positive_int(max_joints, "max_joints"):
            raise ValueError(
                f"{skeleton_id} has {self.num_joints} joints, maximum is {max_joints}"
            )
        if static_joint_names != self.joint_names:
            raise ValueError(f"{skeleton_id} joint_names mismatch between meta.json and static.npy")
        if self.parents[0] != -1 or any(not (0 <= int(p) < j) for j, p in enumerate(self.parents[1:], start=1)):
            raise ValueError(
                f"{skeleton_id} parents must be root-0, parent-before-child topological order: "
                f"parents[:10]={self.parents[:10].tolist()}"
            )

        self.graph_hop = self._build_hop_matrix()
        self.graph_edge = self._build_edge_matrix()
        self.rot6d = self._load_motion_array("rot6d", mmap=mmap)
        self.position = self._load_motion_array("world_pos", mmap=mmap)
        self.camera_root_position = self._load_motion_array(
            "canonical_root_pos", mmap=mmap
        )
        if self.rot6d.ndim != 3 or self.rot6d.shape[-1] != 6:
            raise ValueError(f"{skeleton_id} rot6d must have shape [T,J,6], got {self.rot6d.shape}")
        if self.rot6d.shape[1] != self.num_joints:
            raise ValueError(f"{skeleton_id} joint count mismatch: {self.rot6d.shape[1]} vs {self.num_joints}")
        if self.position.ndim != 3 or self.position.shape[-1] != 3:
            raise ValueError(
                f"{skeleton_id} world_pos must have shape [T,J,3], "
                f"got {self.position.shape}"
            )
        if self.position.shape[1] != self.num_joints:
            raise ValueError(
                f"{skeleton_id} world_pos joint count mismatch: "
                f"{self.position.shape[1]} vs {self.num_joints}"
            )

        self.total_frames = int(self.meta["total_frames"])
        if self.rot6d.shape[0] != self.total_frames:
            raise ValueError(
                f"{skeleton_id} total_frames mismatch: meta={self.total_frames}, "
                f"rot6d={self.rot6d.shape[0]}"
            )
        if self.position.shape[0] != self.total_frames:
            raise ValueError(
                f"{skeleton_id} total_frames mismatch: meta={self.total_frames}, "
                f"world_pos={self.position.shape[0]}"
            )
        if self.camera_root_position.shape != (self.total_frames, 3):
            raise ValueError(
                f"{skeleton_id} canonical_root_pos must have shape "
                f"[{self.total_frames},3], got {self.camera_root_position.shape}"
            )

        self.metric_scale = float(np.asarray(self.static["metric_scale"], dtype=np.float32))
        if not np.isfinite(self.metric_scale) or self.metric_scale <= 0:
            raise ValueError(f"{skeleton_id} invalid metric_scale: {self.metric_scale}")
        self.reset_position = self._build_rest_global_positions()
        self.reset_position = self.reset_position - self.reset_position[0:1]
        reset_rot_mats = _quat_xyzw_to_matrix(
            np.asarray(self.static["rest_rotations_quat"], dtype=np.float32)
        )
        # Keep reset/reference rotations in exactly the same column-major 6D
        # convention as dynamic rot6d.npy.  Taking reset_rot_mats[..., :2, :]
        # here would encode rows and silently corrupt the Decoder memory.
        self.reset_rot6d = _matrix_to_rotation_6d_columns(reset_rot_mats).copy()
        self.rest_translations = self._load_rest_local_translations()
        self.static_pos_joint_mask = self._build_static_mask(static_pos_joints)
        self.static_rot_joint_mask = self._build_static_mask(static_rot_joints)
        # Root translation is not predicted, but joint zero is the supervised
        # camera/global root rotation and must never be frozen to reset pose.
        self.static_rot_joint_mask[0] = False
        # Always use an inventory of the real tar members. Metadata tar paths
        # alone cannot prove that a clip's RGB/mask members actually exist.
        # The persistent cache makes this a cheap signature check after the
        # first complete scan.
        self.tar_inventory, _ = load_or_build_clip_tar_inventory(
            self.data_dir, inventory_mode=self.inventory_mode
        )
        self.discovered_tar_paths = {
            clip: str(info["tar"]) for clip, info in self.tar_inventory.items()
        }
        self.unavailable_rgb_clips: set[str] = set()
        self.truncated_rgb_clips: Dict[str, Dict[str, int]] = {}
        for motion in self.meta.get("motions", []):
            clip_name = str(motion["clip_name"])
            motion_name = self._motion_name(motion)
            meta_frames = max(int(motion.get("frames", 0)), 0)
            if meta_frames == 0:
                continue
            inventory_entry = self._motion_inventory_entry(motion)
            available_frames = int(inventory_entry.get("frames", 0))
            if available_frames == 0:
                self.unavailable_rgb_clips.add(motion_name)
            elif available_frames < meta_frames:
                self.truncated_rgb_clips[motion_name] = {
                    "meta_frames": meta_frames,
                    "usable_frames": available_frames,
                    "dropped_frames": meta_frames - available_frames,
                }
        if self.unavailable_rgb_clips and self.verbose:
            print(
                f"  Excluding {len(self.unavailable_rgb_clips)} {self.skeleton_id} "
                "motions with no contiguous paired RGB/mask frames in any shard"
            )
        if self.truncated_rgb_clips and self.verbose:
            dropped_frames = sum(
                info["dropped_frames"] for info in self.truncated_rgb_clips.values()
            )
            print(
                f"  Truncating {len(self.truncated_rgb_clips)} {self.skeleton_id} "
                f"motions to their real paired RGB/mask prefix ({dropped_frames} "
                "metadata-only frames ignored)"
            )

        self.items = self._build_items(
            eval_ratio=float(eval_ratio),
            split_seed=require_int(split_seed, "split_seed"),
            split_file=split_file,
        )
        if self.verbose:
            print(
                f"Loaded cache dataset {skeleton_id}: {self.total_frames} frames, "
                f"{len(self.items)} {split_mode} segments, {self.num_joints} joints"
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tar_files"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tar_files = {}

    def __del__(self):
        for archive in getattr(self, "_tar_files", {}).values():
            archive.close()

    def _load_motion_array(self, name: str, mmap: bool) -> np.ndarray:
        rel_path = self.meta.get("array_files", {}).get(name, f"{name}.npy")
        path = os.path.join(self.data_dir, rel_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{self.skeleton_id} array not found for {name}: {path}")
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)

    def _motion_name(self, motion: Dict[str, Any]) -> str:
        """Return a split/output-safe name for this motion."""
        clip_name = str(motion["clip_name"])
        duplicate_names = getattr(self, "duplicate_clip_names", set())
        if clip_name not in duplicate_names:
            return clip_name
        camera_folder = str(motion.get("camera_folder", "")).replace("\\", "/")
        camera_name = Path(camera_folder).name
        if not camera_name:
            camera_name = f"motion{int(motion.get('motion_index', 0)):06d}"
        return f"{clip_name}__{camera_name}"

    def _motion_inventory_entry(self, motion: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the one physical RGB/mask shard for a motion view.

        All cache producers use the same ``images/*.tar`` layout.  A metadata
        tar path is preferred; when it is omitted, camera_folder disambiguates
        duplicate clip names before falling back to a unique clip entry.
        """
        clip_name = str(motion["clip_name"])
        relative_tar = str(motion.get("image", {}).get("tar", ""))
        qualified_key = f"{relative_tar}::{clip_name}"
        if relative_tar and qualified_key in self.tar_inventory:
            return self.tar_inventory[qualified_key]
        camera_folder = str(motion.get("camera_folder", "")).replace("\\", "/")
        camera_name = Path(camera_folder).name
        if camera_name:
            matches = [
                entry for entry in self.tar_inventory.values()
                if entry.get("clip_name", clip_name) == clip_name
                and Path(str(entry.get("tar", ""))).stem.lower()
                == f"shard-{camera_name.lower()}"
            ]
            if len(matches) == 1:
                return matches[0]
        direct = self.tar_inventory.get(clip_name)
        if direct:
            return direct
        matches = [
            entry for entry in self.tar_inventory.values()
            if entry.get("clip_name", clip_name) == clip_name
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"{self.skeleton_id} motion {clip_name!r} is ambiguous across "
                f"camera tars: {[entry.get('tar') for entry in matches]}"
            )
        return {}

    def _load_rest_local_translations(self) -> np.ndarray:
        return np.asarray(self.static["rest_translations"], dtype=np.float32).copy()

    def _build_rest_global_positions(self) -> np.ndarray:
        local_t = self._load_rest_local_translations()
        local_r = _quat_xyzw_to_matrix(np.asarray(self.static["rest_rotations_quat"], dtype=np.float32))
        global_pos = np.zeros_like(local_t)
        global_rot = np.zeros((self.num_joints, 3, 3), dtype=np.float32)
        for joint_idx, parent in enumerate(self.parents):
            if 0 <= parent < self.num_joints:
                parent_idx = int(parent)
                global_pos[joint_idx] = global_pos[parent_idx] + global_rot[parent_idx] @ local_t[joint_idx]
                global_rot[joint_idx] = global_rot[parent_idx] @ local_r[joint_idx]
            else:
                global_pos[joint_idx] = local_t[joint_idx]
                global_rot[joint_idx] = local_r[joint_idx]
        return global_pos

    def _build_static_mask(self, joint_ids: Optional[Sequence[int]]) -> np.ndarray:
        mask = np.zeros((self.num_joints,), dtype=np.bool_)
        if joint_ids is None:
            return mask
        if isinstance(joint_ids, str) and joint_ids == "invalid":
            valid = np.asarray(self.static.get("valid_joint_mask", np.ones(self.num_joints, dtype=np.bool_)))
            return (~valid.astype(np.bool_))[: self.num_joints]
        ids = [int(i) for i in joint_ids if 0 <= int(i) < self.num_joints]
        mask[ids] = True
        return mask

    def _build_hop_matrix(self) -> np.ndarray:
        n = self.num_joints
        adj = [[] for _ in range(n)]
        for i, p in enumerate(self.parents):
            if 0 <= p < n:
                adj[i].append(int(p))
                adj[int(p)].append(i)

        hop = np.full((n, n), 5, dtype=np.int64)
        for src in range(n):
            hop[src, src] = 0
            queue = [src]
            while queue:
                cur = queue.pop(0)
                if hop[src, cur] >= 5:
                    continue
                for nxt in adj[cur]:
                    if hop[src, nxt] > hop[src, cur] + 1:
                        hop[src, nxt] = hop[src, cur] + 1
                        queue.append(nxt)
        return hop

    def _build_edge_matrix(self) -> np.ndarray:
        n = self.num_joints
        edge = np.full((n, n), 4, dtype=np.int64)
        np.fill_diagonal(edge, 0)
        for i, p in enumerate(self.parents):
            if 0 <= p < n:
                edge[p, i] = 1
                edge[i, p] = 2
        return edge

    def _open_tar(self, path: str) -> tarfile.TarFile:
        archive = self._tar_files.get(path)
        if archive is None:
            archive = tarfile.open(path, "r")
            self._tar_files[path] = archive
        return archive

    def _read_tar_image(self, archive: tarfile.TarFile, member_name: str) -> Image.Image:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise FileNotFoundError(
                f"{self.skeleton_id} tar member not found: {archive.name}:{member_name}"
            ) from exc
        file_obj = archive.extractfile(member)
        if file_obj is None:
            raise FileNotFoundError(
                f"{self.skeleton_id} cannot extract tar member: {archive.name}:{member_name}"
            )
        return Image.open(io.BytesIO(file_obj.read())).copy()

    def _load_cropped_frame(
        self,
        tar_path: str,
        clip_name: str,
        local_frame: int,
        rgb_ext: str,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        archive = self._open_tar(tar_path)
        stem = f"{clip_name}.{local_frame:06d}"
        rgb = self._read_tar_image(archive, stem + rgb_ext)
        mask = self._read_tar_image(archive, stem + ".png")
        rgb = rgb.convert("RGB")
        crop_box = crop_box_from_mask(rgb, mask, self.bbox_margin)
        cropped = rgb.crop(crop_box)
        image = self.image_transform(cropped).numpy().astype(
            np.float32, copy=False
        )
        full_image = None
        if self.full_image_size is not None:
            # Keep complete frames as uint8. The selected frozen backbone
            # performs its own normalization, and uint8 avoids duplicating a
            # long sequence as both crop/full-frame float32 tensors.
            resized = rgb.resize(
                (self.full_image_size, self.full_image_size),
                resample=Image.Resampling.BICUBIC,
            )
            full_image = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
        return image, np.asarray(crop_box, dtype=np.float32), full_image

    def _build_items(
        self,
        eval_ratio: float,
        split_seed: int,
        split_file: Optional[str],
    ) -> List[Dict[str, Any]]:
        motions = []
        for motion in self.meta.get("motions", []):
            offset = int(motion["offset"])
            frames = min(int(motion["frames"]), self.total_frames - offset)
            clip_name = str(motion["clip_name"])
            motion_name = self._motion_name(motion)
            inventory_entry = self._motion_inventory_entry(motion)
            frames = min(
                frames,
                int(inventory_entry.get("frames", 0)),
            )
            # Keep every non-empty motion. Short clips are padded only at
            # sample construction time and carry a frame_valid_mask so padded
            # timesteps never contribute to loss, metrics, or saved output.
            if frames > 0 and motion_name not in self.unavailable_rgb_clips:
                motions.append({**motion, "offset": offset, "frames": frames})

        if self.motion_source_splits is not None:
            wanted_splits = set(self.motion_source_splits)
            motions = [
                motion
                for motion in motions
                if str(motion.get("source_split")) in wanted_splits
            ]
            if not motions:
                raise ValueError(
                    f"{self.skeleton_id} has no motions in source splits "
                    f"{sorted(wanted_splits)}"
                )

        train_ids, eval_ids = self._resolve_split_ids(motions, eval_ratio, split_seed, split_file)

        items = []
        for motion_id, motion in enumerate(motions):
            selected_ids = train_ids if self.split_mode == "train" else eval_ids
            if motion_id not in selected_ids:
                continue
            window_specs = [(None, None)]
            motion_frames = int(motion["frames"])
            # First-window evaluation is deterministic for either split. In
            # particular, evaluating --split train must not fall back to the
            # random-window behavior used by optimization.
            if self.first_window_eval:
                window_specs = [(0, min(motion_frames, self.window))]
            elif self.split_mode == "test" or self.full_sequence:
                if self.complete_motion_eval:
                    window_specs = [(0, motion_frames)]
                elif motion_frames <= self.window:
                    window_specs = [(0, motion_frames)]
                else:
                    last_start = motion_frames - self.window
                    starts = list(range(0, last_start + 1, self.eval_stride))
                    if starts[-1] != last_start:
                        starts.append(last_start)
                    window_specs = [(start, self.window) for start in starts]
            for window_index, (local_start, valid_length) in enumerate(window_specs):
                image_info = dict(motion.get("image", {}))
                clip_name = str(motion["clip_name"])
                motion_name = self._motion_name(motion)
                inventory_entry = self._motion_inventory_entry(motion)
                if inventory_entry:
                    # Prefer verified archive members over stale metadata even
                    # when meta.json happened to contain a tar path.
                    image_info["tar"] = inventory_entry["tar"]
                items.append({
                    "seq": motion_name,
                    "member_clip_name": clip_name,
                    "motion_index": int(motion.get("motion_index", motion_id)),
                    "offset": int(motion["offset"]),
                    "frames": int(motion["frames"]),
                    "image": image_info,
                    "image_ext": (
                        inventory_entry["rgb_ext"]
                        if inventory_entry
                        else (
                            Path(motion.get("images", {}).get("image_glob", "*.jpg")).suffix.lower()
                            if Path(motion.get("images", {}).get("image_glob", "*.jpg")).suffix.lower()
                            in (".jpg", ".jpeg")
                            else ".jpg"
                        )
                    ),
                    "local_start": local_start,
                    "valid_length": valid_length,
                    "window_index": window_index,
                    # The motion cache is already expressed in this OpenCV
                    # camera frame. Keep the original camera pose metadata for
                    # provenance, but visualization must not apply it twice.
                    "camera": dict(motion.get("camera", {})),
                })
        return items

    def _resolve_split_ids(
        self,
        motions: List[Dict[str, Any]],
        eval_ratio: float,
        split_seed: int,
        split_file: Optional[str],
    ) -> tuple[set[int], set[int]]:
        motion_names = [self._motion_name(motion) for motion in motions]
        animation_names = [str(motion["clip_name"]) for motion in motions]
        if len(set(motion_names)) != len(motion_names):
            raise ValueError(
                f"{self.skeleton_id} motion identities remain ambiguous after "
                "camera qualification"
            )
        if split_file is not None:
            with open(split_file, "r", encoding="utf-8") as f:
                split = json.load(f)
            if self.skeleton_id not in split:
                raise KeyError(f"{split_file} has no split for {self.skeleton_id}")
            train_list = [
                str(x)
                for x in split[self.skeleton_id].get("train", [])
            ]
            val_list = [
                str(x)
                for x in split[self.skeleton_id].get("val", [])
            ]
            train_animations = set(train_list)
            val_animations = set(val_list)
            if len(train_animations) != len(train_list) or len(
                val_animations
            ) != len(val_list):
                raise ValueError(
                    f"{split_file} contains duplicate animation names "
                    f"for {self.skeleton_id}"
                )
            if train_animations & val_animations:
                raise ValueError(
                    f"{split_file} train/val animation overlap for "
                    f"{self.skeleton_id}"
                )
            all_metadata_animations = {
                str(motion["clip_name"])
                for motion in self.meta.get("motions", motions)
            }
            unexpected = (
                train_animations | val_animations
            ) - all_metadata_animations
            if unexpected:
                raise ValueError(
                    f"{split_file} contains unknown animations for "
                    f"{self.skeleton_id}: {sorted(unexpected)[:5]}"
                )
            usable_animations = set(animation_names)
            train_animations &= usable_animations
            val_animations &= usable_animations
            missing = usable_animations - train_animations - val_animations
            if missing:
                raise ValueError(
                    f"{split_file} is stale for {self.skeleton_id}: "
                    f"missing animations={sorted(missing)[:5]}"
                )
            train_ids = {
                index
                for index, animation in enumerate(animation_names)
                if animation in train_animations
            }
            val_ids = {
                index
                for index, animation in enumerate(animation_names)
                if animation in val_animations
            }
            return train_ids, val_ids

        # Split unique underlying animations, never individual camera views.
        # Any future camera3/camera4 motion with an existing clip_name then
        # inherits the persisted assignment automatically.
        unique_animations = list(dict.fromkeys(animation_names))
        rng = np.random.default_rng(split_seed)
        order = np.arange(len(unique_animations))
        rng.shuffle(order)
        eval_count = (
            int(round(len(unique_animations) * eval_ratio))
            if eval_ratio > 0
            else 0
        )
        eval_count = min(
            max(
                eval_count,
                1
                if eval_ratio > 0 and len(unique_animations) > 1
                else 0,
            ),
            len(unique_animations),
        )
        eval_animations = {
            unique_animations[index] for index in order[:eval_count]
        }
        eval_ids = {
            index
            for index, animation in enumerate(animation_names)
            if animation in eval_animations
        }

        return set(range(len(motions))) - eval_ids, eval_ids

    def __len__(self) -> int:
        return len(self.items)

    def io_locality_key(self, idx: int) -> tuple[str, str]:
        """Return the storage shard used by an item for worker-local sampling."""
        item = self.items[idx]
        relative_tar = item.get("image", {}).get("tar")
        if not relative_tar:
            raise ValueError(
                f"{self.skeleton_id}/{item.get('seq')} has no RGB tar locality key"
            )
        return self.skeleton_id, str(relative_tar)

    def __getitem__(self, idx: int | tuple[int, int]) -> Dict[str, Any]:
        forced_local_start = None
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise ValueError("Forced cache index must be (index, local_start)")
            idx, forced_local_start = int(idx[0]), int(idx[1])
        item = self.items[idx]
        if forced_local_start is not None:
            max_start = max(int(item["frames"]) - self.window, 0)
            if not 0 <= forced_local_start <= max_start:
                raise ValueError(
                    f"forced local_start {forced_local_start} outside [0,{max_start}]"
                )
            local_start = forced_local_start
        elif (
            self.split_mode == "train"
            and not self.full_sequence
            and not self.first_window_eval
        ):
            max_start = max(int(item["frames"]) - self.window, 0)
            local_start = np.random.randint(0, max_start + 1)
        else:
            local_start = int(item["local_start"])

        start = item["offset"] + local_start
        valid_length = item.get("valid_length")
        if valid_length is None:
            valid_length = min(self.window, int(item["frames"]) - local_start)
        valid_length = int(valid_length)
        maximum_valid_length = (
            int(item["frames"]) if self.complete_motion_eval else self.window
        )
        if not 1 <= valid_length <= maximum_valid_length:
            raise ValueError(
                f"Invalid window valid_length={valid_length} for {item['seq']}"
            )
        valid_frame_idx = np.arange(start, start + valid_length, dtype=np.int64)
        # Fixed-size train/validation windows use explicit zero padding. The
        # invalid slots have no source-frame identity and are excluded from
        # temporal attention and every supervised objective by this mask.
        output_window = valid_length if self.complete_motion_eval else self.window
        pad_length = output_window - valid_length
        frame_idx = np.concatenate(
            (valid_frame_idx, np.full((pad_length,), -1, dtype=np.int64))
        )
        animation_frame_indices = np.concatenate(
            (
                np.arange(
                    local_start, local_start + valid_length, dtype=np.int64
                ),
                np.full((pad_length,), -1, dtype=np.int64),
            )
        )
        frame_valid_mask = np.arange(output_window) < valid_length

        image_info = item.get("image", {})
        rgb_tar = image_info.get("tar")
        rgb_tar_path = os.path.join(self.data_dir, rgb_tar) if rgb_tar else None
        ref_position = self.reset_position.copy()
        valid_rot6d = np.asarray(self.rot6d[valid_frame_idx], dtype=np.float32)
        valid_position = np.asarray(
            self.position[valid_frame_idx], dtype=np.float32
        )
        valid_position = valid_position - valid_position[:, 0:1]
        valid_camera_root = np.asarray(
            self.camera_root_position[valid_frame_idx], dtype=np.float32
        )
        rot6d = np.concatenate(
            (
                valid_rot6d,
                np.zeros(
                    (pad_length, *valid_rot6d.shape[1:]),
                    dtype=np.float32,
                ),
            ),
            axis=0,
        )
        camera_root_position = np.concatenate(
            (
                valid_camera_root,
                np.zeros((pad_length, 3), dtype=np.float32),
            ),
            axis=0,
        )
        position = np.concatenate(
            (
                valid_position,
                np.zeros(
                    (pad_length, *valid_position.shape[1:]),
                    dtype=np.float32,
                ),
            ),
            axis=0,
        )
        ref_rot6d = self.reset_rot6d.copy()

        camera = item["camera"]
        required_camera_fields = (
            "fx", "fy", "cx", "cy", "width", "height",
            "location", "right", "up", "forward",
        )
        missing_camera_fields = [
            field for field in required_camera_fields if field not in camera
        ]
        if missing_camera_fields:
            raise ValueError(
                f"{self.skeleton_id}/{item['seq']} camera metadata missing "
                f"{missing_camera_fields}"
            )
        camera_intrinsics = np.asarray(
            [camera["fx"], camera["fy"], camera["cx"], camera["cy"]],
            dtype=np.float32,
        )
        camera_image_size = np.asarray(
            [camera["width"], camera["height"]], dtype=np.int64
        )
        camera_extrinsics = {
            field: np.asarray(camera[field], dtype=np.float32)
            for field in ("location", "right", "up", "forward")
        }

        if not rgb_tar_path or not os.path.isfile(rgb_tar_path):
            raise FileNotFoundError(f"{self.skeleton_id} RGB tar not found: {rgb_tar_path}")
        local_frames = np.arange(local_start, local_start + valid_length, dtype=np.int64)
        valid_frame_data = [
            self._load_cropped_frame(
                rgb_tar_path,
                item["member_clip_name"],
                int(local_frame),
                item["image_ext"],
            )
            for local_frame in local_frames
        ]
        valid_images = [frame[0] for frame in valid_frame_data]
        valid_crop_boxes = np.stack(
            [frame[1] for frame in valid_frame_data], axis=0
        )
        crop_width_height = (
            valid_crop_boxes[:, 2:4] - valid_crop_boxes[:, 0:2]
        )
        if np.any(crop_width_height <= 0):
            raise ValueError(
                f"{self.skeleton_id}/{item['seq']} has a non-positive crop box"
            )
        crop_center = 0.5 * (
            valid_crop_boxes[:, :2] + valid_crop_boxes[:, 2:4]
        )
        crop_center_ray = (
            crop_center - camera_intrinsics[None, 2:4]
        ) / camera_intrinsics[None, :2]
        valid_root_crop_camera_xyz = valid_camera_root.copy()
        valid_root_crop_camera_xyz[:, :2] -= (
            crop_center_ray * valid_camera_root[:, 2:3]
        )
        images = valid_images + [np.zeros_like(valid_images[0])] * pad_length
        image_rgb = np.stack(images, axis=0)
        full_image_rgb = None
        if self.full_image_size is not None:
            valid_full_images = [frame[2] for frame in valid_frame_data]
            if any(image is None for image in valid_full_images):
                raise RuntimeError("full_image_size is set but a full frame is missing")
            full_images = valid_full_images + [
                np.zeros_like(valid_full_images[0])
            ] * pad_length
            full_image_rgb = np.stack(full_images, axis=0)
        crop_box = np.concatenate(
            (
                valid_crop_boxes,
                np.zeros((pad_length, 4), dtype=np.float32),
            ),
            axis=0,
        )
        root_crop_camera_xyz = np.concatenate(
            (
                valid_root_crop_camera_xyz,
                np.zeros((pad_length, 3), dtype=np.float32),
            ),
            axis=0,
        )
        # ``seq`` can include a camera suffix to disambiguate two metadata
        # motions with the same clip name. Tar members retain the raw clip
        # name, which is recorded separately as ``member_clip_name``.
        rgb_member_names = cache_rgb_member_names(
            item["member_clip_name"],
            local_frames,
            item["image_ext"],
            pad_length,
        )

        result = {
            "rel": f"{self.skeleton_id}/{item['seq']}",
            "species": self.skeleton_id,
            "F": item["frames"],
            "J": self.num_joints,
            "W": output_window,
            "frame_indices": frame_idx,
            "animation_frame_indices": animation_frame_indices,
            "frame_valid_mask": frame_valid_mask,
            "valid_length": valid_length,
            "motion_offset": item["offset"],
            "motion_frames": item["frames"],
            "window_start": local_start,
            "window_index": item["window_index"],
            "rgb_tar_path": rgb_tar_path,
            "rgb_member_names": rgb_member_names,
            # Authoritative names validated against static.npy/meta.json.
            "joint_names": list(self.joint_names),
            "global_scale": np.float32(1.0),
            "metric_scale": np.float32(self.metric_scale),
            # Known input preprocessing geometry. x/y prediction is defined
            # in this cropped image and deterministically mapped back through
            # this box; the box itself is never predicted.
            "crop_box": crop_box,
            # Supervision only: metric XYZ in the crop-centered camera frame.
            "root_crop_camera_xyz": root_crop_camera_xyz,
            # Supervision only: absolute OpenCV camera XYZ in meters.
            "camera_root_position": camera_root_position,
            "camera_intrinsics": camera_intrinsics,
            "camera_image_size": camera_image_size,
            "camera_extrinsics": camera_extrinsics,
            "ref_position": ref_position,
            "graph_hop": self.graph_hop,
            "graph_edge": self.graph_edge,
            "static_rot_joint_mask": self.static_rot_joint_mask,
            "static_pos_joint_mask": self.static_pos_joint_mask,
            # Supervision only: real GT world-space joints translated to a
            # per-frame root origin. model_inputs_only strips this target.
            "position": position,
            "rot6d_a": rot6d,
            "ref_rot6d_a": ref_rot6d,
            "parent_a": self.parents[None, :],
            # Non-root values are fixed parent-local bone offsets.  The root
            # entry is not used after FK positions are root-centered.
            "offset_a": self.rest_translations[None, ...].astype(np.float32),
            # Normalized mask-bbox crops. Frozen DINOv3 runs inside the model,
            # so no precomputed visual features are read from disk.
            "image_rgb": image_rgb,
        }
        if full_image_rgb is not None:
            # Complete source-camera frame for root-XYZ backbones. Rotation
            # and relative-pose branches continue to consume the mask crop.
            result["full_image_rgb"] = full_image_rgb
        return result


def _is_skeleton_partition_file(path: str) -> bool:
    """True when the split file partitions skeletons rather than animation clips."""
    return Path(os.fspath(path)).name.lower() in {"obj.json", "mobjaverse.json"}


def _iter_cache_configs(
    data_cfg: Dict[str, Any],
    source_splits: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Resolve skeleton entries for the single unified cache reader.

    The Zoo split lists skeletons as top-level keys; the Obj split lists them
    by source partition.
    """
    dataset_root = data_cfg["dataset_root"]
    split_file = data_cfg.get("split_file")
    if not split_file:
        raise ValueError("Cache dataset config requires split_file")
    with open(os.fspath(split_file), "r", encoding="utf-8") as file:
        payload = json.load(file)
    if _is_skeleton_partition_file(split_file):
        if isinstance(payload, dict):
            if source_splits is None:
                values = payload.get("all")
                if values is None:
                    values = [
                        item
                        for split_values in payload.values()
                        if isinstance(split_values, list)
                        for item in split_values
                    ]
            else:
                values = [
                    item for split in source_splits
                    for item in payload.get(str(split), [])
                ]
            entries = list(dict.fromkeys(values))
        else:
            entries = payload
    else:
        if not isinstance(payload, dict):
            raise ValueError("Animation split file must contain skeleton entries")
        entries = sorted(name for name in payload if not name.startswith("_"))
    if isinstance(entries, str) or not isinstance(entries, Sequence):
        raise ValueError("Skeleton split file must contain a list")
    if not entries:
        raise ValueError("Cache dataset config selected no skeleton IDs")

    configs = []
    for entry in entries:
        cfg = {"skeleton_id": entry} if isinstance(entry, str) else dict(entry)
        skeleton_id = cfg.get("skeleton_id")
        if not skeleton_id:
            raise KeyError(f"Cache entry is missing skeleton_id: {entry!r}")
        cfg["skeleton_id"] = str(skeleton_id)
        cfg.setdefault("data_dir", os.path.join(dataset_root, cfg["skeleton_id"]))
        configs.append(cfg)
    return configs


def prepare_cache_tar_inventories(
    data_cfg: Dict[str, Any],
    source_splits: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, int | bool]]:
    """Build or validate every configured cache inventory before splitting."""
    if "dataset_root" not in data_cfg:
        raise KeyError("cache dataset config must define dataset_root")
    summary: Dict[str, Dict[str, int | bool]] = {}
    for skeleton_cfg in _iter_cache_configs(data_cfg, source_splits):
        inventory, rebuilt = load_or_build_clip_tar_inventory(
            skeleton_cfg["data_dir"],
            inventory_mode=data_cfg.get("inventory_mode", "scan"),
        )
        summary[str(skeleton_cfg["skeleton_id"])] = {
            "clips": len(inventory),
            "rebuilt": rebuilt,
        }
    return summary


def build_cache_dataset(
    data_cfg: Dict[str, Any],
    split_mode: str,
    window: int,
    full_sequence: bool = False,
    complete_motion_eval: bool = False,
    first_window_eval: bool = False,
    source_splits: Optional[Sequence[str]] = None,
    eval_ratio: Optional[float] = None,
) -> Dataset:
    if "dataset_root" not in data_cfg:
        raise KeyError("cache dataset config must define dataset_root")

    datasets = []
    for skeleton_cfg in _iter_cache_configs(data_cfg, source_splits):
        datasets.append(
            CacheVideoSegmentDataset(
                skeleton_id=skeleton_cfg["skeleton_id"],
                data_dir=skeleton_cfg["data_dir"],
                window=window,
                split_mode=split_mode,
                image_embed_tokens=data_cfg.get("image_embed_tokens", 257),
                image_embed_dim=data_cfg.get("image_embed_dim", 1024),
                image_embed_output_dtype=data_cfg.get("image_embed_output_dtype", "float32"),
                include_cls_token=data_cfg.get("include_cls_token", True),
                allow_missing_image_embed=data_cfg.get("allow_missing_image_embed", False),
                eval_ratio=(
                    data_cfg.get("eval_ratio", 0.05)
                    if eval_ratio is None
                    else eval_ratio
                ),
                split_seed=data_cfg.get("split_seed", 42),
                split_file=(
                    None
                    if (
                        data_cfg.get("split_file") is not None
                        and _is_skeleton_partition_file(data_cfg["split_file"])
                    )
                    else data_cfg.get("split_file")
                ),
                motion_source_splits=(
                    source_splits
                    if (
                        data_cfg.get("split_file") is not None
                        and _is_skeleton_partition_file(data_cfg["split_file"])
                    )
                    else None
                ),
                # First-window validation does not use eval_stride.
                eval_stride=(
                    window
                    if complete_motion_eval
                    else data_cfg.get("eval_stride", window)
                ),
                validate_feature_cache=data_cfg.get("validate_feature_cache", True),
                full_sequence=full_sequence,
                complete_motion_eval=complete_motion_eval,
                first_window_eval=first_window_eval,
                inventory_mode=data_cfg.get("inventory_mode", "scan"),
                verbose=data_cfg.get("verbose", False),
                image_size=data_cfg.get("image_size", 256),
                full_image_size=data_cfg.get("full_image_size"),
                bbox_margin=data_cfg.get("bbox_margin", 5),
                static_pos_joints=skeleton_cfg.get("static_pos_joints", data_cfg.get("static_pos_joints")),
                static_rot_joints=skeleton_cfg.get("static_rot_joints", data_cfg.get("static_rot_joints")),
                mmap=data_cfg.get("mmap", True),
                max_joints=data_cfg.get("max_joints"),
            )
        )

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


class CacheStepIterableDataset(IterableDataset):
    """Infinite deterministic shuffled stream for step-based DDP training.

    Every rank/worker consumes a disjoint strided shard of the same shuffled
    cycle. Once its shard is exhausted it advances directly to the next
    deterministic cycle, so the DataLoader iterator itself never ends or
    incurs an epoch-boundary refill.
    """

    def __init__(
        self,
        dataset: Dataset,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        if len(dataset) <= 0:
            raise ValueError("CacheStepIterableDataset requires non-empty data")
        self.dataset = dataset
        self.rank = require_int(rank, "iterable rank")
        self.world_size = require_positive_int(
            world_size, "iterable world_size"
        )
        self.seed = require_int(seed, "iterable seed")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"iterable rank must be in [0,{self.world_size}), got {self.rank}"
            )
        self.locality_groups = self._build_locality_groups(dataset)

    @staticmethod
    def _build_locality_groups(dataset: Dataset):
        """Collect item indices by tar while retaining a generic fallback."""
        groups: Dict[Any, List[int]] = {}
        if isinstance(dataset, ConcatDataset):
            offset = 0
            for dataset_index, child in enumerate(dataset.datasets):
                child_groups = CacheStepIterableDataset._build_locality_groups(
                    child
                )
                if child_groups is None:
                    return None
                for key, indices in child_groups.items():
                    groups[(dataset_index, key)] = [
                        offset + index for index in indices
                    ]
                offset += len(child)
            return groups
        locality_key = getattr(dataset, "io_locality_key", None)
        if locality_key is None:
            return None
        for index in range(len(dataset)):
            groups.setdefault(locality_key(index), []).append(index)
        return groups

    def _locality_order(self, cycle: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + cycle)
        group_keys = list(self.locality_groups)
        rng.shuffle(group_keys)
        ordered = []
        for key in group_keys:
            indices = np.asarray(self.locality_groups[key], dtype=np.int64)
            ordered.extend(rng.permutation(indices).tolist())
        return np.asarray(ordered, dtype=np.int64)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        consumer_count = self.world_size * worker_count
        consumer_id = self.rank * worker_count + worker_id
        if len(self.dataset) < consumer_count:
            raise RuntimeError(
                f"Cache iterable has {len(self.dataset)} samples but "
                f"{consumer_count} rank/workers; at least one stream is empty"
            )

        cycle = 0
        while True:
            if self.locality_groups is None:
                order = np.random.default_rng(self.seed + cycle).permutation(
                    len(self.dataset)
                )
                consumer_order = order[consumer_id::consumer_count]
            else:
                # Contiguous balanced shards preserve tar locality within each
                # worker. All samples are still shuffled without replacement
                # and partitioned disjointly across every rank/worker.
                order = self._locality_order(cycle)
                consumer_order = np.array_split(order, consumer_count)[
                    consumer_id
                ]
            for index in consumer_order:
                yield self.dataset[int(index)]
            cycle += 1

# ---------------------------------------------------------------------------
# Shared padded batch collation API.
# ---------------------------------------------------------------------------

import torch


# Collate
# ============================================================
def collate_full_motion_padded(batch, *, max_joints: int, dynamic_max_joints: bool = False):
    """Pad complete evaluation motions to the longest animation in the batch."""
    if not batch:
        raise ValueError("Cannot collate an empty full-motion batch")
    max_frames = max(int(item["W"]) for item in batch)
    padded = []
    for item in batch:
        item = dict(item)
        frames = int(item["W"])
        if not 1 <= frames <= max_frames:
            raise ValueError(f"Invalid full-motion length {frames}")
        pad_frames = max_frames - frames
        for key in ("position", "rot6d_a", "image_rgb", "image_embed"):
            if key not in item:
                continue
            value = np.asarray(item[key])
            if value.shape[0] != frames:
                raise ValueError(
                    f"{key} length {value.shape[0]} != motion length {frames}"
                )
            if pad_frames:
                value = np.concatenate(
                    (value, np.repeat(value[-1:], pad_frames, axis=0)),
                    axis=0,
                )
            item[key] = value

        frame_indices = np.asarray(item["frame_indices"])
        if pad_frames:
            frame_indices = np.concatenate(
                (frame_indices, np.repeat(frame_indices[-1:], pad_frames))
            )
        item["frame_indices"] = frame_indices
        animation_frame_indices = np.asarray(
            item.get("animation_frame_indices", item["frame_indices"])
        )
        if pad_frames:
            animation_frame_indices = np.concatenate(
                (
                    animation_frame_indices[:frames],
                    np.repeat(animation_frame_indices[frames - 1:frames], pad_frames),
                )
            )
        item["animation_frame_indices"] = animation_frame_indices
        item["frame_valid_mask"] = np.arange(max_frames) < frames
        member_names = list(item["rgb_member_names"])
        if pad_frames:
            member_names.extend([member_names[-1]] * pad_frames)
        item["rgb_member_names"] = member_names
        item["W"] = max_frames
        padded.append(item)
    return collate_anyspecies_padded(
        padded, max_joints=max_joints, dynamic_max_joints=dynamic_max_joints
    )


def collate_anyspecies_padded(
    batch, max_joints: int, dynamic_max_joints: bool = False
):
    if not isinstance(dynamic_max_joints, (bool, np.bool_)):
        raise ValueError(
            "dynamic_max_joints must be a boolean, "
            f"got {dynamic_max_joints!r}"
        )
    dynamic_max_joints = bool(dynamic_max_joints)
    if isinstance(max_joints, bool) or not isinstance(max_joints, int):
        raise ValueError(f"max_joints must be a positive integer, got {max_joints!r}")
    if max_joints <= 0:
        raise ValueError(f"max_joints must be positive, got {max_joints}")
    W = batch[0]["W"]
    if any(int(item["W"]) != int(W) for item in batch):
        raise ValueError(
            "Cannot collate samples with different temporal extents; "
            "complete-motion evaluation must use singleton batches"
        )

    # Never silently truncate a new skeleton.  All slicing below is only a
    # defensive shape bound after this explicit contract check.
    batch_max_joints = max(int(item["J"]) for item in batch)
    if batch_max_joints > max_joints:
        raise ValueError(
            f"Skeleton has {batch_max_joints} joints, fixed maximum is {max_joints}"
        )
    J_max = batch_max_joints if dynamic_max_joints else max_joints

    # ===== shared =====
    pos_list, fk_position_target_list, img_list, image_rgb_list = [], [], [], []
    full_image_rgb_list = []
    ref_pos_list = []
    joint_mask_list, ancestor_mask_list = [], []
    scale_list, metric_scale_list, J_valid_list, species_list = [], [], [], []
    rel_list = []
    frame_indices_list = []
    animation_frame_indices_list = []
    frame_valid_mask_list, valid_length_list = [], []
    motion_offset_list, motion_frames_list = [], []
    window_start_list, window_index_list = [], []
    rgb_tar_path_list = []
    rgb_member_names_list = []
    joint_names_list = []
    camera_root_position_list = []
    camera_intrinsics_list = []
    camera_image_size_list = []
    camera_extrinsics_list = []
    crop_box_list = []
    root_crop_camera_xyz_list = []
    hop_list, edge_list = [], []
    static_rot_joint_mask_list = []
    static_pos_joint_mask_list = []

    # ===== a =====
    rot_a_list = []
    ref_rot_a_list = []
    parent_a_list = []
    offset_a_list = []

    for b in batch:
        J = b["J"]
        J_valid_list.append(J)
        species_list.append(b["species"])
        rel_list.append(b.get("rel"))
        frame_indices_list.append(b.get("frame_indices"))
        animation_frame_indices_list.append(
            b.get("animation_frame_indices", b.get("frame_indices"))
        )
        frame_valid_mask_list.append(
            b.get("frame_valid_mask", np.ones((W,), dtype=np.bool_))
        )
        valid_length_list.append(int(b.get("valid_length", W)))
        motion_offset_list.append(b.get("motion_offset"))
        motion_frames_list.append(b.get("motion_frames"))
        window_start_list.append(b.get("window_start"))
        window_index_list.append(b.get("window_index"))
        rgb_tar_path_list.append(b.get("rgb_tar_path"))
        rgb_member_names_list.append(b.get("rgb_member_names"))
        sample_joint_names = b.get("joint_names")
        if sample_joint_names is None:
            raise KeyError(
                f"Cache sample {b.get('rel', '<unknown>')} is missing joint_names"
            )
        sample_joint_names = [str(name) for name in sample_joint_names]
        if len(sample_joint_names) != int(J):
            raise ValueError(
                f"Cache sample {b.get('rel', '<unknown>')} joint_names length "
                f"{len(sample_joint_names)} does not match J={J}"
            )
        joint_names_list.append(sample_joint_names)
        camera_root_position_list.append(b.get("camera_root_position"))
        camera_intrinsics_list.append(b.get("camera_intrinsics"))
        camera_image_size_list.append(b.get("camera_image_size"))
        camera_extrinsics_list.append(b.get("camera_extrinsics"))
        crop_box_list.append(b.get("crop_box"))
        root_crop_camera_xyz_list.append(b.get("root_crop_camera_xyz"))
        scale_list.append(b["global_scale"])
        metric_scale_list.append(b.get("metric_scale", 1.0))

        # -------------------------
        # static_rot_joint_mask
        # -------------------------
        static_rot_joint_mask = b["static_rot_joint_mask"]
        if J < J_max:
            pad = np.zeros((J_max - J,), dtype=np.bool_)
            static_rot_joint_mask = np.concatenate([static_rot_joint_mask, pad])
        static_rot_joint_mask_list.append(static_rot_joint_mask[:J_max])

        # -------------------------
        # static_pos_joint_mask
        # -------------------------
        static_pos_joint_mask = b["static_pos_joint_mask"]
        if J < J_max:
            pad = np.zeros((J_max - J,), dtype=np.bool_)
            static_pos_joint_mask = np.concatenate([static_pos_joint_mask, pad])
        static_pos_joint_mask_list.append(static_pos_joint_mask[:J_max])

        # -------------------------
        # joint mask
        # -------------------------
        mask = np.zeros((J_max,), dtype=np.bool_)
        mask[:min(J, J_max)] = True
        joint_mask_list.append(mask)

        # -------------------------
        # ancestor mask
        # -------------------------
        ancestor_mask = np.zeros((J_max, J_max), dtype=np.bool_)
        parent = b["parent_a"].squeeze(0)
        for i in range(min(J, J_max)):
            ancestor_mask[i, i] = True
            p = parent[i]
            while p != -1:
                ancestor_mask[i, p] = True
                p = parent[p]
        ancestor_mask_list.append(ancestor_mask)

        # -------------------------
        # hop / edge
        # -------------------------
        hop = b["graph_hop"]
        edge = b["graph_edge"]
        hop_pad = np.full((J_max, J_max), fill_value=5, dtype=np.int64)
        edge_pad = np.full((J_max, J_max), fill_value=4, dtype=np.int64)
        hop_pad[:J, :J] = hop
        edge_pad[:J, :J] = edge
        hop_list.append(hop_pad)
        edge_list.append(edge_pad)

        # -------------------------
        # position
        # -------------------------
        if "position" in b:
            pos = b["position"]
            if J < J_max:
                pad = np.zeros((W, J_max - J, 3), dtype=np.float32)
                pos = np.concatenate([pos, pad], axis=1)
            pos_list.append(pos[:, :J_max])
        if "fk_position_target" in b:
            fk_target = b["fk_position_target"]
            if J < J_max:
                pad = np.zeros((W, J_max - J, 3), dtype=np.float32)
                fk_target = np.concatenate([fk_target, pad], axis=1)
            fk_position_target_list.append(fk_target[:, :J_max])

        # -------------------------
        # rot a
        # -------------------------
        rot_a = b["rot6d_a"]
        if J < J_max:
            pad = np.zeros((W, J_max - J, 6), dtype=np.float32)
            rot_a = np.concatenate([rot_a, pad], axis=1)
        rot_a_list.append(rot_a[:, :J_max])

        # -------------------------
        # image_embed
        # -------------------------
        if "image_embed" in b:
            img_list.append(b["image_embed"])
        if "image_rgb" in b:
            image_rgb_list.append(b["image_rgb"])
        if "full_image_rgb" in b:
            full_image_rgb_list.append(b["full_image_rgb"])

        # -------------------------
        # ref position
        # -------------------------
        ref_pos = b["ref_position"]
        if J < J_max:
            pad = np.zeros((J_max - J, 3), dtype=np.float32)
            ref_pos = np.concatenate([ref_pos, pad], axis=0)
        ref_pos_list.append(ref_pos[:J_max])

        # -------------------------
        # ref rot a
        # -------------------------
        ref_rot_a = b["ref_rot6d_a"]
        if J < J_max:
            pad = np.zeros((J_max - J, 6), dtype=np.float32)
            ref_rot_a = np.concatenate([ref_rot_a, pad], axis=0)
        ref_rot_a_list.append(ref_rot_a[:J_max])

        # -------------------------
        # -------------------------
        # parent / offset a
        # -------------------------
        parent_a = b["parent_a"].squeeze(0)
        if J < J_max:
            pad = np.full((J_max - J,), -1, dtype=parent_a.dtype)
            parent_a = np.concatenate([parent_a, pad], axis=0)
        parent_a_list.append(parent_a[:J_max])

        offset_a = b["offset_a"].squeeze(0)
        if J < J_max:
            pad = np.zeros((J_max - J, 3), dtype=np.float32)
            offset_a = np.concatenate([offset_a, pad], axis=0)
        offset_a_list.append(offset_a[:J_max])

    result = {
        # ===== shared =====
        "ref_position": torch.from_numpy(np.stack(ref_pos_list, 0)),
        "joint_mask": torch.from_numpy(np.stack(joint_mask_list, 0)),
        "ancestor_mask": torch.from_numpy(np.stack(ancestor_mask_list, 0)),
        "J_valid": torch.tensor(J_valid_list, dtype=torch.int32),
        "global_scale": torch.from_numpy(np.array(scale_list)).float(),
        "metric_scale": torch.from_numpy(np.array(metric_scale_list)).float(),
        "species": species_list,
        "rel": rel_list,
        "frame_indices": frame_indices_list,
        "animation_frame_indices": animation_frame_indices_list,
        "frame_valid_mask": torch.from_numpy(np.stack(frame_valid_mask_list, 0)),
        "valid_length": valid_length_list,
        "motion_offset": motion_offset_list,
        "motion_frames": motion_frames_list,
        "window_start": window_start_list,
        "window_index": window_index_list,
        "rgb_tar_path": rgb_tar_path_list,
        "rgb_member_names": rgb_member_names_list,
        "joint_names": joint_names_list,
        "graph_hop": torch.from_numpy(np.stack(hop_list, 0)),
        "graph_edge": torch.from_numpy(np.stack(edge_list, 0)),
        "static_rot_joint_mask": torch.from_numpy(np.stack(static_rot_joint_mask_list, 0)),
        "static_pos_joint_mask": torch.from_numpy(np.stack(static_pos_joint_mask_list, 0)),

        # ===== view a =====
        "rot6d_a": torch.from_numpy(np.stack(rot_a_list, 0)),
        "ref_rot6d_a": torch.from_numpy(np.stack(ref_rot_a_list, 0)),
        "parent_a": torch.from_numpy(np.stack(parent_a_list, 0)),
        "offset_a": torch.from_numpy(np.stack(offset_a_list, 0)),

    }
    if pos_list:
        result["position"] = torch.from_numpy(np.stack(pos_list, 0))
    if fk_position_target_list:
        if len(fk_position_target_list) != len(batch):
            raise ValueError(
                "fk_position_target must be present for every batch item"
            )
        result["fk_position_target"] = torch.from_numpy(
            np.stack(fk_position_target_list, 0)
        )
    if img_list:
        result["image_embed"] = torch.from_numpy(np.stack(img_list, 0))
    if image_rgb_list:
        result["image_rgb"] = torch.from_numpy(np.stack(image_rgb_list, 0))
    if full_image_rgb_list:
        if len(full_image_rgb_list) != len(batch):
            raise ValueError("full_image_rgb must be present for every batch item")
        result["full_image_rgb"] = torch.from_numpy(
            np.stack(full_image_rgb_list, 0)
        )
    if all(value is not None for value in crop_box_list):
        result["crop_box"] = torch.from_numpy(np.stack(crop_box_list, 0))
    elif any(value is not None for value in crop_box_list):
        raise ValueError("crop_box must be present for every batch item")
    if all(value is not None for value in root_crop_camera_xyz_list):
        result["root_crop_camera_xyz"] = torch.from_numpy(
            np.stack(root_crop_camera_xyz_list, 0)
        )
    elif any(value is not None for value in root_crop_camera_xyz_list):
        raise ValueError(
            "root_crop_camera_xyz must be present for every batch item"
        )
    camera_values = (
        camera_root_position_list,
        camera_intrinsics_list,
        camera_image_size_list,
        camera_extrinsics_list,
    )
    if all(all(value is not None for value in values) for values in camera_values):
        result["camera_root_position"] = torch.from_numpy(
            np.stack(camera_root_position_list, 0)
        )
        result["camera_intrinsics"] = torch.from_numpy(
            np.stack(camera_intrinsics_list, 0)
        )
        result["camera_image_size"] = torch.from_numpy(
            np.stack(camera_image_size_list, 0)
        )
        # Original source-camera pose. Cached rotations/positions are already
        # in camera coordinates, so this is metadata rather than model input.
        result["camera_extrinsics"] = camera_extrinsics_list
    elif any(any(value is not None for value in values) for values in camera_values):
        raise ValueError("Camera metadata must be present for every batch item")
    return result
