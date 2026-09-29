"""Unified source registry for RGB-motion cache datasets."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from data.cache_reader import (
    build_cache_dataset,
    prepare_cache_tar_inventories,
)


_SOURCE_CONTROL_KEYS = {
    "weight",
    "validation",
    "train_source_splits",
}
_DATA_CONTROL_KEYS = {
    "sources",
    "primary_validation_source",
}

_VALIDATION_IDENTITY_KEYS = frozenset(
    {
        "dataset_root",
        "split_file",
        "full_image_size",
    }
)

_COMMON_SAMPLE_KEYS = frozenset(
    {
        "rel",
        "species",
        "F",
        "J",
        "W",
        "frame_indices",
        "animation_frame_indices",
        "frame_valid_mask",
        "valid_length",
        "motion_offset",
        "motion_frames",
        "window_start",
        "window_index",
        "rgb_tar_path",
        "rgb_member_names",
        "joint_names",
        "global_scale",
        "metric_scale",
        "ref_position",
        "graph_hop",
        "graph_edge",
        "static_rot_joint_mask",
        "static_pos_joint_mask",
        "position",
        "rot6d_a",
        "ref_rot6d_a",
        "parent_a",
        "offset_a",
        "image_rgb",
    }
)

_CAPABILITY_SAMPLE_KEYS = {
    "root_translation": frozenset(
        {
            "crop_box",
            "root_crop_camera_xyz",
            "camera_root_position",
            "camera_intrinsics",
            "camera_image_size",
            "camera_extrinsics",
        }
    ),
    "full_image": frozenset({"full_image_rgb"}),
}

_CAPABILITY_ROOT_TRANSLATION = "root_translation"
_CAPABILITY_FULL_IMAGE = "full_image"


@dataclass(frozen=True)
class CacheSourceBundle:
    train: Dict[str, Dataset]
    train_weights: Dict[str, float]
    validation: Dict[str, Dataset]
    primary_validation_name: str
    cache_versions: Dict[str, int]
    train_sample_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


def _normalize_splits(value: Any, name: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = tuple(str(item) for item in value)
    else:
        raise ValueError(f"{name} must be null, a string, or a sequence")
    values = tuple(item for item in values if item)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(dict.fromkeys(values))



def _source_configs(data_cfg: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    sources = data_cfg.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("data.sources must be a non-empty mapping")
    common = {
        key: value
        for key, value in data_cfg.items()
        if key not in _DATA_CONTROL_KEYS
    }
    resolved: Dict[str, Dict[str, Any]] = {}
    for raw_name, raw_source in sources.items():
        name = str(raw_name)
        if not name:
            raise ValueError("Cache source names must not be empty")
        if not isinstance(raw_source, Mapping):
            raise TypeError(f"data.sources.{name} must be a mapping")
        source = {**common, **dict(raw_source)}
        if not source.get("dataset_root"):
            raise KeyError(f"Cache source {name!r} must define dataset_root")
        resolved[name] = source
    return resolved


def _backend_cfg(source_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip source controls before passing config to the cache reader."""
    return {
        key: value for key, value in source_cfg.items()
        if key not in _SOURCE_CONTROL_KEYS
    }


def _require_fixed_split(
    source_name: str,
    source_cfg: Mapping[str, Any],
) -> None:
    split_file = source_cfg.get("split_file")
    if not split_file:
        raise ValueError(
            f"Source {source_name!r} must define split_file for stable train/validation"
        )
    if not os.path.isfile(os.fspath(split_file)):
        raise FileNotFoundError(
            f"Source {source_name!r} split_file does not exist: {split_file}"
        )


def _source_sample_keys(
    source_cfg: Mapping[str, Any],
    enabled_capabilities: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Declare the sample keys emitted by the unified cache reader."""
    keys = set(_COMMON_SAMPLE_KEYS)
    keys.update(_CAPABILITY_SAMPLE_KEYS[_CAPABILITY_ROOT_TRANSLATION])
    if source_cfg.get("full_image_size") is not None:
        keys.add("full_image_rgb")
    return frozenset(keys)


def _validate_required_capabilities(
    source_keys: Mapping[str, frozenset[str]],
    required_capabilities: frozenset[str],
) -> None:
    """Fail fast when a configured supervision contract is not supported by
    every source participating in the same train/validation run."""
    for capability in sorted(required_capabilities):
        if capability not in _CAPABILITY_SAMPLE_KEYS:
            raise ValueError(f"Unknown sample capability {capability!r}")
        expected = _CAPABILITY_SAMPLE_KEYS[capability]
        missing = sorted(
            name for name, keys in source_keys.items() if not expected.issubset(keys)
        )
        if missing:
            raise ValueError(
                f"Training configuration requires {capability!r} sample fields "
                f"{sorted(expected)} but source(s) {missing} lack this capability"
            )


def required_sample_capabilities(cfg: Mapping[str, Any]) -> frozenset[str]:
    """Return sample capabilities required by the active model configuration."""
    capabilities = set()
    model_params = cfg.get("model", {}).get("params", {})
    if isinstance(model_params, Mapping):
        root_cfg = model_params.get("root_translation_cfg")
        if root_cfg is not None:
            capabilities.add(_CAPABILITY_ROOT_TRANSLATION)
            if isinstance(root_cfg, Mapping):
                root_target = str(root_cfg.get("target", ""))
                if root_target.endswith(".FullFrameRootXYZHead"):
                    capabilities.add(_CAPABILITY_FULL_IMAGE)
    return frozenset(capabilities)


def _stable_source_seed(name: str) -> int:
    """Derive a stable per-source stream seed from the source name."""
    digest = hashlib.sha256(str(name).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _make_sample_contract_transform(
    common_keys: frozenset[str],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Keep only the negotiated common sample contract for mixed collation."""

    def transform(sample: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in sample.items() if key in common_keys}

    return transform


def prepare_cache_inventories(data_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name, source_cfg in _source_configs(data_cfg).items():
        train_splits = _normalize_splits(
            source_cfg.get("train_source_splits"),
            f"data.sources.{name}.train_source_splits",
        )
        summary[name] = prepare_cache_tar_inventories(
            _backend_cfg(source_cfg), source_splits=train_splits
        )
    return summary


def build_cache_source_bundle(
    data_cfg: Mapping[str, Any],
    train_window: int,
    eval_window: int,
    *,
    required_capabilities: Sequence[str] = (),
    validation_first_window_eval: bool = True,
    validation_eval_stride: Optional[int] = None,
) -> CacheSourceBundle:
    """Build any one, two, or three compatible cache sources uniformly.

    The optional ``required_capabilities`` contract is validated across every
    source and every requested validation split before any dataset is built,
    so train and validation never silently observe different supervision
    fields.
    """
    sources = _source_configs(data_cfg)
    required_capabilities = frozenset(required_capabilities)
    prepared: Dict[str, Dict[str, Any]] = {}
    source_keys: Dict[str, frozenset[str]] = {}
    train_weights: Dict[str, float] = {}
    seen_validation_names: set[str] = set()

    for name, source_cfg in sources.items():
        train_splits = _normalize_splits(
            source_cfg.get("train_source_splits"),
            f"data.sources.{name}.train_source_splits",
        )
        _require_fixed_split(name, source_cfg)
        weight = float(source_cfg.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"data.sources.{name}.weight must be finite and positive")
        train_weights[name] = weight
        source_keys[name] = _source_sample_keys(source_cfg, required_capabilities)

        raw_validation = source_cfg.get("validation", {name: {}})
        if not isinstance(raw_validation, Mapping) or not raw_validation:
            raise ValueError(
                f"data.sources.{name}.validation must be a non-empty mapping"
            )
        validation_entries: list[tuple[str, Optional[Tuple[str, ...]], Dict[str, Any]]] = []
        for raw_validation_name, raw_validation_cfg in raw_validation.items():
            validation_name = str(raw_validation_name)
            if validation_name in seen_validation_names:
                raise ValueError(f"Duplicate validation name {validation_name!r}")
            seen_validation_names.add(validation_name)
            if raw_validation_cfg is None:
                raw_validation_cfg = {}
            if not isinstance(raw_validation_cfg, Mapping):
                raise TypeError(
                    f"data.sources.{name}.validation.{validation_name} must be a mapping"
                )
            forbidden = sorted(
                _VALIDATION_IDENTITY_KEYS & set(dict(raw_validation_cfg))
            )
            if forbidden:
                raise ValueError(
                    f"data.sources.{name}.validation.{validation_name} may not "
                    f"override source identity keys: {forbidden}"
                )
            validation_splits = _normalize_splits(
                raw_validation_cfg.get("source_splits"),
                (
                    f"data.sources.{name}.validation.{validation_name}."
                    "source_splits"
                ),
            )
            _require_fixed_split(name, source_cfg)
            validation_entries.append(
                (validation_name, validation_splits, dict(raw_validation_cfg))
            )

        prepared[name] = {
            "source_cfg": source_cfg,
            "train_splits": train_splits,
            "backend_cfg": _backend_cfg(source_cfg),
            "validation_entries": validation_entries,
        }

    _validate_required_capabilities(source_keys, required_capabilities)

    if len(sources) > 1:
        common_keys = frozenset.intersection(*source_keys.values())
        train_sample_transform = _make_sample_contract_transform(common_keys)
    else:
        train_sample_transform = None

    train_datasets: Dict[str, Dataset] = {}
    validation_datasets: Dict[str, Dataset] = {}
    for name, prep in prepared.items():
        source_cfg = prep["source_cfg"]
        train_splits = prep["train_splits"]
        backend_cfg = prep["backend_cfg"]
        train_dataset = build_cache_dataset(
            backend_cfg,
            "train",
            train_window,
            source_splits=train_splits,
            eval_ratio=0.0 if train_splits is not None else None,
        )
        train_datasets[name] = train_dataset

        for validation_name, validation_splits, raw_validation_cfg in prep[
            "validation_entries"
        ]:
            validation_cfg = {**backend_cfg, **raw_validation_cfg}
            validation_cfg.pop("source_splits", None)
            if validation_eval_stride is not None:
                validation_cfg["eval_stride"] = validation_eval_stride
            validation_dataset = build_cache_dataset(
                validation_cfg,
                "test",
                eval_window,
                first_window_eval=validation_first_window_eval,
                complete_motion_eval=False,
                source_splits=validation_splits,
                eval_ratio=1.0 if validation_splits is not None else None,
            )
            validation_datasets[validation_name] = validation_dataset

    primary = str(
        data_cfg.get("primary_validation_source", next(iter(validation_datasets)))
    )
    if primary not in validation_datasets:
        raise KeyError(
            f"Unknown primary validation source {primary!r}; "
            f"available={sorted(validation_datasets)}"
        )
    return CacheSourceBundle(
        train=train_datasets,
        train_weights=train_weights,
        validation=validation_datasets,
        primary_validation_name=primary,
        cache_versions={},
        train_sample_transform=train_sample_transform,
    )

class CacheStepIterableDataset(IterableDataset):
    """Infinite deterministic shuffled stream for any cache dataset."""

    def __init__(
        self,
        dataset: Dataset,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        sample_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        super().__init__()
        if len(dataset) <= 0:
            raise ValueError("CacheStepIterableDataset requires non-empty data")
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.sample_transform = sample_transform
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid cache stream rank/world_size")
        self.locality_groups = self._build_locality_groups(dataset)

    @classmethod
    def _build_locality_groups(cls, dataset: Dataset):
        from torch.utils.data import ConcatDataset
        groups: Dict[Any, list[int]] = {}
        if isinstance(dataset, ConcatDataset):
            offset = 0
            for dataset_index, child in enumerate(dataset.datasets):
                child_groups = cls._build_locality_groups(child)
                if child_groups is None:
                    return None
                for key, indices in child_groups.items():
                    groups[(dataset_index, key)] = [offset + index for index in indices]
                offset += len(child)
            return groups
        locality_key = getattr(dataset, "io_locality_key", None)
        if locality_key is None:
            return None
        for index in range(len(dataset)):
            groups.setdefault(locality_key(index), []).append(index)
        return groups

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        consumer_count = self.world_size * worker_count
        consumer_id = self.rank * worker_count + worker_id
        if len(self.dataset) < consumer_count:
            raise RuntimeError(
                f"Cache stream has {len(self.dataset)} samples but "
                f"{consumer_count} rank/workers"
            )
        cycle = 0
        while True:
            rng = np.random.default_rng(self.seed + cycle)
            if self.locality_groups is None:
                order = rng.permutation(len(self.dataset))
                consumer_order = order[consumer_id::consumer_count]
            else:
                keys = list(self.locality_groups)
                rng.shuffle(keys)
                order = []
                for key in keys:
                    indices = np.asarray(self.locality_groups[key], dtype=np.int64)
                    order.extend(rng.permutation(indices).tolist())
                consumer_order = np.array_split(
                    np.asarray(order, dtype=np.int64), consumer_count
                )[consumer_id]
            for index in consumer_order:
                sample = self.dataset[int(index)]
                if self.sample_transform is not None:
                    sample = self.sample_transform(sample)
                yield sample
            cycle += 1


class MixedCacheStepIterableDataset(IterableDataset):
    """Mix source streams using explicit normalized source probabilities.

    Source ordering is normalized (sorted by name) and each per-source stream
    seed is derived from the stable source name, so changing YAML insertion
    order or the hosting process hash seed does not change the sampled stream.
    """

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        weights: Mapping[str, float],
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        sample_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        super().__init__()
        if not datasets or set(datasets) != set(weights):
            raise ValueError("Mixed cache datasets and weights must have identical keys")
        self.datasets = dict(datasets)
        self.names = tuple(sorted(datasets))
        probabilities = np.asarray([float(weights[name]) for name in self.names])
        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
            raise ValueError("Mixed cache weights must be finite and positive")
        self.probabilities = probabilities / probabilities.sum()
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.sample_transform = sample_transform

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        consumer = self.rank * worker_count + worker_id
        rng = np.random.default_rng(self.seed + 10_000_019 * consumer)
        streams = {
            name: iter(
                CacheStepIterableDataset(
                    self.datasets[name],
                    rank=self.rank,
                    world_size=self.world_size,
                    seed=self.seed + _stable_source_seed(name),
                    sample_transform=self.sample_transform,
                )
            )
            for name in self.names
        }
        while True:
            name = str(rng.choice(self.names, p=self.probabilities))
            yield next(streams[name])
