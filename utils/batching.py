"""Batching cached samples into padded per-species tensors."""





import math
import torch
from data.cache_reader import collate_anyspecies_padded
from utils.config_validation import require_positive_int


_SINGLETON_JOINT_AXES = {
    "ref_position": (1,),
    "joint_mask": (1,),
    "ancestor_mask": (1, 2),
    "graph_hop": (1, 2),
    "graph_edge": (1, 2),
    "static_rot_joint_mask": (1,),
    "static_pos_joint_mask": (1,),
    "rot6d_a": (2,),
    "ref_rot6d_a": (1,),
    "parent_a": (1,),
    "offset_a": (1,),
    "position": (2,),
    "fk_position_target": (2,),
}


_SINGLETON_JOINT_PAD_VALUES = {
    "parent_a": -1,
    "graph_hop": 5,
    "graph_edge": 4,
}


def collate_single_anyspecies_padded(
    sample, *, max_joints: int, dynamic_max_joints: bool = False
):
    """Collate one dataset sample inside its DataLoader worker.

    The leading singleton batch dimension lets DataLoader workers return
    samples independently. The main process later joins them into the actual
    model batch and applies bounded pinning once per merged tensor.
    """
    return collate_anyspecies_padded(
        [sample],
        max_joints=max_joints,
        dynamic_max_joints=dynamic_max_joints,
    )


def _merge_variable_joint_tensors(
    key, values, *, pin_memory, pin_memory_max_bytes
):
    """Merge singleton tensors whose joint axes use per-sample widths."""
    joint_axes = _SINGLETON_JOINT_AXES.get(key)
    if joint_axes is None:
        raise ValueError(
            f"Tensor field {key!r} differs across singleton batches"
        )
    first = values[0]
    if any(value.dtype != first.dtype or value.device != first.device for value in values):
        raise ValueError(
            f"Tensor field {key!r} differs across singleton batches"
        )
    output_shape = [len(values), *first.shape[1:]]
    target_joints = max(value.shape[axis] for value in values for axis in joint_axes)
    for axis in joint_axes:
        output_shape[axis] = target_joints
    output_shape = tuple(output_shape)
    for value in values:
        if value.ndim != first.ndim or value.shape[0] != 1:
            raise ValueError(
                f"Tensor field {key!r} must have a singleton leading "
                "batch dimension in every worker result"
            )
        if any(value.shape[axis] != value.shape[joint_axes[0]] for axis in joint_axes):
            raise ValueError(f"Tensor field {key!r} has inconsistent joint axes")
        for axis, (actual, expected) in enumerate(zip(value.shape, first.shape)):
            if axis not in joint_axes and actual != expected:
                raise ValueError(
                    f"Tensor field {key!r} differs across singleton batches"
                )
    should_pin_output = (
        all(value.is_pinned() for value in values)
        if pin_memory is None
        else pin_memory
    )
    output_bytes = math.prod(output_shape) * first.element_size()
    if pin_memory_max_bytes is not None and output_bytes > pin_memory_max_bytes:
        should_pin_output = False
    allocation_kwargs = {"dtype": first.dtype, "device": first.device}
    if first.device.type == "cpu" and should_pin_output:
        allocation_kwargs["pin_memory"] = True
    output = torch.empty(output_shape, **allocation_kwargs)
    output.fill_(_SINGLETON_JOINT_PAD_VALUES.get(key, 0))
    for batch_index, value in enumerate(values):
        destination = [slice(None)] * output.ndim
        destination[0] = batch_index
        for axis in joint_axes:
            destination[axis] = slice(0, value.shape[axis])
        output[tuple(destination)] = value[0]
    return output


def merge_singleton_batches(
    singletons, *, pin_memory=None, pin_memory_max_bytes=None
):
    """Join worker-produced singleton batches without re-running collation.

    ``DataLoader(pin_memory=True)`` pins every singleton before this function
    sees it.  For large worker-parallel batches that retains all singleton
    pinned buffers while allocating another full pinned output batch.  Passing
    ``pin_memory=True`` with an unpinned loader instead pins only merged
    outputs up to ``pin_memory_max_bytes``. Larger tensors remain pageable and
    are transferred through bounded pinned staging chunks. ``None`` matches
    the pinning of the input tensors.
    """
    if not singletons:
        raise ValueError("Cannot merge an empty list of singleton batches")
    expected_keys = set(singletons[0])
    for index, singleton in enumerate(singletons):
        if set(singleton) != expected_keys:
            raise ValueError(
                f"Singleton batch {index} has inconsistent keys: "
                f"expected={sorted(expected_keys)}, got={sorted(singleton)}"
            )

    merged = {}
    for key in singletons[0]:
        values = [singleton[key] for singleton in singletons]
        first = values[0]
        if isinstance(first, dict):
            if not all(isinstance(value, dict) for value in values):
                raise ValueError(f"Nested field {key!r} differs across batches")
            merged[key] = merge_singleton_batches(
                values,
                pin_memory=pin_memory,
                pin_memory_max_bytes=pin_memory_max_bytes,
            )
        elif isinstance(first, torch.Tensor):
            if any(
                not isinstance(value, torch.Tensor)
                or value.ndim == 0
                or value.shape[0] != 1
                for value in values
            ):
                raise ValueError(
                    f"Tensor field {key!r} must have a singleton leading "
                    "batch dimension in every worker result"
                )
            output_shape = (len(values), *first.shape[1:])
            same_shape_dtype = all(
                value.shape == first.shape and value.dtype == first.dtype
                for value in values
            )
            if not same_shape_dtype:
                merged[key] = _merge_variable_joint_tensors(
                    key,
                    values,
                    pin_memory=pin_memory,
                    pin_memory_max_bytes=pin_memory_max_bytes,
                )
                continue
            should_pin_output = (
                all(value.is_pinned() for value in values)
                if pin_memory is None
                else pin_memory
            )
            output_bytes = math.prod(output_shape) * first.element_size()
            if (
                pin_memory_max_bytes is not None
                and output_bytes > pin_memory_max_bytes
            ):
                should_pin_output = False
            if first.device.type == "cpu" and should_pin_output:
                output = torch.empty(
                    output_shape,
                    dtype=first.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                torch.cat(values, dim=0, out=output)
                merged[key] = output
            else:
                merged[key] = torch.cat(values, dim=0)
        elif isinstance(first, list):
            flattened = []
            for value in values:
                if not isinstance(value, list) or len(value) != 1:
                    raise ValueError(
                        f"List field {key!r} must contain exactly one item "
                        "in every worker result"
                    )
                flattened.extend(value)
            merged[key] = flattened
        else:
            raise TypeError(
                f"Unsupported singleton batch field {key!r}: "
                f"{type(first).__name__}"
            )
    return merged


def _release_worker_shared_storage(value):
    """Move queued CPU tensors into process-local RAM before batch assembly.

    Worker IPC tensors live in /dev/shm. Retaining an entire model batch across
    all ranks can exhaust that filesystem even with prefetch_factor=1.
    Copy as each sample arrives so shared storage scales with worker prefetch,
    not the model batch size. Non-shared and CUDA tensors need no extra copy.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu" and value.is_shared():
            return value.clone()
        return value
    if isinstance(value, dict):
        return {key: _release_worker_shared_storage(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_release_worker_shared_storage(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_release_worker_shared_storage(item) for item in value)
    return value


def iter_worker_parallel_batches(
    loader,
    batch_size,
    *,
    drop_last,
    pin_memory=None,
    pin_memory_max_bytes=None,
):
    """Assemble model batches from independently loaded worker samples.

    PyTorch automatic batching sends all indices of one batch to one worker.
    With 81-frame cache samples that serializes every expensive tar read of the
    batch in a single process.  This iterator keeps DataLoader automatic batching
    off, receives one already-collated sample at a time from every worker, and
    only then forms the model batch in the main process.
    """
    batch_size = require_positive_int(batch_size, "loader batch_size")
    pending = []
    for singleton in loader:
        singleton = _release_worker_shared_storage(singleton)
        pending.append(singleton)
        if len(pending) == batch_size:
            merged = merge_singleton_batches(
                pending,
                pin_memory=pin_memory,
                pin_memory_max_bytes=pin_memory_max_bytes,
            )
            # The merged tensors own their storage. Release worker shared-memory
            # samples before suspending this generator (including validation).
            pending.clear()
            del singleton
            yield merged
            del merged
    if pending and not drop_last:
        merged = merge_singleton_batches(
            pending,
            pin_memory=pin_memory,
            pin_memory_max_bytes=pin_memory_max_bytes,
        )
        pending.clear()
        del singleton
        yield merged
