"""Forward kinematics: differentiable depth-batched Torch and per-motion NumPy."""

from __future__ import annotations

import numpy as np
import torch


def build_fk_layout(parents, device=None):
    """Build small indices on CPU; call before the batch leaves host memory.

    A parent index is clamped to the joints that precede it, so padded entries
    never leave the layout. Each depth reads exclusively from the previous
    depth, avoiding repeated concatenation of all earlier joint transforms.
    """
    array = parents.detach().cpu().numpy()
    batch, joints = array.shape
    if joints < 1:
        raise ValueError('FK requires at least one joint')
    safe = np.zeros_like(array)
    depth = np.zeros_like(array)
    rows = np.arange(batch)
    for joint in range(1, joints):
        safe[:, joint] = np.clip(array[:, joint], 0, joint - 1)
        depth[:, joint] = depth[rows, safe[:, joint]] + 1
    device = parents.device if device is None else device
    groups = []
    previous_slots = np.zeros(batch * joints, dtype=np.int64)
    order = []
    for level in range(int(depth.max()) + 1):
        ids = np.flatnonzero(depth.reshape(-1) == level)
        b, j = ids // joints, ids % joints
        parent_ids = b * joints + safe[b, j]
        slots = previous_slots[parent_ids]
        groups.append((torch.as_tensor(b, device=device),
                       torch.as_tensor(j, device=device),
                       torch.as_tensor(slots, device=device)))
        previous_slots[ids] = np.arange(len(ids))
        order.extend(ids.tolist())
    inverse = torch.as_tensor(np.argsort(order), device=device)
    return groups, inverse


def depth_fk(local_rotation, rest_offset, parents, layout=None):
    if local_rotation.ndim != 5 or local_rotation.shape[-2:] != (3, 3):
        raise ValueError('Local rotations must be [B,T,J,3,3]')
    batch, frames, joints = local_rotation.shape[:3]
    if rest_offset.shape != (batch, joints, 3) or parents.shape != (batch, joints):
        raise ValueError('FK offset/parent shape mismatch')
    groups, inverse = build_fk_layout(parents) if layout is None else layout
    rotations, positions = [], []
    for level, (b, j, parent_slot) in enumerate(groups):
        local = local_rotation[b, :, j]
        if level == 0:
            rotation = local
            position = local.new_zeros((len(b), frames, 3))
        else:
            parent_rotation = rotations[-1][parent_slot]
            parent_position = positions[-1][parent_slot]
            position = parent_position + (
                parent_rotation @ rest_offset[b, j, None, :, None]
            ).squeeze(-1)
            rotation = parent_rotation @ local
        rotations.append(rotation)
        positions.append(position)
    position = torch.cat(positions)[inverse].reshape(batch, joints, frames, 3)
    rotation = torch.cat(rotations)[inverse].reshape(batch, joints, frames, 3, 3)
    return position.permute(0, 2, 1, 3), rotation.permute(0, 2, 1, 3, 4)


def local_rotation_fk(
    local_rotation: np.ndarray,
    rest_offset: np.ndarray,
    parents: np.ndarray,
    fixed_root_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Build root-centered positions from root-global/child-local rotations.

    Joint zero contains the predicted camera/global root rotation, which
    ``fixed_root_rotation`` replaces when it is supplied.
    """
    local_rotation = np.asarray(local_rotation, dtype=np.float32)
    rest_offset = np.asarray(rest_offset, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    if local_rotation.ndim != 4 or local_rotation.shape[-2:] != (3, 3):
        raise ValueError(f"local_rotation must be [T,J,3,3], got {local_rotation.shape}")
    frames, joints = local_rotation.shape[:2]
    if rest_offset.shape != (joints, 3) or parents.shape != (joints,):
        raise ValueError("rest_offset/parents do not match rotation joint count")
    if fixed_root_rotation is not None:
        fixed_root_rotation = np.asarray(fixed_root_rotation, dtype=np.float32)
        if fixed_root_rotation.shape != (3, 3):
            raise ValueError("fixed_root_rotation must be [3,3]")
    if parents[0] != -1 or any(not (0 <= int(p) < j) for j, p in enumerate(parents[1:], 1)):
        raise ValueError("parents must be root-0, parent-before-child order")

    position = np.zeros((frames, joints, 3), dtype=np.float32)
    global_rotation = np.zeros((frames, joints, 3, 3), dtype=np.float32)
    global_rotation[:, 0] = (
        local_rotation[:, 0]
        if fixed_root_rotation is None
        else fixed_root_rotation
    )
    for joint in range(1, joints):
        parent = int(parents[joint])
        position[:, joint] = position[:, parent] + np.einsum(
            "tij,j->ti", global_rotation[:, parent], rest_offset[joint]
        )
        global_rotation[:, joint] = (
            global_rotation[:, parent] @ local_rotation[:, joint]
        )
    return position
