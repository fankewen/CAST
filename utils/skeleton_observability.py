from __future__ import annotations

import math

import numpy as np
import torch

DEFAULT_EDGE_EPSILON = 1e-6


def resolve_edge_epsilon(cfg=None):
    """Return the shared rest-edge observability epsilon.

    Algorithm-1 components (observability catalog, virtual-anchor builder,
    BoneDirectionHead, virtual solver and loss/eval) must use the same
    epsilon when deciding whether a rest bone is non-degenerate.  The default
    lives here; callers may later override it through ``cfg`` without
    forking constants across modules.
    """
    if cfg is None:
        return float(DEFAULT_EDGE_EPSILON)
    model_params = cfg.get("model", {}).get("params", {})
    value = model_params.get("edge_epsilon", DEFAULT_EDGE_EPSILON)
    value = float(value)
    if not math.isfinite(value) or not value > 0.0:
        raise ValueError("model.params.edge_epsilon must be finite and positive")
    return value


def tree_pointer_jumping_state_torch(
    parents: torch.Tensor, joint_mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Return validated initial ancestors and pointer-jumping round count."""
    if parents.ndim != 2 or joint_mask.shape != parents.shape:
        raise ValueError("parents/joint_mask must be aligned [B,J]")
    parents = parents.long()
    joint_mask = joint_mask.bool()
    joints = parents.shape[1]
    joint_index = torch.arange(joints, device=parents.device).view(1, joints)
    valid_parent = (
        joint_mask
        & joint_index.gt(0)
        & parents.ge(0)
        & parents.lt(joint_index)
    )
    ancestor = torch.where(
        valid_parent, parents, torch.full_like(parents, -1)
    )
    rounds = max(1, max(joints - 1, 1).bit_length())
    return ancestor, rounds


def tree_pointer_jump_torch(
    ancestor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one pointer-jumping step from a validated ancestor state."""
    joints = ancestor.shape[1]
    has_ancestor = ancestor.ge(0)
    safe_ancestor = ancestor.clamp(min=0, max=max(joints - 1, 0))
    ancestor_parent = torch.gather(ancestor, 1, safe_ancestor)
    next_ancestor = torch.where(
        has_ancestor, ancestor_parent, torch.full_like(ancestor, -1)
    )
    return has_ancestor, safe_ancestor, next_ancestor


def tree_depth_from_parents_torch(
    parents: torch.Tensor, joint_mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Return batched joint depths using logarithmic pointer jumping."""
    ancestor, rounds = tree_pointer_jumping_state_torch(parents, joint_mask)
    depth = ancestor.ge(0).long()
    for _ in range(rounds):
        has_ancestor, safe_ancestor, next_ancestor = tree_pointer_jump_torch(
            ancestor
        )
        depth = depth + torch.where(
            has_ancestor,
            torch.gather(depth, 1, safe_ancestor),
            torch.zeros_like(depth),
        )
        ancestor = next_ancestor
    return depth, max(0, parents.shape[1] - 1)


def retained_observability_catalog_torch(
    parents: torch.Tensor,
    offsets: torch.Tensor,
    joint_mask: torch.Tensor | None = None,
    *,
    direction_epsilon: float = DEFAULT_EDGE_EPSILON,
    rank_tolerance: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Classify batched owner joints by retained-direction information rank.

    The direction stored at child slot ``c`` describes the retained edge
    ``parents[c] -> c``.  This function scatters each valid edge back to its
    rotation owner and evaluates

    ``H_j = sum_c (I - r_c r_c^T)``.

    All tensors are batch-shaped and the implementation contains no per-joint
    Python loop, so the catalog can be computed in a dataset cache or directly
    on the training device.  Rank is a topology/rest-pose property; callers
    should normally pass non-trainable offsets.
    """

    parents = torch.as_tensor(parents)
    offsets = torch.as_tensor(offsets, device=parents.device)
    if parents.ndim != 2 or offsets.shape != parents.shape + (3,):
        raise ValueError("parents/offsets must be [B,J]/[B,J,3]")
    if parents.dtype != torch.long:
        parents = parents.long()
    if not torch.is_floating_point(offsets):
        offsets = offsets.float()
    batch_size, joints = parents.shape
    if joint_mask is None:
        joint_mask = torch.ones_like(parents, dtype=torch.bool)
    else:
        joint_mask = torch.as_tensor(
            joint_mask, device=parents.device, dtype=torch.bool
        )
        if joint_mask.shape != parents.shape:
            raise ValueError("joint_mask must match parents")

    joint_index = torch.arange(joints, device=parents.device).view(1, joints)
    valid_parent = parents.ge(0) & parents.lt(joints)
    safe_parent = parents.clamp(min=0, max=max(joints - 1, 0))
    retained_edge = joint_mask & valid_parent
    edge_norm = torch.linalg.vector_norm(offsets, dim=-1)
    nonzero_edge = retained_edge & edge_norm.gt(float(direction_epsilon))
    direction = offsets / edge_norm.clamp_min(float(direction_epsilon)).unsqueeze(-1)
    direction = torch.where(nonzero_edge.unsqueeze(-1), direction, torch.zeros_like(direction))

    child_count = torch.zeros(
        batch_size, joints, dtype=torch.long, device=parents.device
    )
    child_count.scatter_add_(1, safe_parent, retained_edge.long())
    valid_count = torch.zeros_like(child_count)
    valid_count.scatter_add_(1, safe_parent, nonzero_edge.long())

    identity = torch.eye(3, dtype=offsets.dtype, device=offsets.device)
    edge_information = identity - direction.unsqueeze(-1) @ direction.unsqueeze(-2)
    edge_information = edge_information * nonzero_edge.unsqueeze(-1).unsqueeze(-1).to(offsets.dtype)
    information = torch.zeros(
        batch_size, joints, 3, 3, dtype=offsets.dtype, device=offsets.device
    )
    scatter_index = safe_parent.unsqueeze(-1).unsqueeze(-1).expand(
        batch_size, joints, 3, 3
    )
    information.scatter_add_(1, scatter_index, edge_information)

    eigenvalues, eigenvectors = torch.linalg.eigh(information.float())
    eigenvalues = eigenvalues.clamp_min(0.0)
    rank = eigenvalues.gt(float(rank_tolerance)).sum(dim=-1).long()
    rank = torch.where(joint_mask, rank, torch.zeros_like(rank))
    rank0 = joint_mask & rank.eq(0)
    rank2 = joint_mask & rank.eq(2)
    rank3 = joint_mask & rank.eq(3)

    null_axis = eigenvectors[..., :, 0]
    first_child_direction = torch.zeros(
        batch_size, joints, 3, dtype=offsets.dtype, device=offsets.device
    )
    first_score = torch.where(
        nonzero_edge,
        joint_index.expand(batch_size, joints),
        torch.full_like(parents, joints),
    )
    # amin gives the first valid child id for each owner without a joint loop.
    owner_first = torch.full(
        (batch_size, joints), joints, dtype=torch.long, device=parents.device
    )
    owner_first.scatter_reduce_(
        1, safe_parent, first_score, reduce="amin", include_self=True
    )
    safe_first = owner_first.clamp(max=max(joints - 1, 0))
    first_child_direction = torch.gather(
        direction, 1, safe_first.unsqueeze(-1).expand(batch_size, joints, 3)
    )
    flip = (null_axis * first_child_direction.float()).sum(dim=-1).lt(0.0)
    null_axis = torch.where(flip.unsqueeze(-1), -null_axis, null_axis)
    null_axis = torch.where(
        rank2.unsqueeze(-1), null_axis, torch.zeros_like(null_axis)
    )

    eigen_sum = eigenvalues.sum(dim=-1)
    condition_ratio = eigenvalues[..., 0] / eigen_sum.clamp_min(1e-8)
    condition_ratio = torch.where(
        valid_count.gt(0), condition_ratio, torch.zeros_like(condition_ratio)
    )
    zero_offset_nonleaf = joint_mask & child_count.gt(0) & valid_count.eq(0)
    collinear_multichild = rank2 & valid_count.gt(1)

    return {
        "child_count": child_count,
        "valid_direction_count": valid_count,
        "valid_edge_mask": nonzero_edge,
        "rank": rank,
        "rank0_mask": rank0,
        "rank2_mask": rank2,
        "rank3_mask": rank3,
        "null_axis": null_axis.to(offsets.dtype),
        "eigenvalues": eigenvalues.to(offsets.dtype),
        "information": information,
        "condition_ratio": condition_ratio.to(offsets.dtype),
        "zero_offset_nonleaf_mask": zero_offset_nonleaf,
        "collinear_multichild_mask": collinear_multichild,
    }


def retained_observability_catalog(
    parents: np.ndarray,
    offsets: np.ndarray,
    *,
    direction_epsilon: float = DEFAULT_EDGE_EPSILON,
    rank_tolerance: float = 1e-6,
) -> dict[str, np.ndarray]:
    """NumPy compatibility wrapper around the shared Torch definition."""

    parents_np = np.asarray(parents, dtype=np.int64)
    offsets_np = np.asarray(offsets, dtype=np.float64)
    if parents_np.ndim != 1 or offsets_np.shape != (len(parents_np), 3):
        raise ValueError("parents/offsets must be [J]/[J,3]")
    child_index = np.arange(len(parents_np), dtype=np.int64)
    invalid_order = (parents_np >= 0) & (parents_np >= child_index)
    if np.any(invalid_order):
        raise ValueError("parents must be topologically ordered")
    result = retained_observability_catalog_torch(
        torch.from_numpy(parents_np).unsqueeze(0),
        torch.from_numpy(offsets_np).unsqueeze(0),
        direction_epsilon=direction_epsilon,
        rank_tolerance=rank_tolerance,
    )
    return {
        key: value[0].cpu().numpy()
        for key, value in result.items()
        if key in {
            "child_count",
            "valid_direction_count",
            "rank",
            "null_axis",
            "eigenvalues",
            "condition_ratio",
            "zero_offset_nonleaf_mask",
            "collinear_multichild_mask",
        }
    }
