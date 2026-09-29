"""Visual bone-direction prediction and exact-length kinematic lifting."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.skeleton_observability import DEFAULT_EDGE_EPSILON


def _validate_topology_tensors(
    parents: torch.Tensor,
    rest_offset: torch.Tensor,
    joint_mask: torch.Tensor,
) -> None:
    if parents.ndim != 2:
        raise ValueError(f"parents must be [B,J], got {tuple(parents.shape)}")
    if rest_offset.shape != parents.shape + (3,):
        raise ValueError(
            "rest_offset must be [B,J,3] aligned with parents, got "
            f"{tuple(rest_offset.shape)}/{tuple(parents.shape)}"
        )
    if joint_mask.shape != parents.shape:
        raise ValueError(
            "joint_mask must align with parents, got "
            f"{tuple(joint_mask.shape)}/{tuple(parents.shape)}"
        )
    if parents.shape[1] and not bool((parents[:, 0] == -1).all()):
        raise ValueError("joint zero must be the root with parent -1")
    joints = parents.shape[1]
    if joints > 1:
        joint_index = torch.arange(joints, device=parents.device).view(1, -1)
        invalid_order = joint_mask & joint_index.gt(0) & (
            parents.lt(0) | parents.ge(joint_index)
        )
        if bool(invalid_order.any()):
            raise ValueError("valid joints must be in parent-before-child order")


def canonical_fallback_axis(axis: str) -> str:
    """Validate a Cartesian fallback-axis selector."""

    axis = str(axis).lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(
            "zero-rest-edge fallback axis must be one of ['x', 'y', 'z'], "
            f"got {axis!r}"
        )
    return axis


def _axis_fallback_like(vector: torch.Tensor, axis: str) -> torch.Tensor:
    fallback = torch.zeros_like(vector)
    fallback[..., {"x": 0, "y": 1, "z": 2}[canonical_fallback_axis(axis)]] = 1.0
    return fallback


def safe_unit_vector(
    vector: torch.Tensor,
    fallback: Optional[torch.Tensor] = None,
    eps: float = DEFAULT_EDGE_EPSILON,
    fallback_axis: str = "x",
) -> torch.Tensor:
    """Normalize vectors with a deterministic finite Cartesian fallback."""

    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    default = _axis_fallback_like(vector, fallback_axis)
    if fallback is None:
        fallback = default
    fallback_norm = torch.linalg.vector_norm(fallback, dim=-1, keepdim=True)
    fallback = torch.where(
        fallback_norm > eps,
        fallback / fallback_norm.clamp_min(eps),
        default,
    )
    return torch.where(norm > eps, vector / norm.clamp_min(eps), fallback)


def child_count_from_parents(
    parents: torch.Tensor,
    joint_mask: torch.Tensor,
) -> torch.Tensor:
    """Count valid retained children for every joint."""

    if parents.shape != joint_mask.shape:
        raise ValueError("parents and joint_mask must have the same shape")
    batch_size, joints = parents.shape
    count = torch.zeros(
        batch_size,
        joints,
        dtype=torch.long,
        device=parents.device,
    )
    valid_child = joint_mask & parents.ge(0) & parents.lt(joints)
    safe_parent = parents.clamp(min=0, max=max(joints - 1, 0))
    count.scatter_add_(1, safe_parent, valid_child.long())
    return count


def positions_to_bone_directions(
    position: torch.Tensor,
    parents: torch.Tensor,
    rest_offset: Optional[torch.Tensor] = None,
    joint_mask: Optional[torch.Tensor] = None,
    frame_mask: Optional[torch.Tensor] = None,
    reliability_temperature: float = 0.25,
    edge_epsilon: float = DEFAULT_EDGE_EPSILON,
    zero_rest_edge_fallback_axis: str = "x",
    include_zero_rest_edges: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert root-relative positions to parent->child unit directions.

    The returned direction at joint ``j`` describes the edge from
    ``parents[j]`` to ``j``.  Root and invalid entries are exactly zero.
    When rest offsets are provided, confidence measures bone-length
    consistency and remains independent of any GT target.  ``edge_epsilon``
    is the shared semantic edge threshold used for both direction
    normalization and rest-edge validity.
    """

    edge_epsilon = float(edge_epsilon)
    if not 0.0 < edge_epsilon < float("inf"):
        raise ValueError("edge_epsilon must be finite and positive")
    if position.ndim != 4 or position.shape[-1] != 3:
        raise ValueError(
            f"position must be [B,T,J,3], got {tuple(position.shape)}"
        )
    batch_size, frames, joints = position.shape[:3]
    if parents.shape != (batch_size, joints):
        raise ValueError("parents do not match position")
    if joint_mask is None:
        joint_mask = torch.ones(
            batch_size, joints, dtype=torch.bool, device=position.device
        )
    else:
        joint_mask = joint_mask.bool()
    if frame_mask is None:
        frame_mask = torch.ones(
            batch_size, frames, dtype=torch.bool, device=position.device
        )
    else:
        frame_mask = frame_mask.bool()

    safe_parent = parents.clamp(min=0, max=max(joints - 1, 0))
    parent_index = safe_parent[:, None, :, None].expand(
        batch_size, frames, joints, 3
    )
    parent_position = torch.gather(position, 2, parent_index)
    edge = position - parent_position
    length = torch.linalg.vector_norm(edge, dim=-1)

    fallback = None
    rest_length = None
    if rest_offset is not None:
        if rest_offset.shape != (batch_size, joints, 3):
            raise ValueError("rest_offset does not match position")
        rest_length = torch.linalg.vector_norm(rest_offset, dim=-1)
        fallback = rest_offset[:, None].expand(batch_size, frames, joints, 3)
    direction = safe_unit_vector(
        edge,
        fallback=fallback,
        eps=edge_epsilon,
        fallback_axis=zero_rest_edge_fallback_axis,
    )

    valid_edge = (
        joint_mask
        & parents.ge(0)
        & parents.lt(joints)
    )
    if rest_length is not None and not include_zero_rest_edges:
        valid_edge = valid_edge & rest_length.gt(edge_epsilon)
    valid = frame_mask.unsqueeze(-1) & valid_edge.unsqueeze(1)
    direction = direction * valid.unsqueeze(-1).to(direction.dtype)

    if rest_offset is None:
        confidence = torch.ones_like(length)
        length_log_error = torch.zeros_like(length)
    else:
        if not reliability_temperature > 0:
            raise ValueError("reliability_temperature must be positive")
        ratio = length / rest_length[:, None].clamp_min(edge_epsilon)
        length_log_error = torch.abs(
            torch.log(ratio.clamp_min(edge_epsilon))
        )
        confidence = torch.exp(
            -length_log_error / float(reliability_temperature)
        )
    confidence = confidence * valid.to(confidence.dtype)
    length_log_error = length_log_error * valid.to(length_log_error.dtype)
    return {
        "direction": direction,
        "confidence": confidence,
        "valid_mask": valid,
        "bone_length": length,
        "bone_length_log_error": length_log_error,
    }


def reconstruct_positions_from_bone_directions(
    direction: torch.Tensor,
    rest_offset: torch.Tensor,
    parents: torch.Tensor,
    joint_mask: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
    edge_epsilon: float = DEFAULT_EDGE_EPSILON,
) -> torch.Tensor:
    """Lift edge directions using exact rest lengths in hierarchy order."""

    edge_epsilon = float(edge_epsilon)
    if not 0.0 < edge_epsilon < float("inf"):
        raise ValueError("edge_epsilon must be finite and positive")
    if direction.ndim != 4 or direction.shape[-1] != 3:
        raise ValueError("direction must be [B,T,J,3]")
    batch_size, frames, joints = direction.shape[:3]
    joint_mask = joint_mask.bool()
    _validate_topology_tensors(parents, rest_offset, joint_mask)
    if frame_mask is None:
        frame_mask = torch.ones(
            batch_size, frames, dtype=torch.bool, device=direction.device
        )
    else:
        frame_mask = frame_mask.bool()
        if frame_mask.shape != (batch_size, frames):
            raise ValueError("frame_mask does not match direction")

    rest_length = torch.linalg.vector_norm(rest_offset, dim=-1)
    rest_direction = safe_unit_vector(rest_offset, eps=edge_epsilon)
    unit_direction = safe_unit_vector(
        direction,
        fallback=rest_direction[:, None].expand_as(direction),
        eps=edge_epsilon,
    )
    zero = direction.new_zeros(batch_size, frames, 3)
    positions = [zero]
    for joint in range(1, joints):
        previous = torch.stack(positions, dim=2)
        parent = parents[:, joint]
        safe_parent = parent.clamp(min=0, max=joint - 1)
        gather_index = safe_parent[:, None, None, None].expand(
            batch_size, frames, 1, 3
        )
        parent_position = torch.gather(previous, 2, gather_index).squeeze(2)
        edge = unit_direction[:, :, joint] * rest_length[:, None, joint, None]
        valid = (
            frame_mask
            & joint_mask[:, joint].unsqueeze(1)
            & parent.ge(0).unsqueeze(1)
            & parent.lt(joint).unsqueeze(1)
            & rest_length[:, joint].gt(edge_epsilon).unsqueeze(1)
        )
        position = torch.where(
            valid.unsqueeze(-1), parent_position + edge, zero
        )
        positions.append(position)
    return torch.stack(positions, dim=2)


class BoneDirectionHead(nn.Module):
    """Predict camera-space edge directions from parent/child visual tokens."""

    def __init__(
        self,
        feature_dim: int = 1152,
        hidden_dim: int = 384,
        confidence_bias: float = -4.0,
        zero_rest_edge_fallback_axis: str = "x",
        include_zero_rest_edges: bool = False,
        edge_epsilon: float = DEFAULT_EDGE_EPSILON,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        if not 0.0 < float(edge_epsilon) < float("inf"):
            raise ValueError("edge_epsilon must be finite and positive")
        self.edge_epsilon = float(edge_epsilon)
        self.zero_rest_edge_fallback_axis = canonical_fallback_axis(
            zero_rest_edge_fallback_axis
        )
        if not isinstance(include_zero_rest_edges, bool):
            raise TypeError("include_zero_rest_edges must be a boolean")
        self.include_zero_rest_edges = include_zero_rest_edges
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.child_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.parent_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.rest_proj = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.direction_proj = nn.Linear(self.hidden_dim, 3)
        self.confidence_proj = nn.Linear(self.hidden_dim, 1)
        # Residual direction parameterization starts exactly at the normalized
        # rest direction.  Zero initialization avoids normalizing a tiny random
        # vector, whose 1/||x|| Jacobian otherwise produces extreme gradients.
        nn.init.zeros_(self.direction_proj.weight)
        nn.init.zeros_(self.direction_proj.bias)
        nn.init.normal_(self.confidence_proj.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.confidence_proj.bias, float(confidence_bias))

    def forward(
        self,
        visual_features: torch.Tensor,
        batch: dict,
        return_reconstructed_positions: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        """Predict bone directions and optionally lift them to joint positions.

        With ``return_reconstructed_positions=False`` only the directions and
        confidences are returned, which is what the GRACE correction consumes;
        the hierarchy FK is skipped, avoiding one small gather/stack sequence
        per joint.
        """
        if not isinstance(return_reconstructed_positions, bool):
            raise TypeError("return_reconstructed_positions must be a bool")
        if visual_features.ndim != 4:
            raise ValueError("visual_features must be [B,T,J,F]")
        batch_size, frames, joints, feature_dim = visual_features.shape
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {feature_dim}"
            )
        parents = batch["parent_a"].long()
        rest_offset = batch["offset_a"].to(dtype=visual_features.dtype)
        joint_mask = batch["joint_mask"].bool()
        # Topology is static for a stream and is validated by the FK helper
        # when reconstruction is requested. Avoid repeating CUDA->Python bool
        # checks on every direction-only inference frame.
        if parents.shape != (batch_size, joints):
            raise ValueError("topology does not match visual_features")
        frame_mask = batch.get("frame_valid_mask")
        if frame_mask is None:
            frame_mask = torch.ones(
                batch_size,
                frames,
                dtype=torch.bool,
                device=visual_features.device,
            )
        else:
            frame_mask = frame_mask.bool()

        safe_parent = parents.clamp(min=0, max=max(joints - 1, 0))
        parent_index = safe_parent[:, None, :, None].expand(
            batch_size, frames, joints, feature_dim
        )
        normalized = self.feature_norm(visual_features)
        parent_feature = torch.gather(normalized, 2, parent_index)

        rest_length = torch.linalg.vector_norm(rest_offset, dim=-1)
        rest_direction = safe_unit_vector(
            rest_offset,
            eps=self.edge_epsilon,
            fallback_axis=self.zero_rest_edge_fallback_axis,
        )
        valid_bone = (
            joint_mask
            & parents.ge(0)
            & parents.lt(joints)
        )
        if not self.include_zero_rest_edges:
            valid_bone = valid_bone & rest_length.gt(self.edge_epsilon)
        valid_weight = valid_bone.to(rest_length.dtype)
        # Normalize each rig independently and exclude root/padding.  Therefore
        # a sample's features cannot change when unrelated rigs share its batch.
        rig_scale = (rest_length * valid_weight).sum(dim=1) / (
            valid_weight.sum(dim=1).clamp_min(1.0)
        )
        rig_scale = rig_scale.clamp_min(1e-8)
        rest_context = torch.cat(
            (
                rest_direction,
                torch.log1p(rest_length / rig_scale.unsqueeze(1)).unsqueeze(-1),
            ),
            dim=-1,
        )
        hidden = (
            self.child_proj(normalized)
            + self.parent_proj(parent_feature)
            + self.rest_proj(rest_context).unsqueeze(1)
        )
        hidden = self.trunk(hidden)
        direction_residual = self.direction_proj(hidden)
        rest_direction_expanded = rest_direction[:, None].expand_as(
            direction_residual
        )
        direction = safe_unit_vector(
            rest_direction_expanded + direction_residual,
            fallback=rest_direction_expanded,
            fallback_axis=self.zero_rest_edge_fallback_axis,
        )
        confidence = torch.sigmoid(self.confidence_proj(hidden).squeeze(-1))

        valid_edge = valid_bone
        valid = frame_mask.unsqueeze(-1) & valid_edge.unsqueeze(1)
        direction = direction * valid.unsqueeze(-1).to(direction.dtype)
        confidence = confidence * valid.to(confidence.dtype)
        reconstructed = None
        if return_reconstructed_positions:
            reconstructed = reconstruct_positions_from_bone_directions(
                direction,
                rest_offset,
                parents,
                joint_mask,
                frame_mask,
                edge_epsilon=self.edge_epsilon,
            )
        result = {
            "pred_bone_direction": direction,
            "pred_bone_confidence": confidence,
            "pred_bone_position": reconstructed,
        }
        if return_reconstructed_positions:
            result["pred_bone_direction_residual"] = direction_residual
            result["pred_bone_valid_mask"] = valid
            result["pred_bone_child_count"] = child_count_from_parents(
                parents, joint_mask
            )
        return result
