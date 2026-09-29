"""GRACE: topology-aware analytic correction of observable rotations."""

from __future__ import annotations

import contextlib
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.base_model.bone_direction_head import (
    canonical_fallback_axis,
    safe_unit_vector,
)
from utils.skeleton_observability import (
    retained_observability_catalog_torch,
    tree_depth_from_parents_torch,
)
from utils.transforms3d import axis_angle_to_matrix, matrix_to_axis_angle

try:
    from ops.fused_depth_geometry import (
        fused_axis_angle_to_matrix,
        fused_confidence_weighted_kabsch,
        fused_depth_geometry,
        fused_fixed_visual_depth,
        fused_matrix_to_axis_angle,
        fused_minimal_axis_angle,
        fused_visual_gate,
    )
except Exception:  # Optional CUDA extension; analytic PyTorch fallback remains available.
    fused_depth_geometry = None
    fused_fixed_visual_depth = None
    fused_axis_angle_to_matrix = None
    fused_confidence_weighted_kabsch = None
    fused_matrix_to_axis_angle = None
    fused_minimal_axis_angle = None
    fused_visual_gate = None


def _fused_cuda_buildable() -> bool:
    """Return whether the optional fused extension can be built here.

    A PyTorch CUDA runtime is sufficient for normal model execution but not
    for JIT compiling a C++/CUDA extension.  In particular, the evaluation
    host may have no CUDA toolkit/nvcc and therefore no ``CUDA_HOME``.  Treat
    that case as an unavailable optimization and use the exact analytic path.
    The ``CAST_FUSED_CUDA=1`` variable is an explicit override for hosts with a
    nonstandard toolkit layout.
    """
    mode = os.environ.get("CAST_FUSED_CUDA", "auto").strip().lower()
    if mode in {"0", "false", "off", "disable", "disabled"}:
        return False
    if mode in {"1", "true", "on", "enable", "enabled"}:
        return True
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        return False
    return bool(
        CUDA_HOME
        and os.path.isfile(os.path.join(CUDA_HOME, "bin", "nvcc"))
    )


if not _fused_cuda_buildable():
    fused_depth_geometry = None
    fused_fixed_visual_depth = None
    fused_axis_angle_to_matrix = None
    fused_confidence_weighted_kabsch = None
    fused_matrix_to_axis_angle = None
    fused_minimal_axis_angle = None
    fused_visual_gate = None


def _fp32_geometry_context(tensor: torch.Tensor):
    """Disable AMP for linalg/SO(3) kernels that require explicit FP32."""

    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return contextlib.nullcontext()


def _determinant_3x3(matrix: torch.Tensor) -> torch.Tensor:
    """Batched differentiable 3x3 determinant without CUDA linalg kernels."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"_determinant_3x3 requires [...,3,3], got {tuple(matrix.shape)}"
        )

    # det(M) = row_0 dot (row_1 cross row_2).  This preserves the exact
    # proper/improper sign semantics needed by Kabsch while avoiding the
    # batched torch.linalg.det CUDA driver path.
    return (
        matrix[..., 0, :]
        * torch.cross(
            matrix[..., 1, :], matrix[..., 2, :], dim=-1
        )
    ).sum(dim=-1)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def minimal_rotation_between_vectors(
    source: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return the shortest SO(3) rotation mapping ``source`` to ``target``.

    Parallel vectors map to identity.  Anti-parallel vectors use a
    deterministic orthogonal axis, avoiding the zero-cross-product singularity.
    """

    with _fp32_geometry_context(source):
        source = safe_unit_vector(source.float(), eps=eps)
        target = safe_unit_vector(target.float(), eps=eps)
        cross = torch.cross(source, target, dim=-1)
        dot = (source * target).sum(dim=-1).clamp(-1.0, 1.0)
        sine_sq = (cross * cross).sum(dim=-1)
        identity = torch.eye(
            3, device=source.device, dtype=torch.float32
        ).expand(source.shape[:-1] + (3, 3))
        cross_matrix = _skew(cross)
        general = (
            identity
            + cross_matrix
            + cross_matrix @ cross_matrix
            * ((1.0 - dot) / sine_sq.clamp_min(eps))
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        fallback_axis = F.one_hot(
            torch.argmin(torch.abs(source), dim=-1), num_classes=3
        ).to(dtype=torch.float32)
        anti_axis = fallback_axis - (
            fallback_axis * source
        ).sum(dim=-1, keepdim=True) * source
        anti_axis = safe_unit_vector(anti_axis, eps=eps)
        anti = (
            2.0 * anti_axis.unsqueeze(-1) @ anti_axis.unsqueeze(-2)
            - identity
        )
        parallel = sine_sq <= eps
        anti_parallel = parallel & dot.lt(0.0)
        result = torch.where(
            parallel.unsqueeze(-1).unsqueeze(-1), identity, general
        )
        result = torch.where(
            anti_parallel.unsqueeze(-1).unsqueeze(-1), anti, result
        )
        return result


def minimal_axis_angle_between_vectors(
    source: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return the shortest source-to-target rotation directly as a rotvec.

    This is the axis-angle counterpart of :func:`minimal_rotation_between_vectors`.
    Keeping the proposal in rotvec form avoids a matrix->axis-angle round trip
    in the streaming single-child path.  Anti-parallel vectors use the same
    deterministic fallback axis as the matrix implementation.
    """
    with _fp32_geometry_context(source):
        source = safe_unit_vector(source.float(), eps=eps)
        target = safe_unit_vector(target.float(), eps=eps)
        cross = torch.cross(source, target, dim=-1)
        dot = (source * target).sum(dim=-1).clamp(-1.0, 1.0)
        sine = torch.linalg.vector_norm(cross, dim=-1)
        angle = torch.atan2(sine, dot)
        axis = cross / sine.clamp_min(eps).unsqueeze(-1)
        fallback_axis = F.one_hot(
            torch.argmin(torch.abs(source), dim=-1), num_classes=3
        ).to(dtype=torch.float32)
        anti_axis = fallback_axis - (
            fallback_axis * source
        ).sum(dim=-1, keepdim=True) * source
        anti_axis = safe_unit_vector(anti_axis, eps=eps)
        anti_parallel = (sine <= eps) & dot.lt(0.0)
        axis = torch.where(anti_parallel.unsqueeze(-1), anti_axis, axis)
        return torch.where(
            (sine <= eps).unsqueeze(-1) & ~anti_parallel.unsqueeze(-1),
            torch.zeros_like(axis),
            axis * angle.unsqueeze(-1),
        )


def confidence_weighted_kabsch(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    eps: float = 1e-6,
    relative_singular_threshold: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve batched Wahba alignment, falling back on one reliable direction.

    Args:
        source/target: ``[..., N, 3]`` unit vectors.
        confidence: ``[..., N]`` nonnegative weights.

    Returns:
        ``(rotation, well_conditioned)`` where rotation maps source to target.
    """

    if source.shape != target.shape or source.shape[-1] != 3:
        raise ValueError("source and target must have matching [...,N,3] shapes")
    if confidence.shape != source.shape[:-1]:
        raise ValueError("confidence must match source without xyz")
    with _fp32_geometry_context(source):
        source = safe_unit_vector(source.float(), eps=eps)
        target = safe_unit_vector(target.float(), eps=eps)
        weight = torch.where(
            torch.isfinite(confidence.float()), confidence.float().clamp_min(0.0),
            torch.zeros_like(confidence.float()),
        )
        raw_covariance = torch.einsum(
            "...n,...ni,...nj->...ij", weight, target, source
        )
        # Degeneracy is a property of the real weighted direction set.  Diagnose
        # it from a detached, unregularized covariance so the numerical SVD
        # stabilizer can never manufacture observability or a second singular value.
        raw_singular = torch.linalg.svdvals(raw_covariance.detach())

        # SVD backward is undefined for repeated singular values.  Stabilize only
        # the matrix used to construct the Kabsch rotation.  The perturbation scales
        # with the true covariance norm; an all-zero covariance receives only an
        # eps-sized absolute floor rather than a unit-scale perturbation.
        identity = torch.eye(
            3, dtype=source.dtype, device=source.device
        ).expand(raw_covariance.shape)
        anisotropic = torch.diag(
            source.new_tensor([1.0, 2.0, 4.0])
        ).expand(raw_covariance.shape)
        covariance_norm = torch.sqrt(
            raw_covariance.detach().square().sum(
                dim=(-2, -1), keepdim=True
            ).clamp_min(0.0)
        )
        regularizer_scale = covariance_norm.clamp_min(eps)
        regularized_covariance = (
            raw_covariance
            + (4.0 * eps) * regularizer_scale * anisotropic
        )
        u, singular, vh = torch.linalg.svd(regularized_covariance)
        uvh = u @ vh
        correction = torch.ones_like(singular)
        correction[..., -1] = _determinant_3x3(uvh)
        kabsch = (u * correction.unsqueeze(-2)) @ vh

        strongest = weight.argmax(dim=-1, keepdim=True)
        gather_index = strongest.unsqueeze(-1).expand(source.shape[:-2] + (1, 3))
        source_primary = torch.gather(source, -2, gather_index).squeeze(-2)
        target_primary = torch.gather(target, -2, gather_index).squeeze(-2)
        fallback = minimal_rotation_between_vectors(source_primary, target_primary, eps)

        count = weight.gt(eps).sum(dim=-1)
        raw_scale = raw_singular[..., 0]
        rank_ok = (
            raw_scale > eps
        ) & (
            raw_singular[..., 1]
            > raw_scale * float(relative_singular_threshold)
        )
        well_conditioned = count.ge(2) & rank_ok & torch.isfinite(kabsch).all(
            dim=-1
        ).all(dim=-1)
        rotation = torch.where(
            well_conditioned.unsqueeze(-1).unsqueeze(-1), kabsch, fallback
        )
        has_any = weight.sum(dim=-1) > eps
        rotation = torch.where(
            has_any.unsqueeze(-1).unsqueeze(-1), rotation, identity
        )
        return rotation, well_conditioned


def _batched_rest_observability_eigenvalues(
    rest_child_direction: torch.Tensor,
    child_valid: torch.Tensor,
) -> torch.Tensor:
    """Return rest observability eigenvalues from compact child slots.

    Args:
        rest_child_direction: ``[B,J,K,3]`` compact outgoing rest bones.
        child_valid: ``[B,J,K]`` validity mask for the compact slots.
    """

    if rest_child_direction.ndim != 4 or rest_child_direction.shape[-1] != 3:
        raise ValueError("rest_child_direction must be [B,J,K,3]")
    if child_valid.shape != rest_child_direction.shape[:-1]:
        raise ValueError("child_valid must match compact child slots")

    with _fp32_geometry_context(rest_child_direction):
        rest_child_direction = safe_unit_vector(
            rest_child_direction.float()
        )
        identity = torch.eye(
            3, dtype=torch.float32, device=rest_child_direction.device
        )
        child_gram = identity - (
            rest_child_direction.unsqueeze(-1)
            @ rest_child_direction.unsqueeze(-2)
        )
        weight = child_valid.to(dtype=torch.float32)
        child_count = weight.sum(dim=-1)
        mean_gram = (
            child_gram * weight.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=-3) / child_count.clamp_min(1.0).unsqueeze(-1).unsqueeze(-1)
        eigenvalues = torch.linalg.eigvalsh(mean_gram).clamp_min(0.0)
        return torch.where(
            child_count.gt(0.0).unsqueeze(-1),
            eigenvalues,
            torch.zeros_like(eigenvalues),
        )


def _batched_child_shape_error(
    target_child_direction: torch.Tensor,
    rest_child_direction: torch.Tensor,
    confidence: torch.Tensor,
    child_valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Measure rotation-invariant disagreement inside each child set.

    A single rigid joint rotation preserves every pairwise dot product between
    outgoing bones. Comparing the predicted and rest-pose Gram matrices thus
    measures whether all predicted child directions can be explained by one
    rotation, without depending on the recursively corrected parent state.

    Args:
        target_child_direction: ``[B,T,J,K,3]`` predicted unit directions.
        rest_child_direction: ``[B,J,K,3]`` rest-pose unit directions.
        confidence: ``[B,T,J,K]`` nonnegative direction confidence.
        child_valid: ``[B,J,K]`` compact child-slot validity.

    Returns:
        ``[B,T,J]`` weighted mean Gram disagreement normalized to ``[0,1]``.
        Zero- and one-child joints return exactly zero.
    """

    if target_child_direction.ndim != 5 or target_child_direction.shape[-1] != 3:
        raise ValueError("target_child_direction must be [B,T,J,K,3]")
    if rest_child_direction.shape != (
        target_child_direction.shape[0],
        target_child_direction.shape[2],
        target_child_direction.shape[3],
        3,
    ):
        raise ValueError("rest_child_direction shape mismatch")
    if confidence.shape != target_child_direction.shape[:-1]:
        raise ValueError("confidence must match target child slots")
    if child_valid.shape != rest_child_direction.shape[:-1]:
        raise ValueError("child_valid must match rest child slots")

    with _fp32_geometry_context(target_child_direction):
        target = safe_unit_vector(target_child_direction.float(), eps=eps)
        rest = safe_unit_vector(rest_child_direction.float(), eps=eps)
        target_gram = torch.einsum("btjkc,btjlc->btjkl", target, target)
        rest_gram = torch.einsum("bjkc,bjlc->bjkl", rest, rest).unsqueeze(1)

        weight = torch.where(
            torch.isfinite(confidence.float()),
            confidence.float().clamp_min(0.0),
            torch.zeros_like(confidence.float()),
        )
        pair_weight = weight.unsqueeze(-1) * weight.unsqueeze(-2)
        pair_valid = child_valid.unsqueeze(-1) & child_valid.unsqueeze(-2)
        compact_width = child_valid.shape[-1]
        upper = torch.ones(
            compact_width,
            compact_width,
            dtype=torch.bool,
            device=child_valid.device,
        ).triu(diagonal=1)
        pair_weight = pair_weight * (
            pair_valid & upper.view(1, 1, compact_width, compact_width)
        ).unsqueeze(1).to(pair_weight.dtype)

        weight_sum = pair_weight.sum(dim=(-2, -1))
        normalized_error = 0.5 * (
            (target_gram - rest_gram).abs() * pair_weight
        ).sum(dim=(-2, -1)) / weight_sum.clamp_min(eps)
        return torch.where(
            weight_sum.gt(eps),
            normalized_error.clamp(0.0, 1.0),
            torch.zeros_like(normalized_error),
        )


def _gather_compact_children(
    values: torch.Tensor,
    child_index: torch.Tensor,
) -> torch.Tensor:
    """Gather ``[B,T,J,...]`` values into fixed ``[B,T,J,K,...]`` slots."""

    if values.ndim < 3:
        raise ValueError("values must start with [B,T,J]")
    batch_size, frames, joints = values.shape[:3]
    if child_index.ndim != 3 or child_index.shape[:2] != (
        batch_size, joints
    ):
        raise ValueError("child_index must be [B,J,K]")
    compact_width = child_index.shape[-1]
    tail_shape = values.shape[3:]
    expanded_values = values.unsqueeze(2).expand(
        batch_size, frames, joints, joints, *tail_shape
    )
    index = child_index.view(
        batch_size, 1, joints, compact_width, *([1] * len(tail_shape))
    ).expand(batch_size, frames, joints, compact_width, *tail_shape)
    return torch.gather(expanded_values, 3, index)


def _level_parallel_global_rotations(
    local_rotation: torch.Tensor,
    parents: torch.Tensor,
    joint_mask: torch.Tensor,
    joint_depth: torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Convert local rotations with logarithmic-depth pointer jumping.

    Each round doubles the number of ancestors already multiplied into every
    joint. A depth-10 skeleton therefore needs four batched rounds instead of ten
    depth passes or eighty joint passes. Matrix order remains exactly root to
    leaf, so this is an exact differentiable tree scan rather than an
    approximation.
    """

    if local_rotation.ndim != 5 or local_rotation.shape[-2:] != (3, 3):
        raise ValueError("local_rotation must be [B,T,J,3,3]")
    batch_size, frames, joints = local_rotation.shape[:3]
    if parents.shape != (batch_size, joints):
        raise ValueError("parents shape mismatch")
    if joint_mask.shape != (batch_size, joints):
        raise ValueError("joint_mask shape mismatch")
    if joint_depth.shape != (batch_size, joints):
        raise ValueError("joint_depth shape mismatch")

    joint_index = torch.arange(
        joints, dtype=parents.dtype, device=parents.device
    ).view(1, joints)
    valid_parent = (
        joint_mask
        & joint_index.gt(0)
        & parents.ge(0)
        & parents.lt(joint_index)
    )
    ancestor = torch.where(
        valid_parent, parents, torch.full_like(parents, -1)
    )
    global_rotation = local_rotation
    for _ in range(max_depth.bit_length()):
        has_ancestor = ancestor.ge(0)
        safe_ancestor = ancestor.clamp(min=0, max=max(joints - 1, 0))
        rotation_index = safe_ancestor.view(
            batch_size, 1, joints, 1, 1
        ).expand(batch_size, frames, joints, 3, 3)
        ancestor_rotation = torch.gather(
            global_rotation, 2, rotation_index
        )
        combined = ancestor_rotation @ global_rotation
        global_rotation = torch.where(
            has_ancestor.unsqueeze(1).unsqueeze(-1).unsqueeze(-1),
            combined,
            global_rotation,
        )
        ancestor_parent = torch.gather(
            ancestor, 1, safe_ancestor
        )
        ancestor = torch.where(
            has_ancestor,
            ancestor_parent,
            torch.full_like(ancestor, -1),
        )
    return global_rotation


class GRACE(nn.Module):
    """GRACE correction: adjust observable global swing while preserving leaves."""

    def __init__(
        self,
        feature_dim: int = 1152,
        gate_mode: str = "confidence",
        gate_hidden_dim: int = 256,
        visual_gate_bias: float = -4.0,
        fixed_gate_scale: float = 1.0,
        max_child_count: int = 4,
        compact_child_slots: int = 16,
        max_tree_depth: int = 12,
        detach_confidence_gate: bool = False,
        detach_direction_gate_features: bool = False,
        exclude_zero_rest_edges: bool = False,
        zero_rest_edge_fallback_axis: str = "x",
        visual_gate_chunk_tokens: int | None = None,
        visual_gate_checkpoint_chunks: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        gate_mode = str(gate_mode).lower()
        if gate_mode not in {"confidence", "geometry", "visual"}:
            raise ValueError(
                "gate_mode must be 'confidence', 'geometry', or 'visual'"
            )
        if max_child_count <= 0:
            raise ValueError("max_child_count must be positive")
        if compact_child_slots <= 0:
            raise ValueError("compact_child_slots must be positive")
        if max_tree_depth <= 0:
            raise ValueError("max_tree_depth must be positive")
        if not 0.0 <= float(fixed_gate_scale) <= 1.0:
            raise ValueError("fixed_gate_scale must be in [0, 1]")
        self.feature_dim = int(feature_dim)
        self.gate_mode = gate_mode
        self.max_child_count = int(max_child_count)
        self.compact_child_slots = int(compact_child_slots)
        self.max_tree_depth = int(max_tree_depth)
        self.fixed_gate_scale = float(fixed_gate_scale)
        self.detach_confidence_gate = bool(detach_confidence_gate)
        self.detach_direction_gate_features = bool(
            detach_direction_gate_features
        )
        self.exclude_zero_rest_edges = bool(exclude_zero_rest_edges)
        self.zero_rest_edge_fallback_axis = canonical_fallback_axis(
            zero_rest_edge_fallback_axis
        )
        if visual_gate_chunk_tokens is None:
            chunk_tokens = 0
        elif isinstance(visual_gate_chunk_tokens, bool):
            raise TypeError("visual_gate_chunk_tokens must be an integer or None")
        else:
            chunk_tokens = int(visual_gate_chunk_tokens)
            if chunk_tokens < 0:
                raise ValueError("visual_gate_chunk_tokens must be a non-negative integer or None")
        self.visual_gate_chunk_tokens = (
            None if chunk_tokens == 0 else chunk_tokens
        )
        if not isinstance(visual_gate_checkpoint_chunks, bool):
            raise TypeError(
                "visual_gate_checkpoint_chunks must be a boolean"
            )
        self.visual_gate_checkpoint_chunks = visual_gate_checkpoint_chunks
        self.eps = float(eps)
        self.gate_override = "learned"
        # Static skeleton metadata is independent of video frames and model
        # predictions.  Keep one device-local entry per rig so training does
        # not rebuild child routing/eigenvalue/depth data every step.
        self._topology_cache: dict[tuple[object, ...], dict[str, object]] = {}
        if gate_mode in {"geometry", "visual"}:
            if gate_hidden_dim <= 0 or (gate_mode == "visual" and feature_dim <= 0):
                raise ValueError("learned gate dimensions must be positive")
            self.visual_norm = (
                nn.LayerNorm(self.feature_dim) if gate_mode == "visual" else None
            )
            gate_input_dim = self.feature_dim + 6 if gate_mode == "visual" else 6
            self.visual_gate = nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            nn.init.normal_(self.visual_gate[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.visual_gate[-1].bias, float(visual_gate_bias))
        else:
            self.visual_norm = None
            self.visual_gate = None

    def _run_visual_gate_mlp(self, gate_input: torch.Tensor) -> torch.Tensor:
        """Run one bounded VisualGate token block with optional recomputation."""

        if self.training and self.visual_gate_checkpoint_chunks:
            return checkpoint(
                self.visual_gate,
                gate_input,
                use_reentrant=False,
            )
        return self.visual_gate(gate_input)

    def _run_visual_gate_chunked(
        self, gate_input: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate one already-concatenated gate tensor in token chunks."""

        if self.visual_gate_chunk_tokens is None:
            return self.visual_gate(gate_input)
        flattened = gate_input.reshape(-1, gate_input.shape[-1])
        chunk_size = int(self.visual_gate_chunk_tokens)
        chunks = [
            self._run_visual_gate_mlp(flattened[start : start + chunk_size])
            for start in range(0, flattened.shape[0], chunk_size)
        ]
        logits = torch.cat(chunks, dim=0)
        return logits.reshape(gate_input.shape[:-1] + (logits.shape[-1],))

    def _run_visual_gate_feature_parts(
        self,
        feature_parts: list[torch.Tensor],
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run VisualGate only for selected tokens and stream concatenation."""

        if not feature_parts:
            raise ValueError("VisualGate feature_parts must be non-empty")
        leading_shape = feature_parts[0].shape[:-1]
        for index, part in enumerate(feature_parts[1:], start=1):
            if part.shape[:-1] != leading_shape:
                raise ValueError(
                    "VisualGate feature parts must share leading dimensions; "
                    f"part 0 has {leading_shape}, part {index} has {part.shape[:-1]}"
                )
        if token_mask is not None:
            if token_mask.shape != leading_shape:
                raise ValueError(
                    "VisualGate token_mask must match feature leading shape; "
                    f"got {token_mask.shape} versus {leading_shape}"
                )
            flat_mask = token_mask.reshape(-1).bool()
        else:
            flat_mask = None

        flattened_parts = [
            part.reshape(-1, part.shape[-1]) for part in feature_parts
        ]
        token_count = flattened_parts[0].shape[0]
        if flat_mask is not None:
            # Compact before feature concatenation and before every MLP Linear.
            # In the depth recurrence only current, observable joints can affect
            # the correction, so padding joints, invalid frames, and joints at
            # other depths must not consume VisualGate compute or activations.
            flattened_parts = [part[flat_mask] for part in flattened_parts]

        active_count = flattened_parts[0].shape[0]
        if active_count == 0:
            output_dim = self.visual_gate[-1].out_features
            return feature_parts[0].new_zeros(leading_shape + (output_dim,))

        if self.visual_gate_chunk_tokens is None:
            active_logits = self.visual_gate(
                torch.cat(flattened_parts, dim=-1)
            )
        else:
            chunk_size = int(self.visual_gate_chunk_tokens)
            chunks = []
            for start in range(0, active_count, chunk_size):
                end = min(start + chunk_size, active_count)
                gate_chunk = torch.cat(
                    [part[start:end] for part in flattened_parts], dim=-1
                )
                chunks.append(self._run_visual_gate_mlp(gate_chunk))
            active_logits = torch.cat(chunks, dim=0)

        if flat_mask is None:
            return active_logits.reshape(
                leading_shape + (active_logits.shape[-1],)
            )
        logits = active_logits.new_zeros(token_count, active_logits.shape[-1])
        logits[flat_mask] = active_logits
        return logits.reshape(leading_shape + (active_logits.shape[-1],))

    def set_gate_override(self, mode: str) -> None:
        """Override correction strength for controlled checkpoint evaluation."""

        mode = str(mode).lower()
        allowed = {"learned", "confidence", "full_on", "full_off"}
        if mode not in allowed:
            raise ValueError(f"gate override must be one of {sorted(allowed)}")
        self.gate_override = mode

    def _build_topology_metadata(
        self,
        parents: torch.Tensor,
        joint_mask: torch.Tensor,
        rest_offset: torch.Tensor,
        device: torch.device,
    ) -> dict[str, object]:
        """Build compact valid-child slots and vectorized tree metadata."""

        catalog = retained_observability_catalog_torch(
            parents,
            rest_offset,
            joint_mask,
            direction_epsilon=self.eps,
        )
        visual_gate = self.gate_mode == "visual"
        if self.exclude_zero_rest_edges and not visual_gate:
            valid_edge = catalog["valid_edge_mask"]
        else:
            # Keep zero-length child slots: they cannot add observability
            # rank, but dropping them would change per-joint confidence and
            # child counts.
            valid_edge = (
                joint_mask
                & parents.ge(0)
                & parents.lt(parents.shape[1])
            )
        batch_size, joints = parents.shape
        owner_ids = torch.arange(joints, device=device).view(1, joints, 1)
        child_mask = valid_edge.unsqueeze(1) & parents.unsqueeze(1).eq(owner_ids)
        child_count = child_mask.sum(dim=-1)
        compact_width = (
            max(
                1,
                int(child_count.max().item()) if child_count.numel() else 0,
            )
            if visual_gate
            else min(max(1, joints), self.compact_child_slots)
        )
        if not visual_gate:
            torch._assert_async(
                child_count.le(compact_width).all(),
                "retained topology exceeds compact_child_slots; increase the "
                "configured slot capacity instead of truncating child directions",
            )
        child_slot = (child_mask.long().cumsum(dim=-1) - 1).clamp(
            min=0, max=compact_width - 1
        )
        child_ids = torch.arange(joints, device=device).view(1, 1, joints)
        child_index = torch.zeros(
            batch_size, joints, compact_width, dtype=torch.long, device=device
        )
        child_index.scatter_add_(
            2,
            child_slot,
            child_ids.expand(batch_size, joints, joints) * child_mask.long(),
        )
        child_valid = (
            torch.arange(compact_width, device=device).view(1, 1, -1)
            < child_count.unsqueeze(-1)
        )

        joint_depth, _ = tree_depth_from_parents_torch(
            parents, joint_mask
        )
        if visual_gate:
            max_depth = (
                int(joint_depth.max().item()) if joint_depth.numel() else 0
            )
        else:
            torch._assert_async(
                joint_depth.le(self.max_tree_depth).all(),
                "retained topology exceeds max_tree_depth; increase the explicit "
                "depth capacity rather than falling back to a per-joint loop",
            )
            max_depth = self.max_tree_depth
        depth_indices = tuple(
            torch.nonzero(
                (joint_mask & joint_depth.eq(depth)).reshape(-1),
                as_tuple=False,
            ).squeeze(-1)
            for depth in range(max_depth + 1)
        )
        depth_single_indices = tuple(
            torch.nonzero(
                child_count.reshape(-1)[depth_indices[depth]].eq(1),
                as_tuple=False,
            ).squeeze(-1)
            for depth in range(max_depth + 1)
        )
        depth_multi_indices = tuple(
            torch.nonzero(
                child_count.reshape(-1)[depth_indices[depth]].gt(1),
                as_tuple=False,
            ).squeeze(-1)
            for depth in range(max_depth + 1)
        )
        depth_conversion_indices = tuple(
            torch.nonzero(
                child_count.reshape(-1)[depth_indices[depth]].gt(0),
                as_tuple=False,
            ).squeeze(-1)
            for depth in range(max_depth + 1)
        )
        return {
            "child_index": child_index,
            "child_valid": child_valid,
            "child_count": child_count,
            "joint_depth": joint_depth,
            "max_depth": max_depth,
            "catalog": catalog,
            "depth_indices": depth_indices,
            "depth_single_indices": depth_single_indices,
            "depth_multi_indices": depth_multi_indices,
            "depth_conversion_indices": depth_conversion_indices,
        }

    @staticmethod
    def _pad_metadata_tensor(
        values: list[torch.Tensor], *, dim: int, size: int
    ) -> torch.Tensor:
        padded = []
        for value in values:
            if value.shape[dim] < size:
                pad_shape = list(value.shape)
                pad_shape[dim] = size - value.shape[dim]
                pad = torch.zeros(
                    pad_shape, dtype=value.dtype, device=value.device
                )
                value = torch.cat((value, pad), dim=dim)
            padded.append(value)
        return torch.cat(padded, dim=0)

    def _topology_metadata(
        self,
        parents: torch.Tensor,
        joint_mask: torch.Tensor,
        rest_offset: torch.Tensor,
        device: torch.device,
        species: object = None,
        rel: object = None,
    ) -> dict[str, object]:
        """Return cached static topology data, merging per-rig entries safely."""

        batch_size, joints = parents.shape
        cache_identity = rel if isinstance(rel, (list, tuple)) else species
        # MObjaverse uses category names for ``species`` while assets in one
        # category can have different skeletons.  ``rel`` is asset/clip
        # qualified, so prefer it whenever the collator provides it.
        if not isinstance(cache_identity, (list, tuple)) or len(cache_identity) != batch_size:
            # A streaming batch is normally B=1 and keeps the same static
            # tensors for the lifetime of the stream. Cache that entry by
            # storage identity even when the caller has no rel/species tag.
            if batch_size == 1:
                key = (
                    "tensor",
                    int(parents.data_ptr()),
                    int(joint_mask.data_ptr()),
                    int(rest_offset.data_ptr()),
                    joints,
                    device.type,
                    device.index,
                    self.gate_mode,
                    self.exclude_zero_rest_edges,
                    self.compact_child_slots,
                    self.max_tree_depth,
                    self.eps,
                )
                entry = self._topology_cache.get(key)
                if entry is None:
                    entry = self._build_topology_metadata(
                        parents, joint_mask, rest_offset, device
                    )
                    self._topology_cache[key] = entry
                return entry
            return self._build_topology_metadata(
                parents, joint_mask, rest_offset, device
            )

        entries: list[dict[str, object]] = []
        for index, name in enumerate(cache_identity):
            key = (
                str(name),
                joints,
                device.type,
                device.index,
                self.gate_mode,
                self.exclude_zero_rest_edges,
                self.compact_child_slots,
                self.max_tree_depth,
                self.eps,
            )
            entry = self._topology_cache.get(key)
            if entry is None:
                entry = self._build_topology_metadata(
                    parents[index : index + 1],
                    joint_mask[index : index + 1],
                    rest_offset[index : index + 1],
                    device,
                )
                self._topology_cache[key] = entry
            entries.append(entry)

        # No batch merge/padding is needed for the common B=1 streaming case.
        if batch_size == 1:
            return entries[0]

        max_width = max(
            int(entry["child_index"].shape[-1])
            for entry in entries
        )
        child_index = self._pad_metadata_tensor(
            [entry["child_index"] for entry in entries], dim=2, size=max_width
        )
        child_valid = self._pad_metadata_tensor(
            [entry["child_valid"] for entry in entries], dim=2, size=max_width
        )
        child_count = torch.cat(
            [entry["child_count"] for entry in entries], dim=0
        )
        joint_depth = torch.cat(
            [entry["joint_depth"] for entry in entries], dim=0
        )
        catalog_keys = tuple(entries[0]["catalog"].keys())
        catalog = {
            key: torch.cat(
                [entry["catalog"][key] for entry in entries], dim=0
            )
            for key in catalog_keys
        }
        max_depth = max(int(entry["max_depth"]) for entry in entries)
        depth_indices = []
        depth_single_indices = []
        depth_multi_indices = []
        depth_conversion_indices = []
        for depth in range(max_depth + 1):
            parts = []
            single_parts = []
            multi_parts = []
            conversion_parts = []
            offset = 0
            for index, entry in enumerate(entries):
                local = entry["depth_indices"]
                if depth < len(local) and local[depth].numel():
                    parts.append(local[depth] + index * joints)
                    single_parts.append(entry["depth_single_indices"][depth] + offset)
                    multi_parts.append(entry["depth_multi_indices"][depth] + offset)
                    conversion_parts.append(
                        entry["depth_conversion_indices"][depth] + offset
                    )
                    offset += local[depth].numel()
            depth_indices.append(
                torch.cat(parts, dim=0)
                if parts
                else torch.empty(0, dtype=torch.long, device=device)
            )
            depth_single_indices.append(
                torch.cat(single_parts, dim=0)
                if single_parts
                else torch.empty(0, dtype=torch.long, device=device)
            )
            depth_multi_indices.append(
                torch.cat(multi_parts, dim=0)
                if multi_parts
                else torch.empty(0, dtype=torch.long, device=device)
            )
            depth_conversion_indices.append(
                torch.cat(conversion_parts, dim=0)
                if conversion_parts
                else torch.empty(0, dtype=torch.long, device=device)
            )
        return {
            "child_index": child_index,
            "child_valid": child_valid,
            "child_count": child_count,
            "joint_depth": joint_depth,
            "max_depth": max_depth,
            "catalog": catalog,
            "depth_indices": tuple(depth_indices),
            "depth_single_indices": tuple(depth_single_indices),
            "depth_multi_indices": tuple(depth_multi_indices),
            "depth_conversion_indices": tuple(depth_conversion_indices),
        }

    @torch.no_grad()
    def build_streaming_topology_cache(
        self, batch: dict, *, device: torch.device | None = None
    ) -> dict[str, object]:
        """Build all skeleton-only correction metadata once for a stream.

        Streaming inference uses one fixed rig.  This cache deliberately owns
        the tensors derived only from parents, masks, and rest offsets so the
        per-frame correction path never has to normalize or compact rest
        geometry again.
        """
        if device is None:
            for key in ("parent_a", "joint_mask", "offset_a"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break
            else:
                device = torch.device("cpu")
        parents = batch["parent_a"].to(device=device, dtype=torch.long)
        joint_mask = batch["joint_mask"].to(device=device, dtype=torch.bool)
        rest_offset = batch["offset_a"].to(device=device, dtype=torch.float32)
        static_rot = batch.get("static_rot_joint_mask")
        if static_rot is None:
            static_rot = torch.zeros_like(joint_mask)
        else:
            static_rot = static_rot.to(device=device, dtype=torch.bool)
        correction_joint_mask = (
            joint_mask
            if self.gate_mode == "visual"
            else joint_mask & ~static_rot
        )
        topology = self._topology_metadata(
            parents, joint_mask, rest_offset, device,
            species=batch.get("species"), rel=batch.get("rel"),
        )
        child_index = topology["child_index"]
        compact_width = child_index.shape[-1]
        rest_direction = safe_unit_vector(
            rest_offset,
            eps=self.eps,
            fallback_axis=self.zero_rest_edge_fallback_axis,
        )
        compact_rest_direction = torch.gather(
            rest_direction.unsqueeze(1).expand(
                parents.shape[0], parents.shape[1], parents.shape[1], 3
            ),
            2,
            child_index.unsqueeze(-1).expand(
                parents.shape[0], parents.shape[1], compact_width, 3
            ),
        )
        if self.gate_mode == "visual":
            topology_eigen = _batched_rest_observability_eigenvalues(
                compact_rest_direction, topology["child_valid"]
            )
            topology_eigen = topology_eigen * joint_mask.unsqueeze(-1).to(
                topology_eigen.dtype
            )
            gate_child_count = topology["child_count"]
        else:
            topology_eigen = topology["catalog"]["eigenvalues"].float()
            gate_child_count = topology["catalog"]["valid_direction_count"]
        safe_parent = parents.clamp(min=0, max=max(parents.shape[1] - 1, 0))
        joint_index = torch.arange(
            parents.shape[1], device=device, dtype=parents.dtype
        ).view(1, -1)
        if self.gate_mode == "visual":
            root_mask = joint_index.eq(0).expand_as(joint_mask) & joint_mask
            valid_parent = (
                joint_mask & joint_index.gt(0)
                & parents.ge(0) & parents.lt(joint_index)
            )
        else:
            root_mask = parents.lt(0) & joint_mask
            valid_parent = (
                joint_mask & ~root_mask & parents.ge(0)
                & parents.lt(parents.shape[1])
            )
        depth_batch_indices = []
        depth_joint_indices = []
        depth_parent_flat = []
        for depth_flat in topology["depth_indices"]:
            if depth_flat.numel() == 0:
                depth_batch_indices.append(depth_flat)
                depth_joint_indices.append(depth_flat)
                depth_parent_flat.append(depth_flat)
                continue
            depth_batch = torch.div(depth_flat, parents.shape[1], rounding_mode="floor")
            depth_joint = depth_flat.remainder(parents.shape[1])
            depth_batch_indices.append(depth_batch)
            depth_joint_indices.append(depth_joint)
            depth_parent_flat.append(
                depth_batch * parents.shape[1] + safe_parent[depth_batch, depth_joint]
            )
        return {
            "parents": parents,
            "joint_mask": joint_mask,
            "rest_offset": rest_offset,
            "correction_joint_mask": correction_joint_mask,
            "topology": topology,
            "compact_rest_direction": compact_rest_direction,
            "topology_eigen": topology_eigen,
            "gate_child_count": gate_child_count,
            "safe_parent": safe_parent,
            "root_mask": root_mask,
            "valid_parent": valid_parent,
            "depth_batch_indices": tuple(depth_batch_indices),
            "depth_joint_indices": tuple(depth_joint_indices),
            "depth_parent_flat": tuple(depth_parent_flat),
        }

    def _forward_batched(
        self,
        *,
        baseline: torch.Tensor,
        baseline_input: torch.Tensor,
        compact_target: torch.Tensor,
        compact_target_has_direction: torch.Tensor,
        compact_confidence: torch.Tensor,
        compact_rest_direction: torch.Tensor,
        child_valid: torch.Tensor,
        child_count: torch.Tensor,
        catalog: dict[str, torch.Tensor],
        joint_mask: torch.Tensor,
        correction_joint_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        parents: torch.Tensor,
        joint_depth: torch.Tensor,
        max_depth: int,
        depth_indices: tuple[torch.Tensor, ...],
        depth_single_indices: tuple[torch.Tensor, ...],
        depth_multi_indices: tuple[torch.Tensor, ...],
        depth_conversion_indices: tuple[torch.Tensor, ...],
        aggregate_confidence: torch.Tensor,
        visual_features: torch.Tensor | None,
        return_diagnostics: bool = True,
        static_cache: Optional[dict[str, object]] = None,
    ) -> dict[str, torch.Tensor]:
        """Apply rank-aware proposals with depth-parallel tree recurrence.

        Parent-corrected global rotations are a real sequential dependency, so
        the only Python loop is over skeleton depth. All joints at one depth and
        every batch/frame/child slot are processed in a single tensor program.
        """

        batch_size, frames, joints = baseline.shape[:3]
        visual_gate = self.gate_mode == "visual"
        rank2 = catalog["rank2_mask"]
        rank3 = catalog["rank3_mask"]
        if visual_gate:
            eligible_joint = correction_joint_mask & child_count.gt(0)
        else:
            # Static rotation owners are an output contract, not merely a loss
            # mask. They must not leak changed global state into descendants.
            eligible_joint = correction_joint_mask & (rank2 | rank3)
        identity_matrix = torch.eye(
            3, dtype=torch.float32, device=baseline.device
        )
        flat_joint_count = batch_size * joints
        baseline_flat = baseline.permute(0, 2, 1, 3, 4).reshape(
            flat_joint_count, frames, 3, 3
        )
        corrected_global_flat = (
            baseline_flat.clone()
            if visual_gate
            else identity_matrix.view(1, 1, 3, 3).expand(
                flat_joint_count, frames, 3, 3
            ).clone()
        )
        gate_all_flat = baseline.new_zeros(flat_joint_count, frames)
        proposed_all_flat = gate_all_flat.clone() if return_diagnostics else None
        applied_all_flat = gate_all_flat.clone() if return_diagnostics else None
        observable_all_flat = torch.zeros_like(
            gate_all_flat, dtype=torch.bool
        )
        proposal_valid_all_flat = (
            torch.zeros_like(gate_all_flat, dtype=torch.bool)
            if return_diagnostics
            else None
        )

        safe_parent = (
            static_cache["safe_parent"]
            if static_cache is not None
            else parents.clamp(min=0, max=max(joints - 1, 0))
        )
        parent_index = safe_parent.view(batch_size, 1, joints, 1, 1).expand(
            batch_size, frames, joints, 3, 3
        )
        if static_cache is not None:
            root_mask = static_cache["root_mask"]
            valid_parent = static_cache["valid_parent"]
            topology_eigen = static_cache["topology_eigen"]
            gate_child_count = static_cache["gate_child_count"]
        else:
            joint_index = torch.arange(
                joints, device=parents.device, dtype=parents.dtype
            ).view(1, joints)
            if visual_gate:
                root_mask = joint_index.eq(0).expand(batch_size, joints) & joint_mask
                valid_parent = (
                    joint_mask & joint_index.gt(0) & parents.ge(0)
                    & parents.lt(joint_index)
                )
                topology_eigen = _batched_rest_observability_eigenvalues(
                    compact_rest_direction, child_valid
                )
                topology_eigen = topology_eigen * joint_mask.unsqueeze(-1).to(
                    topology_eigen.dtype
                )
                gate_child_count = child_count
            else:
                root_mask = parents.lt(0) & joint_mask
                valid_parent = (
                    joint_mask & ~root_mask & parents.ge(0) & parents.lt(joints)
                )
                topology_eigen = catalog["eigenvalues"].float()
                gate_child_count = catalog["valid_direction_count"]
        normalized_count = (
            gate_child_count.clamp(max=self.max_child_count).float()
            / float(self.max_child_count)
        ).unsqueeze(1).unsqueeze(-1).expand(batch_size, frames, joints, 1)
        normalized_visual = None
        visual_gate_static_hidden = None
        if self.gate_mode == "visual":
            assert self.visual_norm is not None and visual_features is not None
            normalized_visual = self.visual_norm(visual_features.float())
            # In inference the first gate projection only depends on the
            # visual/topology features for a given frame.  Precompute that
            # static contribution once; confidence and disagreement are the
            # only depth-dependent inputs left for the recurrent loop.
            if (
                not self.training
                and (
                    self.visual_gate_chunk_tokens is None
                    or self.visual_gate_chunk_tokens >= frames * joints
                )
            ):
                first = self.visual_gate[0]
                visual_gate_static_hidden = F.linear(
                    normalized_visual,
                    first.weight[:, : self.feature_dim],
                    first.bias,
                )
                visual_gate_static_hidden = visual_gate_static_hidden + F.linear(
                    normalized_count,
                    first.weight[:, self.feature_dim + 2 : self.feature_dim + 3],
                )
                visual_gate_static_hidden = visual_gate_static_hidden + F.linear(
                    topology_eigen.unsqueeze(1).expand(
                        batch_size, frames, joints, 3
                    ),
                    first.weight[:, self.feature_dim + 3 : self.feature_dim + 6],
                )

        confidence_input = (
            aggregate_confidence.detach()
            if self.detach_confidence_gate
            else aggregate_confidence
        )
        confidence_gate = confidence_input * self.fixed_gate_scale
        learned_gate_once = torch.ones_like(aggregate_confidence)
        learned_gate_input_once = None
        if self.visual_gate is not None and self.gate_mode == "geometry":
            shape_confidence = compact_confidence * (
                compact_target_has_direction.to(compact_confidence.dtype)
            )
            shape_error = _batched_child_shape_error(
                compact_target,
                compact_rest_direction,
                shape_confidence,
                catalog["valid_direction_count"].unsqueeze(-1).gt(
                    torch.arange(
                        compact_target.shape[-2], device=baseline.device
                    ).view(1, 1, -1)
                ),
                self.eps,
            )
            confidence_feature = aggregate_confidence
            disagreement_feature = shape_error
            if self.detach_direction_gate_features:
                confidence_feature = confidence_feature.detach()
                disagreement_feature = disagreement_feature.detach()
            learned_gate_input_once = torch.cat(
                (
                    confidence_feature.unsqueeze(-1),
                    disagreement_feature.unsqueeze(-1),
                    normalized_count,
                    topology_eigen.unsqueeze(1).expand(
                        batch_size, frames, joints, 3
                    ),
                ),
                dim=-1,
            )
            learned_gate_once = torch.sigmoid(
                self.visual_gate(learned_gate_input_once).squeeze(-1)
            )

        # Streaming fast path for one fixed visual-gated topology.  Each
        # launch owns a complete skeleton depth: parent recurrence, direction
        # proposal, learned gate, SO(3) blend, and local/global writeback all
        # execute inside one CUDA kernel.  The only remaining host loop is the
        # mathematically required ordering between tree depths.
        use_fixed_visual_depth = (
            fused_fixed_visual_depth is not None
            and not self.training
            and not return_diagnostics
            and baseline.is_cuda
            and static_cache is not None
            and visual_gate
            and self.visual_gate is not None
            and visual_gate_static_hidden is not None
            and compact_rest_direction.shape[-2] <= 16
            and visual_gate_static_hidden.shape[-1] <= 256
        )
        if use_fixed_visual_depth:
            corrected_local_flat = baseline_flat.clone()
            gate_override = {
                "learned": 0,
                "confidence": 1,
                "full_on": 2,
                "full_off": 3,
            }[self.gate_override]
            first = self.visual_gate[0]
            for depth in range(max_depth + 1):
                depth_flat = depth_indices[depth]
                if depth_flat.numel() == 0:
                    continue
                fused_fixed_visual_depth(
                    baseline_flat,
                    compact_rest_direction,
                    compact_target,
                    compact_target_has_direction,
                    compact_confidence,
                    aggregate_confidence,
                    visual_gate_static_hidden,
                    first.weight[:, self.feature_dim : self.feature_dim + 2],
                    self.visual_gate[2].weight,
                    self.visual_gate[2].bias,
                    self.visual_gate[4].weight,
                    self.visual_gate[4].bias,
                    depth_flat,
                    static_cache["depth_parent_flat"][depth],
                    child_count,
                    root_mask,
                    valid_parent,
                    eligible_joint,
                    frame_mask,
                    corrected_global_flat,
                    corrected_local_flat,
                    gate_all_flat,
                    observable_all_flat,
                    self.fixed_gate_scale,
                    gate_override,
                    self.eps,
                )
            corrected_local = corrected_local_flat.reshape(
                batch_size, joints, frames, 3, 3
            ).permute(0, 2, 1, 3, 4)
            corrected_global = corrected_global_flat.reshape(
                batch_size, joints, frames, 3, 3
            ).permute(0, 2, 1, 3, 4)
            return {
                "corrected_local_rotation": corrected_local.to(baseline_input.dtype),
                "corrected_global_rotation": corrected_global.to(baseline_input.dtype),
            }

        for depth in range(max_depth + 1):
            depth_flat = depth_indices[depth]
            if depth_flat.numel() == 0:
                continue
            if static_cache is not None:
                depth_index = depth
                depth_batch = static_cache["depth_batch_indices"][depth_index]
                depth_joint = static_cache["depth_joint_indices"][depth_index]
                parent_flat = static_cache["depth_parent_flat"][depth_index]
            else:
                depth_batch = torch.div(
                    depth_flat, joints, rounding_mode="floor"
                )
                depth_joint = depth_flat.remainder(joints)
                depth_parent = safe_parent[depth_batch, depth_joint]
                parent_flat = depth_batch * joints + depth_parent

            baseline_depth = baseline_flat[depth_flat]
            parent_global_depth = corrected_global_flat[parent_flat]
            provisional_depth = parent_global_depth @ baseline_depth
            root_depth = root_mask[depth_batch, depth_joint]
            provisional_depth = torch.where(
                root_depth[:, None, None, None],
                baseline_depth,
                provisional_depth,
            )
            if visual_gate:
                valid_owner_depth = (
                    root_depth | valid_parent[depth_batch, depth_joint]
                )
                provisional_depth = torch.where(
                    valid_owner_depth[:, None, None, None],
                    provisional_depth,
                    baseline_depth,
                )

            # Only joints at this tree depth enter the direction solver. The
            # original dense path recomputed B*T*J*K geometry at every depth,
            # even though all other joints were immediately discarded by the
            # depth mask. Compacting to the current [batch,joint] rows keeps
            # the required parent-to-child recurrence while removing that
            # redundant padded/depth-mismatched work.
            rest_direction_depth = compact_rest_direction[
                depth_batch, depth_joint
            ]
            target_has_direction_depth = compact_target_has_direction[
                depth_batch, :, depth_joint
            ]
            confidence_depth = compact_confidence[
                depth_batch, :, depth_joint
            ]
            target_compact_depth = compact_target[depth_batch, :, depth_joint]
            if (
                fused_depth_geometry is not None
                and baseline.is_cuda
                and not self.training
            ):
                source_depth, target_depth, disagreement_depth = fused_depth_geometry(
                    provisional_depth,
                    rest_direction_depth,
                    target_compact_depth,
                    target_has_direction_depth,
                    confidence_depth,
                    self.eps,
                )
            else:
                source_depth = safe_unit_vector(
                    torch.einsum(
                        "ntxy,nky->ntkx",
                        provisional_depth,
                        rest_direction_depth,
                    ),
                    eps=self.eps,
                )
                target_depth = torch.where(
                    target_has_direction_depth.unsqueeze(-1),
                    target_compact_depth,
                    source_depth,
                )
                disagreement_depth = None
            eligible_depth = eligible_joint[depth_batch, depth_joint]
            identity_depth = identity_matrix.view(1, 1, 3, 3).expand(
                depth_flat.numel(), frames, 3, 3
            )

            proposal_axis_angle_depth = torch.zeros(
                depth_flat.numel(), frames, 3,
                dtype=torch.float32, device=baseline.device
            )
            if visual_gate:
                single_rows = depth_single_indices[depth]
                multi_rows = depth_multi_indices[depth]
                correction_depth = identity_depth.clone()
                well_conditioned = torch.zeros(
                    depth_flat.numel(), frames,
                    dtype=torch.bool, device=baseline.device
                )
                if single_rows.numel():
                    single_source = source_depth.index_select(
                        0, single_rows
                    )[..., 0, :]
                    single_target = target_depth.index_select(
                        0, single_rows
                    )[..., 0, :]
                    single_axis_angle = (
                        fused_minimal_axis_angle(
                            single_source, single_target, self.eps
                        )
                        if (
                            fused_minimal_axis_angle is not None
                            and single_source.is_cuda
                            and not self.training
                        )
                        else minimal_axis_angle_between_vectors(
                            single_source, single_target, self.eps
                        )
                    )
                    proposal_axis_angle_depth = proposal_axis_angle_depth.index_copy(
                        0, single_rows,
                        single_axis_angle,
                    )
                    single_valid = confidence_depth.index_select(
                        0, single_rows
                    )[..., 0].gt(self.eps)
                    well_conditioned = well_conditioned.index_copy(
                        0, single_rows, single_valid
                    )
                if multi_rows.numel():
                    multi_source = source_depth.index_select(0, multi_rows)
                    multi_target = target_depth.index_select(0, multi_rows)
                    multi_confidence = confidence_depth.index_select(0, multi_rows)
                    if (
                        fused_confidence_weighted_kabsch is not None
                        and multi_source.is_cuda
                        and not self.training
                    ):
                        multi_correction, multi_valid = (
                            fused_confidence_weighted_kabsch(
                                multi_source, multi_target,
                                multi_confidence, self.eps,
                            )
                        )
                    else:
                        multi_correction, multi_valid = confidence_weighted_kabsch(
                            multi_source, multi_target,
                            multi_confidence, self.eps,
                        )
                    correction_depth = correction_depth.index_copy(
                        0, multi_rows, multi_correction
                    )
                    well_conditioned = well_conditioned.index_copy(
                        0, multi_rows, multi_valid
                    )
                correction_depth = torch.where(
                    eligible_depth[:, None, None, None],
                    correction_depth,
                    identity_depth,
                )
                proposal_valid_depth = (
                    well_conditioned & eligible_depth[:, None]
                )
            else:
                correction_depth, well_conditioned = (
                    confidence_weighted_kabsch(
                        source_depth,
                        target_depth,
                        confidence_depth,
                        self.eps,
                    )
                )
                correction_depth = torch.where(
                    eligible_depth[:, None, None, None],
                    correction_depth,
                    identity_depth,
                )
                rank3_depth = rank3[depth_batch, depth_joint]
                rank2_depth = rank2[depth_batch, depth_joint]
                rank3_kabsch_valid = (
                    well_conditioned
                    & rank3_depth[:, None]
                    & eligible_depth[:, None]
                )
                rank2_candidate_valid = (
                    confidence_depth.gt(self.eps).any(dim=-1)
                    & rank2_depth[:, None]
                )
                proposal_valid_depth = (
                    rank3_kabsch_valid | rank2_candidate_valid
                ) & eligible_depth[:, None]
            if disagreement_depth is None:
                source_dot = (
                    source_depth * target_depth
                ).sum(dim=-1).clamp(-1.0, 1.0)
                source_cross = torch.linalg.vector_norm(
                    torch.cross(source_depth, target_depth, dim=-1), dim=-1
                )
                child_angle = torch.atan2(source_cross, source_dot)
                disagreement_depth = (
                    (child_angle * confidence_depth).sum(dim=-1)
                    / confidence_depth.sum(dim=-1).clamp_min(self.eps)
                )
                disagreement_depth = torch.where(
                    eligible_depth[:, None],
                    disagreement_depth,
                    torch.zeros_like(disagreement_depth),
                )
            else:
                disagreement_depth = torch.where(
                    eligible_depth[:, None],
                    disagreement_depth,
                    torch.zeros_like(disagreement_depth),
                )

            current_observable_depth = (
                frame_mask[depth_batch]
                & eligible_depth[:, None]
            )
            learned_gate_depth = learned_gate_once[
                depth_batch, :, depth_joint
            ]
            if self.visual_gate is not None and self.gate_mode == "visual":
                confidence_feature = aggregate_confidence[
                    depth_batch, :, depth_joint
                ]
                disagreement_feature = disagreement_depth
                if self.detach_direction_gate_features:
                    confidence_feature = confidence_feature.detach()
                    disagreement_feature = disagreement_feature.detach()
                if visual_gate_static_hidden is not None:
                    dynamic = torch.cat(
                        (
                            confidence_feature.unsqueeze(-1),
                            (disagreement_feature / math.pi).unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                    first = self.visual_gate[0]
                    static_selected = visual_gate_static_hidden[
                        depth_batch, :, depth_joint
                    ]
                    if (
                        fused_visual_gate is not None
                        and static_selected.is_cuda
                        and static_selected.shape[-1] <= 256
                    ):
                        logits = fused_visual_gate(
                            static_selected,
                            dynamic,
                            first.weight[:, self.feature_dim : self.feature_dim + 2],
                            self.visual_gate[2].weight,
                            self.visual_gate[2].bias,
                            self.visual_gate[4].weight,
                            self.visual_gate[4].bias,
                            current_observable_depth,
                        )
                    else:
                        hidden = static_selected + F.linear(
                            dynamic,
                            first.weight[:, self.feature_dim : self.feature_dim + 2],
                        )
                        hidden = self.visual_gate[1](hidden)
                        hidden = self.visual_gate[2](hidden)
                        hidden = self.visual_gate[3](hidden)
                        logits = self.visual_gate[4](hidden)
                        logits = torch.where(
                            current_observable_depth.unsqueeze(-1),
                            logits,
                            torch.zeros_like(logits),
                        )
                else:
                    feature_parts = [
                        normalized_visual[depth_batch, :, depth_joint],
                        confidence_feature.unsqueeze(-1),
                        (disagreement_feature / math.pi).unsqueeze(-1),
                        normalized_count[depth_batch, :, depth_joint],
                        topology_eigen[
                            depth_batch, depth_joint
                        ][:, None, :].expand(-1, frames, -1),
                    ]
                    logits = self._run_visual_gate_feature_parts(
                        feature_parts,
                        token_mask=current_observable_depth,
                    )
                learned_gate_depth = torch.sigmoid(logits.squeeze(-1))
            confidence_gate_depth = confidence_gate[
                depth_batch, :, depth_joint
            ]
            if self.gate_mode == "confidence":
                gate_depth = confidence_gate_depth
            else:
                gate_depth = confidence_gate_depth * learned_gate_depth
            if self.gate_override == "confidence":
                gate_depth = confidence_gate_depth
            elif self.gate_override == "full_on":
                gate_depth = torch.ones_like(confidence_gate_depth)
            elif self.gate_override == "full_off":
                gate_depth = torch.zeros_like(confidence_gate_depth)

            active_depth = current_observable_depth & gate_depth.gt(0.0)
            current_gate_depth = gate_depth * active_depth.to(gate_depth.dtype)
            # Ineligible/zero-child rows are already identity corrections. Do
            # the SO(3) conversions only for rows that can affect the output;
            # scatter exact zeros/identity back for the rest.
            conversion_rows = (
                depth_conversion_indices[depth]
                if visual_gate
                else torch.arange(depth_flat.numel(), device=baseline.device)
            )
            axis_angle_depth = torch.zeros(
                depth_flat.numel(), frames, 3,
                dtype=torch.float32, device=baseline.device
            )
            if visual_gate:
                axis_angle_depth = proposal_axis_angle_depth
            blended_depth = identity_depth.clone()
            if conversion_rows.numel():
                if visual_gate:
                    multi_conversion_rows = depth_multi_indices[depth]
                    if multi_conversion_rows.numel():
                        correction_selected = correction_depth.index_select(
                            0, multi_conversion_rows
                        )
                        axis_selected = (
                            fused_matrix_to_axis_angle(correction_selected.float())
                            if (
                                fused_matrix_to_axis_angle is not None
                                and correction_selected.is_cuda
                                and not self.training
                            )
                            else matrix_to_axis_angle(correction_selected.float())
                        )
                        axis_angle_depth = axis_angle_depth.index_copy(
                            0, multi_conversion_rows, axis_selected
                        )
                    axis_selected = axis_angle_depth.index_select(
                        0, conversion_rows
                    )
                else:
                    correction_selected = correction_depth.index_select(
                        0, conversion_rows
                    )
                    axis_selected = (
                        fused_matrix_to_axis_angle(correction_selected.float())
                        if (
                            fused_matrix_to_axis_angle is not None
                            and correction_selected.is_cuda
                            and not self.training
                        )
                        else matrix_to_axis_angle(correction_selected.float())
                    )
                gate_selected = current_gate_depth.index_select(
                    0, conversion_rows
                )
                applied_selected = (
                    axis_selected * gate_selected.unsqueeze(-1)
                )
                if not visual_gate:
                    axis_angle_depth = axis_angle_depth.index_copy(
                        0, conversion_rows, axis_selected
                    )
                applied_matrix = (
                    fused_axis_angle_to_matrix(applied_selected)
                    if (
                        fused_axis_angle_to_matrix is not None
                        and applied_selected.is_cuda
                        and not self.training
                    )
                    else axis_angle_to_matrix(applied_selected)
                )
                blended_depth = blended_depth.index_copy(
                    0,
                    conversion_rows,
                    applied_matrix,
                )
            applied_axis_angle_depth = (
                axis_angle_depth * current_gate_depth.unsqueeze(-1)
            )
            corrected_depth = blended_depth @ provisional_depth
            corrected_depth = torch.where(
                active_depth.unsqueeze(-1).unsqueeze(-1),
                corrected_depth,
                provisional_depth,
            )

            corrected_global_flat = corrected_global_flat.index_copy(
                0, depth_flat, corrected_depth
            )
            gate_all_flat = gate_all_flat.index_copy(
                0, depth_flat, current_gate_depth
            )
            if return_diagnostics:
                proposed_all_flat = proposed_all_flat.index_copy(
                    0,
                    depth_flat,
                    torch.linalg.vector_norm(axis_angle_depth, dim=-1),
                )
                applied_all_flat = applied_all_flat.index_copy(
                    0,
                    depth_flat,
                    torch.linalg.vector_norm(applied_axis_angle_depth, dim=-1),
                )
            observable_all_flat = observable_all_flat.index_copy(
                0, depth_flat, current_observable_depth
            )
            if return_diagnostics:
                proposal_valid_all_flat = proposal_valid_all_flat.index_copy(
                    0, depth_flat, proposal_valid_depth
                )
        corrected_global = corrected_global_flat.reshape(
            batch_size, joints, frames, 3, 3
        ).permute(0, 2, 1, 3, 4)
        gate_all = gate_all_flat.reshape(
            batch_size, joints, frames
        ).permute(0, 2, 1)
        observable_all = observable_all_flat.reshape(
            batch_size, joints, frames
        ).permute(0, 2, 1)

        parent_global = torch.gather(corrected_global, 2, parent_index)
        relative_candidate = parent_global.transpose(-1, -2) @ corrected_global
        local_candidate = torch.where(
            root_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1),
            corrected_global,
            relative_candidate,
        )
        local_candidate = torch.where(
            (valid_parent | root_mask).unsqueeze(1).unsqueeze(-1).unsqueeze(-1),
            local_candidate,
            baseline,
        )
        active_all = observable_all & gate_all.gt(0.0)
        corrected_local = torch.where(
            active_all.unsqueeze(-1).unsqueeze(-1),
            local_candidate.to(baseline_input.dtype),
            baseline_input,
        )
        # Report the globals induced by the final local rotations. The
        # provisional recurrence above is used to build owner candidates, but
        # siblings corrected at the same depth must not be mistaken for a
        # parent's final state when serializing the output contract.
        final_global = (
            corrected_global
            if (visual_gate or not return_diagnostics)
            else _level_parallel_global_rotations(
                corrected_local.float(), parents, joint_mask, joint_depth, max_depth
            )
        )
        result = {
            "corrected_local_rotation": corrected_local,
            "corrected_global_rotation": final_global.to(baseline_input.dtype),
        }
        if return_diagnostics:
            result.update({
                "observable_swing_gate": gate_all,
                "observable_swing_child_count": gate_child_count.float().unsqueeze(
                    1
                ).expand(batch_size, frames, joints),
                "observable_swing_confidence": aggregate_confidence,
                "observable_swing_mask": observable_all,
                "observable_swing_proposed_angle": proposed_all_flat.reshape(
                    batch_size, joints, frames
                ).permute(0, 2, 1),
                "observable_swing_applied_angle": applied_all_flat.reshape(
                    batch_size, joints, frames
                ).permute(0, 2, 1),
                "observable_swing_kabsch_valid": proposal_valid_all_flat.reshape(
                    batch_size, joints, frames
                ).permute(0, 2, 1),
            })
        return result


    def forward(
        self,
        baseline_local_rotation: torch.Tensor,
        target_bone_direction: torch.Tensor,
        bone_confidence: torch.Tensor,
        batch: dict,
        visual_features: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
        topology_cache: Optional[dict[str, object]] = None,
    ) -> dict[str, torch.Tensor]:
        if (
            baseline_local_rotation.ndim != 5
            or baseline_local_rotation.shape[-2:] != (3, 3)
        ):
            raise ValueError("baseline_local_rotation must be [B,T,J,3,3]")
        batch_size, frames, joints = baseline_local_rotation.shape[:3]
        if target_bone_direction.shape != (batch_size, frames, joints, 3):
            raise ValueError("target_bone_direction shape mismatch")
        if bone_confidence.shape != (batch_size, frames, joints):
            raise ValueError("bone_confidence shape mismatch")
        if self.gate_mode == "visual":
            if visual_features is None:
                raise ValueError("visual gate requires visual_features")
            if visual_features.shape != (
                batch_size, frames, joints, self.feature_dim
            ):
                raise ValueError("visual_features shape mismatch")

        input_dtype = baseline_local_rotation.dtype
        device = baseline_local_rotation.device
        if topology_cache is None:
            parents = batch["parent_a"].to(device=device, dtype=torch.long)
            joint_mask = batch["joint_mask"].to(device=device, dtype=torch.bool)
        else:
            parents = topology_cache["parents"]
            joint_mask = topology_cache["joint_mask"]
        if parents.shape != (batch_size, joints) or joint_mask.shape != (batch_size, joints):
            raise ValueError("topology shape mismatch")
        if topology_cache is None:
            static_rot_joint_mask = batch.get("static_rot_joint_mask")
            if static_rot_joint_mask is None:
                static_rot_joint_mask = torch.zeros_like(joint_mask)
            else:
                static_rot_joint_mask = static_rot_joint_mask.to(
                    device=device, dtype=torch.bool
                )
                if static_rot_joint_mask.shape != (batch_size, joints):
                    raise ValueError("static_rot_joint_mask shape mismatch")
            correction_joint_mask = (
                joint_mask
                if self.gate_mode == "visual"
                else joint_mask & ~static_rot_joint_mask
            )
        else:
            correction_joint_mask = topology_cache["correction_joint_mask"]
        frame_mask = batch.get("frame_valid_mask")
        if frame_mask is None:
            frame_mask = torch.ones(
                batch_size, frames, dtype=torch.bool, device=device
            )
        else:
            frame_mask = frame_mask.to(device=device, dtype=torch.bool)
            if frame_mask.shape != (batch_size, frames):
                raise ValueError("frame_valid_mask shape mismatch")

        if topology_cache is None:
            rest_offset = batch["offset_a"].to(
                device=device, dtype=torch.float32
            )
            if rest_offset.shape != (batch_size, joints, 3):
                raise ValueError("offset_a shape mismatch")
            topology = self._topology_metadata(
                parents, joint_mask, rest_offset, device,
                species=batch.get("species"), rel=batch.get("rel"),
            )
        else:
            rest_offset = topology_cache["rest_offset"]
            topology = topology_cache["topology"]
        child_index = topology["child_index"]
        child_valid = topology["child_valid"]
        compact_width = child_index.shape[-1]

        # Keep all geometry in explicit FP32. The only Python recurrence is
        # over tree depth; each depth compacts its active [batch,joint] rows so
        # masked and other-depth joints never enter the direction solver.
        with _fp32_geometry_context(baseline_local_rotation):
            baseline = baseline_local_rotation.float()
            target_input = target_bone_direction.float()
            confidence_input = bone_confidence.float()
            confidence = torch.where(
                torch.isfinite(confidence_input),
                confidence_input.clamp(0.0, 1.0),
                torch.zeros_like(confidence_input),
            )
            target_finite = torch.isfinite(target_input).all(dim=-1)
            target_norm = torch.linalg.vector_norm(target_input, dim=-1, keepdim=True)
            target_has_direction = target_finite & target_norm.squeeze(-1).gt(self.eps)
            target_unit = torch.where(
                target_has_direction.unsqueeze(-1),
                target_input / target_norm.clamp_min(self.eps),
                torch.zeros_like(target_input),
            )
            compact_target = _gather_compact_children(target_unit, child_index)
            compact_target_has_direction = _gather_compact_children(
                target_has_direction, child_index
            )
            compact_confidence = _gather_compact_children(
                torch.where(target_finite, confidence, torch.zeros_like(confidence)),
                child_index,
            ) * child_valid.unsqueeze(1).float()
            if (
                self.gate_mode != "visual"
                and not self.exclude_zero_rest_edges
            ):
                compact_target_has_direction = (
                    compact_target_has_direction | child_valid.unsqueeze(1)
                )
            if topology_cache is None:
                rest_direction = safe_unit_vector(
                    rest_offset,
                    eps=self.eps,
                    fallback_axis=self.zero_rest_edge_fallback_axis,
                )
                compact_rest_direction = torch.gather(
                    rest_direction.unsqueeze(1).expand(batch_size, joints, joints, 3),
                    2,
                    child_index.unsqueeze(-1).expand(batch_size, joints, compact_width, 3),
                )
            else:
                compact_rest_direction = topology_cache["compact_rest_direction"]
            positive_count = compact_confidence.gt(self.eps).sum(dim=-1)
            aggregate_confidence = (
                compact_confidence.sum(dim=-1)
                / positive_count.clamp_min(1).to(compact_confidence.dtype)
            ).clamp(0.0, 1.0)
            aggregate_confidence = aggregate_confidence * joint_mask.unsqueeze(1).float()
            merged = self._forward_batched(
                baseline=baseline,
                baseline_input=baseline_local_rotation,
                compact_target=compact_target,
                compact_target_has_direction=compact_target_has_direction,
                compact_confidence=compact_confidence,
                compact_rest_direction=compact_rest_direction,
                child_valid=child_valid,
                child_count=topology["child_count"],
                catalog=topology["catalog"],
                joint_mask=joint_mask,
                correction_joint_mask=correction_joint_mask,
                frame_mask=frame_mask,
                parents=parents,
                joint_depth=topology["joint_depth"],
                max_depth=topology["max_depth"],
                depth_indices=topology["depth_indices"],
                depth_single_indices=topology["depth_single_indices"],
                depth_multi_indices=topology["depth_multi_indices"],
                depth_conversion_indices=topology["depth_conversion_indices"],
            aggregate_confidence=aggregate_confidence,
                visual_features=(visual_features.float() if visual_features is not None else None),
                return_diagnostics=return_diagnostics,
                static_cache=topology_cache,
            )
        if self.gate_mode != "visual":
            merged["corrected_global_rotation"] = merged[
                "corrected_global_rotation"
            ].to(input_dtype)
        return merged
