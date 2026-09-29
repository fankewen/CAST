"""Loader for the fixed-topology fused depth geometry CUDA op."""

from __future__ import annotations

import torch


_EXT = None


def _load_extension():
    """Load the compiled CUDA extension."""
    global _EXT
    if _EXT is None:
        try:
            from . import cast_fused_depth_geometry as extension
        except ImportError as exc:
            raise ImportError(
                "The compiled CAST/GRACE depth-geometry extension is "
                "missing. Build it from the repository root with: "
                "python ops/setup.py build_ext --inplace"
            ) from exc

        _EXT = extension

    return _EXT


def fused_depth_geometry(
    provisional: torch.Tensor,
    rest_direction: torch.Tensor,
    target: torch.Tensor,
    target_has_direction: torch.Tensor,
    confidence: torch.Tensor,
    eps: float,
):
    """Fuse source projection, target fallback, and disagreement reduction."""
    if not provisional.is_cuda:
        raise RuntimeError("fused_depth_geometry requires CUDA tensors")

    return _load_extension().forward(
        provisional.contiguous(),
        rest_direction.contiguous(),
        target.contiguous(),
        target_has_direction.contiguous(),
        confidence.contiguous(),
        float(eps),
    )


def fused_axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return _load_extension().axis_angle_to_matrix(axis_angle.contiguous())


def fused_matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return _load_extension().matrix_to_axis_angle(matrix.contiguous())


def fused_minimal_axis_angle(
    source: torch.Tensor, target: torch.Tensor, eps: float
) -> torch.Tensor:
    return _load_extension().minimal_axis_angle(
        source.contiguous(), target.contiguous(), float(eps)
    )


def fused_confidence_weighted_kabsch(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    eps: float,
):
    return _load_extension().confidence_weighted_kabsch(
        source.contiguous(), target.contiguous(), confidence.contiguous(), float(eps)
    )


def fused_visual_gate(
    static_hidden: torch.Tensor,
    dynamic: torch.Tensor,
    dynamic_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    return _load_extension().visual_gate(
        static_hidden.contiguous(),
        dynamic.contiguous(),
        dynamic_weight.contiguous(),
        hidden_weight.contiguous(),
        hidden_bias.contiguous(),
        output_weight.contiguous(),
        output_bias.contiguous(),
        token_mask.contiguous(),
    )


def fused_fixed_visual_depth(
    baseline: torch.Tensor,
    rest: torch.Tensor,
    target: torch.Tensor,
    has_direction: torch.Tensor,
    confidence: torch.Tensor,
    aggregate_confidence: torch.Tensor,
    static_hidden: torch.Tensor,
    dynamic_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    depth_flat: torch.Tensor,
    parent_flat: torch.Tensor,
    child_count: torch.Tensor,
    root_mask: torch.Tensor,
    valid_parent: torch.Tensor,
    eligible: torch.Tensor,
    frame_mask: torch.Tensor,
    corrected_global: torch.Tensor,
    corrected_local: torch.Tensor,
    gate_out: torch.Tensor,
    observable_out: torch.Tensor,
    fixed_gate_scale: float,
    gate_override: int,
    eps: float,
) -> None:
    """Run one complete fixed-topology visual-correction tree depth."""
    _load_extension().fixed_visual_depth(
        baseline.contiguous(),
        rest.contiguous(),
        target.contiguous(),
        has_direction.contiguous(),
        confidence.contiguous(),
        aggregate_confidence.contiguous(),
        static_hidden.contiguous(),
        dynamic_weight.contiguous(),
        hidden_weight.contiguous(),
        hidden_bias.contiguous(),
        output_weight.contiguous(),
        output_bias.contiguous(),
        depth_flat.contiguous(),
        parent_flat.contiguous(),
        child_count.contiguous(),
        root_mask.contiguous(),
        valid_parent.contiguous(),
        eligible.contiguous(),
        frame_mask.contiguous(),
        corrected_global,
        corrected_local,
        gate_out,
        observable_out,
        float(fixed_gate_scale),
        int(gate_override),
        float(eps),
    )
