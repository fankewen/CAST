"""Reusable execution metadata for padded graph-attention batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from models.attention_layout import cached_tensor_metadata


GraphExecutionMode: TypeAlias = Literal["auto", "dense", "grouped"]
JointLengthGroups: TypeAlias = tuple[tuple[int, torch.Tensor], ...]
DEFAULT_AUTO_GROUP_MAX_PAIR_RATIO = 0.80
_auto_group_max_pair_ratio = DEFAULT_AUTO_GROUP_MAX_PAIR_RATIO


def set_auto_group_max_pair_ratio(value: float) -> None:
    """Set the per-process auto-selection threshold used by graph layers."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("auto group max pair ratio must be in [0, 1]")
    global _auto_group_max_pair_ratio
    _auto_group_max_pair_ratio = value


def get_auto_group_max_pair_ratio() -> float:
    """Return the per-process auto-selection threshold."""
    return _auto_group_max_pair_ratio


@dataclass(frozen=True)
class JointLengthLayout:
    """One batch's reusable graph-attention grouping and execution choice."""

    groups: JointLengthGroups
    use_grouped: bool
    pair_ratio: float


def build_joint_length_groups(
    mask: torch.Tensor | None,
    mode: GraphExecutionMode = "auto",
    max_group_pair_ratio: float | None = None,
) -> JointLengthLayout | None:
    """Build one reusable dense/grouped execution decision for a batch.

    The estimate compares exact-length quadratic pair work with padded dense
    pair work. Grouping is selected conservatively only when it removes at
    least 20 percent of the pair work. This avoids marginal cases where extra
    launches and gather/scatter overhead can outweigh the saved computation.
    """

    if mask is None:
        return None
    if mask.ndim < 2:
        raise ValueError("joint mask must end with a joint axis")
    if mode not in {"auto", "dense", "grouped"}:
        raise ValueError(
            "graph execution mode must be auto, dense, or grouped"
        )
    if max_group_pair_ratio is None:
        max_group_pair_ratio = get_auto_group_max_pair_ratio()
    max_group_pair_ratio = float(max_group_pair_ratio)
    if not 0.0 <= max_group_pair_ratio <= 1.0:
        raise ValueError("max_group_pair_ratio must be in [0, 1]")

    def build():
        flat_mask = mask.bool().reshape(-1, mask.shape[-1])
        valid_lengths = flat_mask.sum(dim=-1, dtype=torch.int64)
        unique_lengths, counts = torch.unique(
            valid_lengths,
            sorted=True,
            return_counts=True,
        )
        length_values = [int(value) for value in unique_lengths.tolist()]
        count_values = [int(value) for value in counts.tolist()]
        groups = tuple(
            (valid_length, valid_lengths.eq(valid_length))
            for valid_length in length_values
        )

        total_rows = flat_mask.shape[0]
        padded_joints = flat_mask.shape[1]
        dense_pairs = total_rows * padded_joints * padded_joints
        grouped_pairs = sum(
            count * valid_length * valid_length
            for valid_length, count in zip(length_values, count_values)
        )
        pair_ratio = (
            float(grouped_pairs) / float(dense_pairs)
            if dense_pairs > 0
            else 1.0
        )
        nonempty_group_count = sum(
            valid_length > 0 for valid_length in length_values
        )
        if mode == "dense":
            use_grouped = False
        elif mode == "grouped":
            use_grouped = nonempty_group_count > 1 or 0 in length_values
        else:
            use_grouped = (
                (nonempty_group_count > 1 or 0 in length_values)
                and pair_ratio <= max_group_pair_ratio
            )
        return JointLengthLayout(
            groups=groups,
            use_grouped=use_grouped,
            pair_ratio=pair_ratio,
        )

    return cached_tensor_metadata(
        "joint_length_groups",
        (mask,),
        (mode, max_group_pair_ratio),
        build,
    )
