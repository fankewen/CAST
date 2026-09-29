"""Rotation conversions used by the unified video2motion pipeline."""

import torch
import torch.nn.functional as F


def rot6d_to_rotmat_tensor(rot_6d):
    """Decode the cache 6D convention into rotation matrices.

    The six channels are ``concat(R[:, 0], R[:, 1])`` (two complete columns),
    not a row-major flatten of ``R[:2, :]``.  The network head may emit
    non-orthogonal vectors, so Gram--Schmidt normalization is part of decoding.
    """
    # Validation slices each motion out of a padded [B, T, J, 6] batch.  Such
    # slices are commonly non-contiguous, so view() is not valid here even
    # though their logical shape is correct.
    x = rot_6d.reshape(-1, 6)
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    eps = 1e-6
    a1_norm = torch.linalg.vector_norm(a1, dim=1, keepdim=True)
    root_axis = torch.zeros_like(a1)
    root_axis[:, 0] = 1
    b1 = torch.where(a1_norm > eps, a1 / a1_norm.clamp_min(eps), root_axis)
    a2_orthogonal = a2 - (b1 * a2).sum(1, keepdim=True) * b1
    a2_norm = torch.linalg.vector_norm(a2_orthogonal, dim=1, keepdim=True)
    # When both predicted columns are zero/parallel, choose the coordinate axis
    # least aligned with b1 and orthogonalize it. This guarantees a finite SO(3)
    # matrix instead of the all-zero "rotation" returned by normalize(0).
    fallback_axis = F.one_hot(
        torch.argmin(torch.abs(b1), dim=1), num_classes=3
    ).to(dtype=b1.dtype)
    fallback_b2 = fallback_axis - (fallback_axis * b1).sum(1, keepdim=True) * b1
    fallback_b2 = F.normalize(fallback_b2, dim=1, eps=eps)
    b2 = torch.where(
        a2_norm > eps,
        a2_orthogonal / a2_norm.clamp_min(eps),
        fallback_b2,
    )
    b3 = torch.cross(b1, b2, dim=1)
    rotmat = torch.stack([b1, b2, b3], dim=-1)
    rotmat = rotmat.reshape(*rot_6d.shape[:-1], 3, 3)
    return rotmat


def project_rotmat_to_so3(matrix: torch.Tensor) -> torch.Tensor:
    """Project arbitrary/averaged 3x3 matrices to the nearest proper rotation."""
    original_dtype = matrix.dtype
    u, _, vh = torch.linalg.svd(matrix.float())
    uvh = u @ vh
    correction = torch.ones(matrix.shape[:-2] + (3,), device=matrix.device, dtype=u.dtype)
    correction[..., -1] = torch.linalg.det(uvh)
    projected = (u * correction.unsqueeze(-2)) @ vh
    return projected.to(original_dtype)


def rotmat_to_rot6d_tensor(matrix: torch.Tensor) -> torch.Tensor:
    """Encode matrices as concat(R[:,0], R[:,1]), matching the cache format."""
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)
