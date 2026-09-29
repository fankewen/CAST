"""Small Triton kernels for the verified 8-layer + 8-layer model.

The kernel below only replaces the relation-bias construction.  It preserves
the existing SDPA implementation for the numerically sensitive softmax/value
path while removing four intermediate matmul/gather launches per graph layer.

The verified configuration uses ``q_dim=256`` and ``num_heads=8``, giving a
head dimension of 32.  The kernel is enabled by default for head dimensions
up to 32 and automatically falls back to PyTorch for larger dimensions.
"""
from __future__ import annotations

import torch
import os

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU/test environments
    triton = None
    tl = None


_SUPPORTED_HEAD_DIM = 32


if triton is not None:

    @triton.jit
    def _relation_bias_kernel(
        q_ptr, k_ptr, qr_ptr, kr_ptr, out_ptr,
        sqb, sqj, sqd,
        skb, skj, skd,
        srb, srq, srk, srd,
        skrb, skrq, skrk, skrd,
        sob, soq, sok,
        J, D, scale,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        qi = tl.program_id(1)
        # The caller flattens batch and head into the leading dimension.
        q_base = q_ptr + pid * sqb + qi * sqj
        k_base = k_ptr + pid * skb + qi * skj
        qr_base = qr_ptr + pid * srb + qi * srq
        kr_base = kr_ptr + pid * skrb + qi * skrq
        out_base = out_ptr + pid * sob + qi * soq
        d = tl.arange(0, BLOCK_D)
        qv = tl.load(q_base + d * sqd, mask=d < D, other=0.0)
        kv_query = tl.load(k_base + d * skd, mask=d < D, other=0.0)
        for start in tl.range(0, J, BLOCK_K):
            kk = start + tl.arange(0, BLOCK_K)
            valid_k = kk < J
            qr = tl.load(
                qr_base + kk[:, None] * srk + d[None, :] * srd,
                mask=valid_k[:, None] & (d[None, :] < D), other=0.0,
            )
            kr = tl.load(
                kr_base + kk[:, None] * skrk + d[None, :] * skrd,
                mask=valid_k[:, None] & (d[None, :] < D), other=0.0,
            )
            rel_q = tl.sum(qr * qv[None, :], axis=1)
            rel_k = tl.sum(kr * kv_query[None, :], axis=1)
            # The output is laid out [B*H, J, J].
            # Return the unscaled relation logits.  The caller applies the
            # attention scale together with the regular SDPA mask, exactly as
            # in the unfused implementation.
            tl.store(out_base + kk * sok, rel_q + rel_k, mask=valid_k)


def fixed_relation_bias(q, k, relation_cache, scale):
    """Return unscaled relation bias [B,H,J,J] using one fused launch."""
    # Set CAST_USE_FIXED_GRAPH_KERNEL=0 to force the PyTorch path.
    use_kernel = os.environ.get("CAST_USE_FIXED_GRAPH_KERNEL", "1") == "1"
    if (
        not use_kernel
        or
        triton is None
        or not q.is_cuda
        or relation_cache is None
        or q.ndim != 4
        or q.shape[-1] > _SUPPORTED_HEAD_DIM
    ):
        q_rel = relation_cache["query"]
        k_rel = relation_cache["key"]
        return ((q.unsqueeze(3) * q_rel).sum(-1) +
                (k.unsqueeze(3) * k_rel).sum(-1))

    B, H, J, D = q.shape
    out = torch.empty((B, H, J, J), device=q.device, dtype=q.dtype)
    # Flatten B/H into one leading dimension.  All tensors are contiguous from
    # build_streaming_relation_cache, so strides are simple and stable.
    qf = q.reshape(B * H, J, D)
    kf = k.reshape(B * H, J, D)
    qrf = relation_cache["query"].reshape(B * H, J, J, D)
    krf = relation_cache["key"].reshape(B * H, J, J, D)
    outf = out.reshape(B * H, J, J)
    grid = (B * H, J)
    _relation_bias_kernel[grid](
        qf, kf, qrf, krf, outf,
        qf.stride(0), qf.stride(1), qf.stride(2),
        kf.stride(0), kf.stride(1), kf.stride(2),
        qrf.stride(0), qrf.stride(1), qrf.stride(2), qrf.stride(3),
        krf.stride(0), krf.stride(1), krf.stride(2), krf.stride(3),
        outf.stride(0), outf.stride(1), outf.stride(2),
        J, D, float(scale), BLOCK_D=_SUPPORTED_HEAD_DIM, BLOCK_K=64,
    )
    return out
