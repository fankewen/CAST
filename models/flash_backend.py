from __future__ import annotations

import torch
from models.attention_layout import packed_layout
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
except (ImportError, OSError):
    flash_attn_varlen_func = None


def flash_attn_is_available() -> bool:
    return flash_attn_varlen_func is not None


def can_use_flash_attn(*tensors: torch.Tensor) -> bool:
    return (
        flash_attn_varlen_func is not None
        and all(tensor.is_cuda for tensor in tensors)
        and all(
            tensor.dtype in (torch.float16, torch.bfloat16)
            for tensor in tensors
        )
    )


def flash_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_mask: torch.Tensor,
    kv_mask: torch.Tensor,
    *,
    dropout_p: float,
    softmax_scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    all_sequences_active: bool = False,
) -> torch.Tensor | None:
    """Run official flash-attn varlen attention on packed valid tokens.

    Args:
        q: ``[N,Lq,H,D]``.
        k/v: ``[N,Lk,H,D]``.
        q_mask: ``[N,Lq]`` valid query tokens.
        kv_mask: ``[N,Lk]`` valid key/value tokens.

    Returns:
        A padded tensor shaped like ``q`` when FlashAttention is usable,
        otherwise ``None`` so the caller can use its compatibility fallback.
        When ``all_sequences_active`` is true, every sequence is known to have
        at least one valid token and the caller has already removed inactive
        rows, avoiding a second gather/scatter of the projected tensors.
    """
    if not can_use_flash_attn(q, k, v):
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("FlashAttention q/k/v must be [N,L,H,D]")
    if k.shape != v.shape:
        raise ValueError("FlashAttention k and v must share a shape")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("FlashAttention q/k/v batch, head, and dim mismatch")
    if q_mask.shape != q.shape[:2] or kv_mask.shape != k.shape[:2]:
        raise ValueError("FlashAttention masks must match q/k sequence shapes")

    q_mask = q_mask.bool()
    kv_mask = kv_mask.bool()
    q_lengths = q_mask.sum(dim=-1, dtype=torch.int32)
    kv_lengths = kv_mask.sum(dim=-1, dtype=torch.int32)
    if all_sequences_active:
        if torch.any((q_lengths <= 0) | (kv_lengths <= 0)):
            raise ValueError(
                "all_sequences_active=True requires non-empty q/kv sequences"
            )
        active = None
        q_active = q
        k_active = k
        v_active = v
        q_mask_active = q_mask
        kv_mask_active = kv_mask
        q_lengths_active = q_lengths
        kv_lengths_active = kv_lengths
    else:
        active = (q_lengths > 0) & (kv_lengths > 0)
        if not torch.any(active):
            return torch.zeros_like(q)
        q_active = q[active].contiguous()
        k_active = k[active].contiguous()
        v_active = v[active].contiguous()
        q_mask_active = q_mask[active]
        kv_mask_active = kv_mask[active]
        q_lengths_active = q_lengths[active]
        kv_lengths_active = kv_lengths[active]

    q_unpadded = q_active[q_mask_active].contiguous()
    k_unpadded = k_active[kv_mask_active].contiguous()
    v_unpadded = v_active[kv_mask_active].contiguous()
    cu_q = torch.nn.functional.pad(
        torch.cumsum(q_lengths_active, dim=0, dtype=torch.int32),
        (1, 0),
    )
    cu_k = torch.nn.functional.pad(
        torch.cumsum(kv_lengths_active, dim=0, dtype=torch.int32),
        (1, 0),
    )
    max_q = int(q_lengths_active.max().item())
    max_k = int(kv_lengths_active.max().item())

    out_unpadded = flash_attn_varlen_func(
        q_unpadded,
        k_unpadded,
        v_unpadded,
        cu_q,
        cu_k,
        max_q,
        max_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=window_size,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )

    out_active = torch.zeros_like(q_active)
    out_active[q_mask_active] = out_unpadded
    if all_sequences_active:
        return out_active
    out = torch.zeros_like(q)
    out[active] = out_active
    return out


def flash_unpadded_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    *,
    dropout_p: float,
    layout: dict | None = None,
    softmax_scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    causal: bool = False,
) -> torch.Tensor | None:
    """Run FlashAttention on already-packed equal-length q/k/v sequences."""
    if not can_use_flash_attn(q, k, v):
        return None
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Packed FlashAttention q/k/v must share [tokens,H,D]")
    if lengths.ndim != 1 or lengths.dtype != torch.int32:
        raise ValueError("Packed FlashAttention lengths must be int32 [N]")
    if layout is None:
        cu = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
        max_length = int(lengths.max().item())
    else:
        cu, max_length = layout["cu"], layout["max_length"]
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu,
        cu,
        max_length,
        max_length,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )


def flash_mha_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_mask: torch.Tensor,
    key_mask: torch.Tensor | None,
    mha: torch.nn.MultiheadAttention,
    *,
    dropout_p: float,
    post_projection: torch.nn.Module | None = None,
) -> torch.Tensor | None:
    """Run varlen FlashAttention using an MHA module's parameters.

    Padding is removed *before* the Q/K/V projections.  This matters for the
    short joint sequences used by MocapAnything: packing already-projected
    tensors skipped only the attention kernel while padded joints still paid
    for all three input projections and the output projection.
    """
    if (
        not flash_attn_is_available()
        or not query.is_cuda
        or not key.is_cuda
        or not value.is_cuda
    ):
        return None
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Flash MHA query/key/value must be [N,L,D]")
    if key.shape != value.shape:
        raise ValueError("Flash MHA key and value must share a shape")
    if query.shape[0] != key.shape[0]:
        raise ValueError("Flash MHA query/key batch mismatch")
    if query_mask.shape != query.shape[:2] or (key_mask is not None and key_mask.shape != key.shape[:2]):
        raise ValueError("Flash MHA masks must match sequence shapes")
    if mha.in_proj_weight is None:
        raise ValueError("Flash MHA requires a packed in_proj_weight")
    if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
        raise ValueError("Flash MHA does not support bias_k/bias_v/zero attention")

    layout = packed_layout(query_mask, key_mask, key.shape[1])
    q_indices, k_indices = layout["q_indices"], layout["k_indices"]
    if q_indices.numel() == 0:
        return torch.zeros_like(query)
    query_unpadded = query.flatten(0, 1).index_select(0, q_indices)
    key_unpadded = key.flatten(0, 1).index_select(0, k_indices)
    value_unpadded = value.flatten(0, 1).index_select(0, k_indices)

    q_weight, k_weight, v_weight = mha.in_proj_weight.chunk(3, dim=0)
    if mha.in_proj_bias is None:
        q_bias = k_bias = v_bias = None
    else:
        q_bias, k_bias, v_bias = mha.in_proj_bias.chunk(3, dim=0)

    q = F.linear(query_unpadded, q_weight, q_bias)
    k = F.linear(key_unpadded, k_weight, k_bias)
    v = F.linear(value_unpadded, v_weight, v_bias)
    if not can_use_flash_attn(q, k, v):
        return None

    head_dim = mha.embed_dim // mha.num_heads
    q = q.view(-1, mha.num_heads, head_dim)
    k = k.view(-1, mha.num_heads, head_dim)
    v = v.view(-1, mha.num_heads, head_dim)
    cu_q, cu_k = layout["cu_q"], layout["cu_k"]
    max_q, max_k = layout["max_q"], layout["max_k"]

    out_unpadded = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        dropout_p=dropout_p,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
    )
    out_unpadded = out_unpadded.reshape(-1, mha.embed_dim)
    # The output projection is also token-wise, so apply it before scattering
    # and avoid projecting padding outputs.
    out_unpadded = F.linear(
        out_unpadded,
        mha.out_proj.weight,
        mha.out_proj.bias,
    )
    if post_projection is not None:
        out_unpadded = post_projection(out_unpadded)

    out = out_unpadded.new_zeros(query.shape[0] * query.shape[1], out_unpadded.shape[-1])
    out = out.index_copy(0, q_indices, out_unpadded)
    return out.view(*query.shape[:2], out_unpadded.shape[-1])
