from __future__ import annotations

import torch
from models.attention_layout import temporal_layout, masked_pointwise
import torch.nn as nn
import torch.nn.functional as F

from models.flash_backend import (
    flash_attn_is_available,
    flash_unpadded_attention,
)


# =========================================================
# RoPE helpers for per-joint temporal attention
# =========================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


def apply_rope_1d_perjoint(q: torch.Tensor, k: torch.Tensor):
    """
    q, k: [BJ, H, T, Dh]
    """
    BJ, H, T, Dh = q.shape
    assert Dh % 2 == 0, f"RoPE head dim must be even, got {Dh}"

    device = q.device
    dtype = q.dtype
    half_dim = Dh // 2

    pos = torch.arange(T, device=device, dtype=torch.float32)
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

    freqs = torch.outer(pos, inv_freq)  # [T, half_dim]
    cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=dtype)  # [T, Dh]
    sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,Dh]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def apply_rope_1d_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
):
    """Apply the same temporal RoPE to packed ``[tokens,H,Dh]`` values."""
    if q.ndim != 3 or k.shape != q.shape:
        raise ValueError("Packed RoPE q/k must share [tokens,H,Dh]")
    if position_ids.shape != (q.shape[0],):
        raise ValueError("Packed RoPE position_ids must match token count")
    head_dim = q.shape[-1]
    if head_dim % 2:
        raise ValueError("Packed RoPE requires an even head dimension")
    half_dim = head_dim // 2
    freq_seq = torch.arange(
        half_dim, device=q.device, dtype=torch.float32
    )
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = frequencies.cos().repeat_interleave(2, dim=-1).to(q.dtype)
    sin = frequencies.sin().repeat_interleave(2, dim=-1).to(q.dtype)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


# =========================================================
# Per-joint Temporal Attention with RoPE + optional local window
# =========================================================
class TemporalPerJointMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nheads=8,
        dropout=0.1,
        temporal_window=2,
        use_temporal_bias=True,
        causal=False,
        causal_chunk_size=1,
    ):
        super().__init__()
        assert d_model % nheads == 0
        self.d_model = d_model
        self.nheads = nheads
        self.att_size = d_model // nheads
        self.scale = self.att_size ** -0.5
        assert self.att_size % 2 == 0, f"RoPE requires even head dim, got {self.att_size}"

        if temporal_window is not None:
            if isinstance(temporal_window, bool) or not isinstance(temporal_window, int):
                raise TypeError("temporal_window must be a positive integer or None")
            if temporal_window <= 0:
                raise ValueError("temporal_window must be positive when specified")
        self.temporal_window = temporal_window
        self.causal = bool(causal)
        if isinstance(causal_chunk_size, bool) or not isinstance(causal_chunk_size, int) or causal_chunk_size < 1:
            raise ValueError("causal_chunk_size must be a positive integer")
        if causal_chunk_size > 1 and not self.causal:
            raise ValueError("causal_chunk_size > 1 requires causal=True")
        self.causal_chunk_size = causal_chunk_size
        # Global temporal attention keeps RoPE but has no finite relative
        # offset vocabulary and therefore does not instantiate a local bias.
        self.use_temporal_bias = bool(use_temporal_bias and temporal_window is not None)

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._inference_qkv = None
        half_dim = self.att_size // 2
        freq_seq = torch.arange(half_dim, dtype=torch.float32)
        self.register_buffer("rope_inv_freq", 1.0 / (10000 ** (freq_seq / half_dim)), persistent=False)

        if self.use_temporal_bias:
            self.temporal_bias = nn.Embedding(2 * temporal_window + 1, nheads)
        else:
            self.temporal_bias = None
        self.last_backend = "uninitialized"

    @torch.no_grad()
    def prepare_inference_fusion(self):
        if self._inference_qkv is not None:
            return
        fused = nn.Linear(self.d_model, 3 * self.d_model, bias=True).to(
            device=self.linear_q.weight.device, dtype=self.linear_q.weight.dtype
        )
        fused.weight.copy_(torch.cat((self.linear_q.weight, self.linear_k.weight, self.linear_v.weight), dim=0))
        fused.bias.copy_(torch.cat((self.linear_q.bias, self.linear_k.bias, self.linear_v.bias), dim=0))
        fused.requires_grad_(False)
        self._inference_qkv = fused

    def _project_qkv(self, x):
        if self.training:
            weight = torch.cat(
                (self.linear_q.weight, self.linear_k.weight, self.linear_v.weight),
                dim=0,
            )
            bias = torch.cat(
                (self.linear_q.bias, self.linear_k.bias, self.linear_v.bias),
                dim=0,
            )
            return F.linear(x, weight, bias).chunk(3, dim=-1)
        if self._inference_qkv is not None:
            return self._inference_qkv(x).chunk(3, dim=-1)
        return self.linear_q(x), self.linear_k(x), self.linear_v(x)

    def _apply_rope_single_position(self, q, k, position):
        if not isinstance(position, torch.Tensor):
            position = torch.as_tensor(position, device=q.device)
        frequencies = position.to(torch.float32) * self.rope_inv_freq
        cos = frequencies.cos().repeat_interleave(2).to(q.dtype).view(1, 1, -1)
        sin = frequencies.sin().repeat_interleave(2).to(q.dtype).view(1, 1, -1)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

    def _expand_mask_to_btj(self, mask, B, T, J, device):
        if mask is None:
            return torch.ones(B, T, J, device=device, dtype=torch.bool)
        if mask.dim() == 2:
            return mask.unsqueeze(1).expand(-1, T, -1).bool()
        elif mask.dim() == 3:
            return mask.bool()
        else:
            raise ValueError(f"mask dim should be 2 or 3, got {mask.dim()}")

    def _build_window_mask(self, T, device):
        time_ids = torch.arange(T, device=device)
        delta_t = time_ids[:, None] - time_ids[None, :]
        if self.temporal_window is None:
            visible = torch.ones((T, T), device=device, dtype=torch.bool)
        else:
            visible = delta_t.abs() <= self.temporal_window
        if self.causal:
            chunk_ids = time_ids // self.causal_chunk_size
            visible = visible & (chunk_ids[:, None] >= chunk_ids[None, :])
        return visible, delta_t

    def forward(
        self,
        q,   # [B,T,J,D]
        k,   # [B,T,J,D]
        v,   # [B,T,J,D]
        mask=None,  # [B,J] or [B,T,J]
    ):
        B, T, J, D = q.shape
        device = q.device
        orig_size = q.size()
        self_attention = q is k and q is v

        # FlashAttention's causal flag is triangular per token, not per chunk.
        # Chunk sizes above one use the explicit visibility mask in SDPA.
        if self.causal_chunk_size == 1 and self.temporal_bias is None and flash_attn_is_available() and q.is_cuda:
            layout = temporal_layout(mask, B, T, J, device)
            indices = layout["indices"]
            if indices.numel() == 0:
                return torch.zeros_like(q)
            q_flat = q.permute(0, 2, 1, 3).reshape(B * J * T, D)
            if self_attention:
                k_flat = v_flat = q_flat
                q_unpadded = q_flat.index_select(0, indices)
                q_unpadded, k_unpadded, v_unpadded = self._project_qkv(
                    q_unpadded
                )
            else:
                k_flat = k.permute(0, 2, 1, 3).reshape(B * J * T, D)
                v_flat = v.permute(0, 2, 1, 3).reshape(B * J * T, D)
                q_unpadded = self.linear_q(q_flat.index_select(0, indices))
                k_unpadded = self.linear_k(k_flat.index_select(0, indices))
                v_unpadded = self.linear_v(v_flat.index_select(0, indices))
            active_lengths = layout["lengths"]
            q_unpadded = q_unpadded.view(
                -1, self.nheads, self.att_size
            )
            k_unpadded = k_unpadded.view(
                -1, self.nheads, self.att_size
            )
            v_unpadded = v_unpadded.view(
                -1, self.nheads, self.att_size
            )
            position_ids = layout["positions"]
            q_unpadded, k_unpadded = apply_rope_1d_packed(
                q_unpadded,
                k_unpadded,
                position_ids,
            )
            flash_window = (
                (-1, -1)
                if self.temporal_window is None
                else (
                    self.temporal_window,
                    0 if self.causal else self.temporal_window,
                )
            )
            out_unpadded = flash_unpadded_attention(
                q_unpadded,
                k_unpadded,
                v_unpadded,
                active_lengths,
                layout=layout,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                window_size=flash_window,
                causal=self.causal,
            )
            if out_unpadded is not None:
                out_unpadded = out_unpadded.reshape(-1, D)
                out_unpadded = self.output_layer(out_unpadded)
                out_unpadded = self.dropout(out_unpadded)
                padded_out = out_unpadded.new_zeros(B * J * T, D)
                padded_out = padded_out.index_copy(0, indices, out_unpadded)
                out = padded_out.view(B, J, T, D).permute(
                    0, 2, 1, 3
                ).contiguous()
                self.last_backend = "flash_attn"
                assert out.size() == orig_size
                return out

        token_mask_btj = self._expand_mask_to_btj(mask, B, T, J, device)

        q = q.permute(0, 2, 1, 3).contiguous().view(B * J, T, D)
        if self_attention:
            k = v = q
        else:
            k = k.permute(0, 2, 1, 3).contiguous().view(B * J, T, D)
            v = v.permute(0, 2, 1, 3).contiguous().view(B * J, T, D)
        token_mask = (
            token_mask_btj.permute(0, 2, 1)
            .contiguous()
            .view(B * J, T)
        )
        active_joint_sequence = token_mask.any(dim=-1)
        if not torch.any(active_joint_sequence):
            return torch.zeros_like(q).view(B, J, T, D).permute(
                0, 2, 1, 3
            ).contiguous()

        # Compatibility path for CPU, missing FlashAttention, or temporal
        # relation bias.  It still removes completely padded joint sequences
        # before dense Q/K/V and SDPA computation.
        if self_attention:
            q, k, v = self._project_qkv(q[active_joint_sequence])
        else:
            q = self.linear_q(q[active_joint_sequence])
            k = self.linear_k(k[active_joint_sequence])
            v = self.linear_v(v[active_joint_sequence])
        token_mask_active = token_mask[active_joint_sequence]
        active_count = q.shape[0]
        q = q.view(
            active_count, T, self.nheads, self.att_size
        ).transpose(1, 2)
        k = k.view(
            active_count, T, self.nheads, self.att_size
        ).transpose(1, 2)
        v = v.view(
            active_count, T, self.nheads, self.att_size
        ).transpose(1, 2)
        q, k = apply_rope_1d_perjoint(q, k)

        window_vis, delta_t = self._build_window_mask(T, device)
        allowed = (
            window_vis.unsqueeze(0).unsqueeze(0)
            & token_mask_active[:, None, None, :]
        )
        attn_mask = allowed
        if self.temporal_bias is not None:
            dt_clamped = delta_t.clamp(
                -self.temporal_window, self.temporal_window
            )
            dt_index = dt_clamped + self.temporal_window
            t_bias = self.temporal_bias(dt_index)
            t_bias = t_bias.permute(2, 0, 1).unsqueeze(0)
            attn_mask = (t_bias * self.scale).expand(
                active_count, -1, -1, -1
            ).masked_fill(~allowed, float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(active_count, T, D)
        out = self.output_layer(out)
        out = self.dropout(out)
        out = out * token_mask_active.unsqueeze(-1).to(out.dtype)
        self.last_backend = "sdpa"

        padded_out = out.new_zeros(B * J, T, D)
        padded_out[active_joint_sequence] = out
        out = padded_out.view(B, J, T, D).permute(
            0, 2, 1, 3
        ).contiguous()

        assert out.size() == orig_size
        return out

    @torch.no_grad()
    def forward_cached(self, x, mask, cache: TemporalKVCache):
        if self.causal_chunk_size != 1:
            raise RuntimeError("One-frame KV streaming requires causal_chunk_size=1; use forward with complete chunks")
        if not self.causal:
            raise RuntimeError("KV cache requires causal temporal attention")
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("forward_cached expects x with shape [B,1,J,D]")
        B, _, J, D = x.shape
        token_mask = self._expand_mask_to_btj(mask, B, 1, J, x.device)[:, 0]
        x0 = x[:, 0].reshape(B * J, D)
        qh, kh, vh = self._project_qkv(x0)
        qh = qh.view(B * J, self.nheads, self.att_size)
        kh = kh.view(B * J, self.nheads, self.att_size)
        vh = vh.view(B * J, self.nheads, self.att_size)
        position = cache.next_position
        qh, kh = self._apply_rope_single_position(qh, kh, position)
        keys, values = cache.update(kh.unsqueeze(2), vh.unsqueeze(2))
        attn_mask = None
        if self.temporal_window is not None:
            # Empty slots of a half-filled ring carry a distance beyond any
            # window, so this comparison also drops them.
            delta = cache.relative_delta(x.device)
            allowed = delta <= self.temporal_window
            attn_mask = allowed.view(1, 1, 1, -1)
            if self.temporal_bias is not None:
                idx = delta.clamp(0, self.temporal_window) + self.temporal_window
                bias = self.temporal_bias(idx).transpose(0, 1).unsqueeze(0).unsqueeze(2)
                attn_mask = (bias * self.scale).masked_fill(~allowed.view(1, 1, 1, -1), float("-inf"))
        elif cache.max_length is not None:
            # Without a temporal window the empty slots have to be excluded
            # explicitly while the ring is still filling.
            attn_mask = cache.valid_mask(x.device).view(1, 1, 1, -1)
        out = F.scaled_dot_product_attention(
            qh.unsqueeze(2), keys, values,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        ).transpose(1, 2).reshape(B * J, D)
        out = self.dropout(self.output_layer(out)).view(B, 1, J, D)
        out = out * token_mask.unsqueeze(1).unsqueeze(-1).to(out.dtype)
        self.last_backend = "sdpa_kv_cache"
        return out


# =========================================================
# Per-joint Temporal Transformer Block
# =========================================================
class TemporalPerJointTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        nheads=8,
        dropout=0.1,
        ff_mult=4,
        temporal_window=2,
        use_temporal_bias=True,
        causal=False,
        causal_chunk_size=1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalPerJointMultiHeadAttention(
            d_model=dim,
            nheads=nheads,
            dropout=dropout,
            temporal_window=temporal_window,
            use_temporal_bias=use_temporal_bias,
            causal=causal,
            causal_chunk_size=causal_chunk_size,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, joint_mask=None):
        residual = x
        x = masked_pointwise(x, joint_mask, self.norm1)
        x = self.attn(x, x, x, mask=joint_mask)
        x = residual + x

        if joint_mask is not None:
            if joint_mask.dim() == 2:
                mask4d = joint_mask.unsqueeze(1).unsqueeze(-1).float()
            else:
                mask4d = joint_mask.unsqueeze(-1).float()
            x = x * mask4d

        x = masked_pointwise(
            x, joint_mask, lambda h: self.ffn(self.norm2(h)), residual=True
        )

        if joint_mask is not None:
            x = x * mask4d

        return x

    def forward_streaming(self, x, joint_mask=None, cache: TemporalKVCache | None = None):
        if cache is None:
            raise ValueError("forward_streaming requires a TemporalKVCache")
        residual = x
        x = masked_pointwise(x, joint_mask, self.norm1)
        x = self.attn.forward_cached(x, joint_mask, cache)
        x = residual + x
        if joint_mask is not None:
            mask4d = (joint_mask.unsqueeze(1) if joint_mask.dim() == 2 else joint_mask).unsqueeze(-1).to(x.dtype)
            x = x * mask4d
        x = masked_pointwise(x, joint_mask, lambda h: self.ffn(self.norm2(h)), residual=True)
        if joint_mask is not None:
            x = x * mask4d
        return x


class WaveletDetailTemporalBranch(nn.Module):
    """FreePose-style local model for Haar detail coefficients."""

    def __init__(
        self,
        dim,
        dropout=0.1,
        ff_mult=4,
        max_radius=4,
    ):
        super().__init__()
        if (
            isinstance(max_radius, bool)
            or not isinstance(max_radius, int)
            or max_radius <= 1
        ):
            raise ValueError("max_radius must be an integer greater than 1")
        self.kernel_sizes = tuple(range(3, 2 * max_radius, 2))
        self.norm1 = nn.LayerNorm(dim)
        self.local_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=dim,
                )
                for kernel in self.kernel_sizes
            ]
        )
        self.fuse = nn.Linear(len(self.kernel_sizes) * dim, dim)
        self.gate = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, joint_mask=None):
        batch, frames, joints, dim = x.shape
        if joint_mask is None:
            token_mask = torch.ones(
                batch, frames, joints, dtype=torch.bool, device=x.device
            )
        elif joint_mask.dim() == 2:
            token_mask = joint_mask.bool().unsqueeze(1).expand(-1, frames, -1)
        else:
            token_mask = joint_mask.bool()

        h = self.norm1(x)
        h_bjct = (
            h.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch * joints, dim, frames)
        )
        local = [conv(h_bjct) for conv in self.local_convs]
        local = torch.cat(local, dim=1)
        local = (
            local.reshape(
                batch, joints, len(self.kernel_sizes) * dim, frames
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        local = self.fuse(local)
        gate = self.gate(h_bjct)
        gate = (
            gate.reshape(batch, joints, dim, frames)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        x = x + self.dropout(local * gate)
        x = x * token_mask.unsqueeze(-1).to(x.dtype)
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x * token_mask.unsqueeze(-1).to(x.dtype)


class WaveletDualTemporalPerJointTransformerBlock(nn.Module):
    """Haar-wavelet dual branch for temporal feature coefficients.

    The approximation branch is global self-attention plus FFN. The detail
    branch is multi-scale local temporal convolution, learned gating, and FFN.
    Their coefficient updates are recombined by the exact inverse Haar
    transform before the identity residual.
    """

    def __init__(
        self,
        dim,
        nheads=8,
        dropout=0.1,
        ff_mult=4,
        local_window=4,
        global_pool=2,
    ):
        super().__init__()
        if global_pool != 2:
            raise ValueError(
                "The Haar wavelet block uses a fixed decimation factor of 2; "
                f"got global_pool={global_pool}"
            )
        self.global_pool = 2
        self.low_branch = TemporalPerJointTransformerBlock(
            dim=dim,
            nheads=nheads,
            dropout=dropout,
            ff_mult=ff_mult,
            temporal_window=None,
            use_temporal_bias=False,
        )
        self.high_branch = WaveletDetailTemporalBranch(
            dim=dim,
            dropout=dropout,
            ff_mult=ff_mult,
            max_radius=local_window,
        )

    @staticmethod
    def _expand_mask(x, joint_mask):
        batch, frames, joints = x.shape[:3]
        if joint_mask is None:
            return torch.ones(
                batch,
                frames,
                joints,
                dtype=torch.bool,
                device=x.device,
            )
        if joint_mask.dim() == 2:
            return joint_mask.bool().unsqueeze(1).expand(-1, frames, -1)
        if joint_mask.dim() == 3:
            return joint_mask.bool()
        raise ValueError(
            "joint_mask must have shape [B,J] or [B,T,J], "
            f"got {tuple(joint_mask.shape)}"
        )

    @staticmethod
    def _haar_analysis(x, token_mask):
        even = x[:, 0::2]
        even_mask = token_mask[:, 0::2]
        odd = x[:, 1::2]
        odd_mask = token_mask[:, 1::2]
        if odd.shape[1] < even.shape[1]:
            odd = F.pad(odd, (0, 0, 0, 0, 0, 1))
            odd_mask = F.pad(odd_mask, (0, 0, 0, 1), value=False)

        both = even_mask & odd_mask
        even_only = even_mask & ~odd_mask
        odd_only = odd_mask & ~even_mask
        inv_sqrt2 = 2.0 ** -0.5
        approximation = (even + odd) * inv_sqrt2
        detail = (even - odd) * inv_sqrt2
        approximation = torch.where(
            even_only.unsqueeze(-1), even, approximation
        )
        approximation = torch.where(
            odd_only.unsqueeze(-1), odd, approximation
        )
        approximation_mask = even_mask | odd_mask
        approximation = approximation * approximation_mask.unsqueeze(-1).to(
            x.dtype
        )
        detail = detail * both.unsqueeze(-1).to(x.dtype)
        return approximation, detail, approximation_mask, both

    @staticmethod
    def _haar_synthesis(
        approximation,
        detail,
        token_mask,
        output_frames,
    ):
        even_mask = token_mask[:, 0::2]
        odd_mask = token_mask[:, 1::2]
        if odd_mask.shape[1] < even_mask.shape[1]:
            odd_mask = F.pad(odd_mask, (0, 0, 0, 1), value=False)
        even_only = even_mask & ~odd_mask
        odd_only = odd_mask & ~even_mask
        inv_sqrt2 = 2.0 ** -0.5
        even = (approximation + detail) * inv_sqrt2
        odd = (approximation - detail) * inv_sqrt2
        even = torch.where(even_only.unsqueeze(-1), approximation, even)
        odd = torch.where(odd_only.unsqueeze(-1), approximation, odd)
        even = even * even_mask.unsqueeze(-1).to(even.dtype)
        odd = odd * odd_mask.unsqueeze(-1).to(odd.dtype)
        output = approximation.new_zeros(
            approximation.shape[0],
            approximation.shape[1] * 2,
            approximation.shape[2],
            approximation.shape[3],
        )
        output[:, 0::2] = even
        output[:, 1::2] = odd
        return output[:, :output_frames]

    def forward(self, x, joint_mask=None):
        token_mask = self._expand_mask(x, joint_mask)
        low_input, high_input, low_mask, high_mask = self._haar_analysis(
            x,
            token_mask,
        )

        low_output = self.low_branch(
            low_input,
            joint_mask=low_mask,
        )
        high_output = self.high_branch(
            high_input,
            joint_mask=high_mask,
        )
        update = self._haar_synthesis(
            low_output - low_input,
            high_output - high_input,
            token_mask,
            x.shape[1],
        )
        output = x + update
        return output * token_mask.unsqueeze(-1).to(output.dtype)


DualRateTemporalPerJointTransformerBlock = (
    WaveletDualTemporalPerJointTransformerBlock
)
