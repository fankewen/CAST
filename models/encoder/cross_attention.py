from models.attention_layout import masked_pointwise, active_token_rows
### cross_attention.py ###
import torch
import torch.nn as nn

from models.flash_backend import flash_mha_attention


class SimpleSelfAttention(nn.Module):
    """
    Pre-norm multi-head self-attention over joints, with padding-joint masking.
    Input: [B,J,D] -> Output: [B,J,D]
    """

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.last_backend = "uninitialized"

    def forward(self, x, joint_mask=None):
        h = masked_pointwise(x, joint_mask, self.norm)
        key_padding_mask = None
        if joint_mask is not None:
            key_padding_mask = ~joint_mask.bool()

        token_mask = (
            joint_mask.bool()
            if joint_mask is not None
            else torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        )
        out = flash_mha_attention(
            h,
            h,
            h,
            token_mask,
            token_mask,
            self.attn,
            dropout_p=self.attn.dropout if self.training else 0.0,
            post_projection=self.out,
        )
        if out is None:
            out, _ = self.attn(
                query=h,
                key=h,
                value=h,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            self.last_backend = "torch_mha"
        else:
            self.last_backend = "flash_attn"
        if self.last_backend != "flash_attn":
            out = self.out(out)
        out = self.dropout(out)
        x = x + out

        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()
        return x


class JointImageCrossAttention(nn.Module):
    """
    Cross-attention from joint queries to image patch features.

    Supports both the static case (query [B,J,D], kv [B,P,D]) and the temporal
    case (query [B,T,J,D], kv [B,T,P,D]) by folding time into the batch
    dimension in the temporal case.
    """

    def __init__(self, d_model, nheads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_backend = "uninitialized"

    def forward(self, query_feat, image_feat, joint_mask=None):
        if query_feat.dim() == 3:
            q = masked_pointwise(query_feat, joint_mask, self.norm_q)
            kv = masked_pointwise(image_feat, active_token_rows(joint_mask), self.norm_kv)

            query_mask = (
                joint_mask.bool()
                if joint_mask is not None
                else torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
            )
            kv_mask = None  # All image tokens are valid; cache by query mask.
            out = flash_mha_attention(
                q,
                kv,
                kv,
                query_mask,
                kv_mask,
                self.attn,
                dropout_p=self.attn.dropout if self.training else 0.0,
                post_projection=self.out_proj,
            )
            if out is None:
                out, _ = self.attn(
                    query=q,
                    key=kv,
                    value=kv,
                    need_weights=False,
                )
                self.last_backend = "torch_mha"
            else:
                self.last_backend = "flash_attn"
            if self.last_backend != "flash_attn":
                out = self.out_proj(out)
            out = self.dropout(out)
            out = query_feat + out

            if joint_mask is not None:
                out = out * joint_mask.unsqueeze(-1).float()
            return out

        if query_feat.dim() == 4:
            B, T, J, D = query_feat.shape
            P = image_feat.shape[2]

            q = masked_pointwise(query_feat, joint_mask, self.norm_q).reshape(B * T, J, D)
            kv = masked_pointwise(image_feat, active_token_rows(joint_mask), self.norm_kv).reshape(B * T, P, D)

            if joint_mask is None:
                query_mask = torch.ones(
                    B * T, J, dtype=torch.bool, device=q.device
                )
            elif joint_mask.dim() == 2:
                query_mask = joint_mask[:, None, :].expand(
                    -1, T, -1
                ).reshape(B * T, J).bool()
            else:
                query_mask = joint_mask.reshape(B * T, J).bool()
            kv_mask = None  # All image tokens are valid; cache by query mask.
            out = flash_mha_attention(
                q,
                kv,
                kv,
                query_mask,
                kv_mask,
                self.attn,
                dropout_p=self.attn.dropout if self.training else 0.0,
                post_projection=self.out_proj,
            )
            if out is None:
                out, _ = self.attn(
                    query=q,
                    key=kv,
                    value=kv,
                    need_weights=False,
                )
                self.last_backend = "torch_mha"
            else:
                self.last_backend = "flash_attn"
            if self.last_backend != "flash_attn":
                out = self.out_proj(out)
            out = self.dropout(out)
            out = out.reshape(B, T, J, D)
            out = query_feat + out

            if joint_mask is not None:
                if joint_mask.dim() == 2:
                    out = out * joint_mask.unsqueeze(1).unsqueeze(-1).float()
                else:
                    out = out * joint_mask.unsqueeze(-1).float()
            return out

        raise ValueError(f"Unsupported query_feat dim: {query_feat.dim()}")
