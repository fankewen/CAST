from models.attention_layout import masked_pointwise
from models.kv_cache import TemporalKVCache
import torch.nn as nn

from .film import FiLMCondition
from .graph_attention import GraphMultiHeadAttention
from .temporal_attention import (
    TemporalPerJointTransformerBlock,
    WaveletDualTemporalPerJointTransformerBlock,
)


# =========================================================
# Rot Decoder
# =========================================================
class RotDecoderBlock(nn.Module):
    def __init__(
        self,
        q_dim=256,
        num_heads=8,
        dropout=0.1,
        temporal_window=2,
        temporal_mode="global",
        causal=False,
        causal_chunk_size=1,
        dual_local_window=4,
        dual_global_pool=2,
        use_tree_mask=False,
        use_rest_film=True,
    ):
        super().__init__()
        self.use_rest_film = use_rest_film

        self.film = FiLMCondition(q_dim) if use_rest_film else None
        if causal_chunk_size > 1 and (not causal or temporal_mode != "global"):
            raise ValueError("Chunk causal attention requires causal=True and temporal_mode='global'")
        if temporal_mode == "global":
            self.temporal = TemporalPerJointTransformerBlock(
                dim=q_dim,
                nheads=num_heads,
                dropout=dropout,
                ff_mult=4,
                temporal_window=temporal_window,
                use_temporal_bias=True,
                causal=causal,
                causal_chunk_size=causal_chunk_size,
            )
        elif temporal_mode in {"wavelet_dual", "dual_rate"}:
            self.temporal = WaveletDualTemporalPerJointTransformerBlock(
                dim=q_dim,
                nheads=num_heads,
                dropout=dropout,
                ff_mult=4,
                local_window=dual_local_window,
                global_pool=dual_global_pool,
            )
        else:
            raise ValueError(
                "temporal_mode must be 'global' or 'wavelet_dual', "
                f"got {temporal_mode!r}"
            )
        self.graph_norm = nn.LayerNorm(q_dim)
        self.graph = GraphMultiHeadAttention(
            q_dim,
            num_heads,
            dropout=dropout,
            use_tree_mask=use_tree_mask,
        )
        self.ffn_norm = nn.LayerNorm(q_dim)
        self.ffn = nn.Sequential(
            nn.Linear(q_dim, q_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(q_dim * 4, q_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,               # [B,T,J,D]
        rest_t,          # [B,T,J,D]
        joint_mask,      # [B,J]
        ancestor_mask,      # [B,J,J]
        graph_hop,       # [B,J,J]
        graph_edge,      # [B,J,J]
        temporal_joint_mask=None,  # [B,T,J]
        graph_length_groups=None,
    ):
        B, T, J, D = x.shape
        joint_mask_bt = (
            temporal_joint_mask
            if temporal_joint_mask is not None
            else joint_mask.unsqueeze(1).expand(-1, T, -1)
        )
        if self.film is not None:
            x = self.film(x, rest_t)
            x = x * joint_mask_bt.unsqueeze(-1).float()

        x = self.temporal(
            x,
            joint_mask=joint_mask_bt,
        )

        x2 = masked_pointwise(x, joint_mask_bt, self.graph_norm)
        x2 = x2.reshape(B * T, J, D)
        jm = joint_mask_bt.reshape(B * T, J)
        gm = ancestor_mask.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)
        gh = graph_hop.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)
        ge = graph_edge.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)

        x2 = self.graph(
            x2, x2, x2,
            gh, ge,
            mask=jm,
            tree_mask=gm,
            length_groups=graph_length_groups,
        )
        x = x + x2.reshape(B, T, J, D)
        x = x * joint_mask_bt.unsqueeze(-1).float()

        x = masked_pointwise(
            x, joint_mask_bt, lambda h: self.ffn(self.ffn_norm(h)), residual=True
        )
        x = x * joint_mask_bt.unsqueeze(-1).float()
        return x

    def forward_streaming(
        self, x, rest_t, joint_mask, ancestor_mask, graph_hop, graph_edge,
        cache: TemporalKVCache, graph_length_groups=None,
        relation_cache=None,
    ):
        if x.shape[1] != 1:
            raise ValueError("streaming decoder blocks accept one frame")
        joint_mask_bt = joint_mask if joint_mask.dim() == 3 else joint_mask.unsqueeze(1)
        if self.film is not None:
            x = self.film(x, rest_t)
            x = x * joint_mask_bt.unsqueeze(-1).to(x.dtype)
        x = self.temporal.forward_streaming(x, joint_mask=joint_mask_bt, cache=cache)
        B, _, J, D = x.shape
        x2 = masked_pointwise(x, joint_mask_bt, self.graph_norm).reshape(B, J, D)
        jm = joint_mask_bt[:, 0]
        x2 = self.graph(
            x2, x2, x2, graph_hop, graph_edge,
            mask=jm, tree_mask=ancestor_mask, length_groups=graph_length_groups,
            relation_cache=relation_cache,
        )
        x = (x + x2.unsqueeze(1)) * joint_mask_bt.unsqueeze(-1).to(x.dtype)
        x = masked_pointwise(x, joint_mask_bt, lambda h: self.ffn(self.ffn_norm(h)), residual=True)
        return x * joint_mask_bt.unsqueeze(-1).to(x.dtype)
