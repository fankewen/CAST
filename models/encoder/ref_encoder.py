from models.attention_layout import masked_pointwise
### ref_encoder.py ###
import torch
import torch.nn as nn

from .positional_embedding import FrequencyPositionalEmbedding
from .graph_attention import GraphMultiHeadAttention
from models.graph_attention_layout import build_joint_length_groups
from .cross_attention import SimpleSelfAttention


class FeedForwardResidual(nn.Module):
    """Independent pre-norm FFN residual paired with one attention sublayer."""

    def __init__(self, dim, dropout, mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, token_mask=None):
        # Packing already restores zeros at invalid tokens and broadcasts [B,J]
        # masks across time. Reapplying the raw mask here would break that case.
        return masked_pointwise(
            x, token_mask, lambda h: self.ffn(self.norm(h)), residual=True
        )


class RefFusionBlock(nn.Module):
    """
    One fusion layer for the reference-frame query encoder.

    Every attention sublayer owns an independent pre-norm FFN residual:
      graph attention -> graph FFN
      joint self-attention -> self-attention FFN
    """

    def __init__(
        self,
        q_dim=256,
        num_heads=8,
        dropout=0.1,
        use_tree_mask=True,
        use_graph=True,
    ):
        super().__init__()
        self.graph = (
            GraphMultiHeadAttention(
                q_dim,
                num_heads,
                dropout=dropout,
                use_tree_mask=use_tree_mask,
            )
            if use_graph
            else None
        )
        self.graph_norm = nn.LayerNorm(q_dim) if use_graph else None
        self.graph_ffn = (
            FeedForwardResidual(q_dim, dropout)
            if use_graph
            else None
        )
        self.self_attn = SimpleSelfAttention(q_dim, heads=num_heads, dropout=dropout)
        self.self_ffn = FeedForwardResidual(q_dim, dropout)

    def forward(
        self,
        x,
        joint_mask=None,
        graph_hop=None,
        graph_edge=None,
        tree_mask=None,
        graph_length_groups=None,
    ):
        if self.graph is not None:
            if graph_hop is None or graph_edge is None:
                raise ValueError(
                    "graph_hop and graph_edge are required when use_graph=True"
                )
            h = masked_pointwise(x, joint_mask, self.graph_norm)
            x2 = self.graph(
                h, h, h,
                graph_hop, graph_edge,
                mask=joint_mask,
                tree_mask=tree_mask,
                length_groups=graph_length_groups,
            )
            x = x + x2
            if joint_mask is not None:
                x = x * joint_mask.unsqueeze(-1).float()
            x = self.graph_ffn(x, joint_mask)

        x = self.self_attn(x, joint_mask=joint_mask)
        x = self.self_ffn(x, joint_mask)

        return x


class RefQueryEncoder(nn.Module):
    """
    Encodes the reference pose + reference image into a per-joint query
    embedding used by the downstream temporal model.

    Input:
      - ref_position     [B,J,3]
      - joint_mask       [B,J]
      - graph_hop        [B,J,J]
      - graph_edge       [B,J,J]
      - tree_mask        [B,J,J] (optional ancestor mask)

    Output:
      - per-joint query  [B,J,q_dim]
    """

    def __init__(
        self,
        q_dim=256,
        num_heads=8,
        num_layers=4,
        use_graph_ref_inner=False,
        dropout=0.1,
    ):
        super().__init__()
        self.pos_embedder = FrequencyPositionalEmbedding(num_freqs=8, input_dim=3)
        self.pose_proj = nn.Linear(self.pos_embedder.out_dim, q_dim)

        self.use_graph_ref_inner = use_graph_ref_inner

        self.fusion_blocks = nn.ModuleList([
            RefFusionBlock(
                q_dim=q_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_tree_mask=(i % 2 == 0),
                use_graph=self.use_graph_ref_inner,
            )
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(q_dim)

    def forward(
        self,
        ref_position,
        joint_mask=None,
        graph_hop=None,
        graph_edge=None,
        tree_mask=None,
    ):
        ref_position_enc = self.pos_embedder(ref_position)
        x = self.pose_proj(ref_position_enc)

        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()

        graph_length_groups = (
            build_joint_length_groups(joint_mask)
            if self.use_graph_ref_inner
            else None
        )

        for blk in self.fusion_blocks:
            x = blk(
                x,
                joint_mask=joint_mask,
                graph_hop=graph_hop if self.use_graph_ref_inner else None,
                graph_edge=graph_edge if self.use_graph_ref_inner else None,
                tree_mask=tree_mask if self.use_graph_ref_inner else None,
                graph_length_groups=graph_length_groups,
            )

        x = masked_pointwise(x, joint_mask, self.final_norm)
        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()
        return x
