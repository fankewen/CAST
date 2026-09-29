from __future__ import annotations

from models.attention_layout import masked_pointwise, active_token_rows
### temporal_trunk.py ###
import torch
import torch.nn as nn
from models.attention_layout import checkpoint

from .graph_attention import GraphMultiHeadAttention
from models.graph_attention_layout import build_joint_length_groups
from .temporal_attention import (
    TemporalPerJointTransformerBlock,
    WaveletDualTemporalPerJointTransformerBlock,
)
from .cross_attention import JointImageCrossAttention


class FeedForwardResidual(nn.Module):
    """Pre-norm residual FFN owned by one attention sublayer."""

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


class EncoderTemporalBlock(nn.Module):
    """
    One temporal trunk block:
      1) frame-wise joint-image cross-attention + its own FFN
      2) optional per-frame graph self-attention + its own FFN
      3) per-joint temporal self-attention + its own FFN

    Image evidence must be injected before temporal attention. The
    joint-image cross-attention is applied independently within each frame:
    joint tokens attend only to that frame's image patches. In the first block,
    ``x`` is a time-wise copy of one static reference query; applying temporal
    attention before this frame-wise visual update cannot model motion.
    """

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
        use_graph=True,
        use_tree_mask=True,
    ):
        super().__init__()
        self.use_graph = use_graph

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

        if use_graph:
            self.graph_norm = nn.LayerNorm(q_dim)
            self.graph = GraphMultiHeadAttention(
                q_dim,
                num_heads,
                dropout=dropout,
                use_tree_mask=use_tree_mask,
            )

        self.cross_img = JointImageCrossAttention(
            d_model=q_dim,
            nheads=num_heads,
            dropout=dropout,
        )
        self.cross_ffn = FeedForwardResidual(q_dim, dropout)
        if use_graph:
            self.graph_ffn = FeedForwardResidual(q_dim, dropout)

    def forward(
        self,
        x,
        img_feat,
        joint_mask,
        graph_hop=None,
        graph_edge=None,
        tree_mask=None,
        graph_length_groups=None,
        return_stage_features=False,
    ):
        """
        x       : [B,T,J,D]
        img_feat: [B,T,P,D]
        """
        B, T, J, D = x.shape

        # Inject distinct per-frame visual evidence first. This is especially
        # important in block 0, whose incoming reference queries are otherwise
        # identical at every time position.
        x = self.cross_img(x, img_feat, joint_mask=joint_mask)
        x = self.cross_ffn(x, joint_mask)
        rgb_feature = x

        if self.use_graph and graph_hop is not None:
            x2 = masked_pointwise(x, joint_mask, self.graph_norm).reshape(B * T, J, D)
            jm = joint_mask.reshape(B * T, J)
            gh = graph_hop.reshape(B * T, J, J)
            ge = graph_edge.reshape(B * T, J, J)
            tm = tree_mask.reshape(B * T, J, J) if tree_mask is not None else None

            x2 = self.graph(
                x2, x2, x2,
                gh, ge,
                mask=jm,
                tree_mask=tm,
                length_groups=graph_length_groups,
            )
            x = x + x2.reshape(B, T, J, D)
            x = x * joint_mask.unsqueeze(-1).float()
            x = self.graph_ffn(x, joint_mask)
        graph_feature = x

        # Global temporal attention now operates on visual, skeleton-aware
        # per-frame features in every block, including the first one. The
        # TemporalPerJointTransformerBlock owns the FFN for this attention.
        x = self.temporal(x, joint_mask=joint_mask)

        if return_stage_features:
            return rgb_feature, graph_feature, x
        return x

    def forward_streaming(
        self, x, img_feat, joint_mask, graph_hop=None, graph_edge=None,
        tree_mask=None, graph_length_groups=None, cache: TemporalKVCache | None = None,
        return_stage_features=False, relation_cache=None,
    ):
        """Process one frame and update this block's temporal KV cache."""
        if cache is None:
            raise ValueError("EncoderTemporalBlock requires a temporal cache")
        if not hasattr(self.temporal, "forward_streaming"):
            raise RuntimeError("streaming is only implemented for global temporal mode")
        B, T, J, D = x.shape
        if T != 1:
            raise ValueError("streaming blocks accept one frame at a time")
        x = self.cross_img(x, img_feat, joint_mask=joint_mask)
        x = self.cross_ffn(x, joint_mask)
        rgb_feature = x
        if self.use_graph and graph_hop is not None:
            x2 = masked_pointwise(x, joint_mask, self.graph_norm).reshape(B, J, D)
            jm = joint_mask.reshape(B, J)
            tm = tree_mask.reshape(B, J, J) if tree_mask is not None else None
            x2 = self.graph(x2, x2, x2, graph_hop, graph_edge, mask=jm,
                            tree_mask=tm, length_groups=graph_length_groups,
                            relation_cache=relation_cache)
            x = self.graph_ffn(x + x2.unsqueeze(1), joint_mask)
        graph_feature = x
        x = self.temporal.forward_streaming(x, joint_mask=joint_mask, cache=cache)
        if return_stage_features:
            return rgb_feature, graph_feature, x
        return x


class EncoderTemporalModel(nn.Module):
    """
    Encoder temporal trunk: stacks EncoderTemporalBlocks on top of the
    per-joint reference query, attending to per-frame image features and
    producing per-frame 3D joint positions.
    """

    def __init__(
        self,
        num_layers=12,
        q_dim=256,
        img_dim=1024,
        num_joints=150,
        num_heads=8,
        temporal_window=2,
        temporal_mode="global",
        causal=False,
        causal_chunk_size=1,
        dual_local_window=4,
        dual_global_pool=2,
        use_graph_temporal_inner=False,
        use_checkpoint=False,
        dropout=0.1,
        output_mode="pose",
        concat_last_stage_features=False,
    ):
        super().__init__()
        if output_mode not in {"pose", "features"}:
            raise ValueError(
                f"output_mode must be 'pose' or 'features', got {output_mode!r}"
            )
        self.output_mode = output_mode
        self.concat_last_stage_features = bool(concat_last_stage_features)
        if self.concat_last_stage_features and output_mode != "features":
            raise ValueError(
                "concat_last_stage_features requires output_mode='features'"
            )
        if self.concat_last_stage_features and not use_graph_temporal_inner:
            raise ValueError(
                "concat_last_stage_features requires "
                "use_graph_temporal_inner=True"
            )
        self.use_graph_temporal_inner = use_graph_temporal_inner
        self.use_checkpoint = use_checkpoint
        self.img_proj = nn.Linear(img_dim, q_dim)
        self.output_feature_dim = (
            q_dim * 3 if self.concat_last_stage_features else q_dim
        )

        self.blocks = nn.ModuleList([
            EncoderTemporalBlock(
                q_dim=q_dim,
                num_heads=num_heads,
                dropout=dropout,
                temporal_window=temporal_window,
                temporal_mode=temporal_mode,
                causal=causal,
                causal_chunk_size=causal_chunk_size,
                dual_local_window=dual_local_window,
                dual_global_pool=dual_global_pool,
                use_graph=use_graph_temporal_inner,
                use_tree_mask=(i % 2 == 0),
            )
            for i in range(num_layers)
        ])

        if output_mode == "pose":
            self.relative_pose_head = nn.Sequential(
                nn.LayerNorm(q_dim),
                nn.Linear(q_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
            )
            self.root_position_head = nn.Sequential(
                nn.LayerNorm(q_dim),
                nn.Linear(q_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
            )
            for head in (self.relative_pose_head, self.root_position_head):
                nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
                nn.init.zeros_(head[-1].bias)
        else:
            # Rotation-only cache training consumes the rich joint features
            # directly. Do not instantiate permanently unused position heads:
            # doing so both wastes parameters and violates DDP's used-parameter
            # contract when static_graph=True.
            self.relative_pose_head = None
            self.root_position_head = None

    def forward(
        self,
        ref_query,
        cond_img,
        joint_mask=None,
        graph_hop=None,
        graph_edge=None,
        tree_mask=None,
    ):
        """
        ref_query: [B,J,D]
        cond_img : [B,T,P,img_dim]
        """
        B, T, _, _ = cond_img.shape
        _, J, D = ref_query.shape

        x = ref_query.unsqueeze(1).expand(B, T, J, D).contiguous()
        img_feat = masked_pointwise(cond_img, active_token_rows(joint_mask), self.img_proj)

        graph_hop_in = graph_hop if self.use_graph_temporal_inner else None
        graph_edge_in = graph_edge if self.use_graph_temporal_inner else None
        tree_mask_in = tree_mask if self.use_graph_temporal_inner else None
        graph_length_groups = (
            build_joint_length_groups(joint_mask)
            if self.use_graph_temporal_inner
            else None
        )

        last_stage_features = None
        for block_index, blk in enumerate(self.blocks):
            return_stage_features = (
                self.concat_last_stage_features
                and block_index == len(self.blocks) - 1
            )
            if self.use_checkpoint and self.training:
                def custom_forward(
                    x_in,
                    block=blk,
                    capture_stage_features=return_stage_features,
                ):
                    return block(
                        x=x_in,
                        img_feat=img_feat,
                        joint_mask=joint_mask,
                        graph_hop=graph_hop_in,
                        graph_edge=graph_edge_in,
                        tree_mask=tree_mask_in,
                        graph_length_groups=graph_length_groups,
                        return_stage_features=capture_stage_features,
                    )
                block_output = checkpoint(
                    custom_forward,
                    x,
                    use_reentrant=False,
                )
            else:
                block_output = blk(
                    x=x,
                    img_feat=img_feat,
                    joint_mask=joint_mask,
                    graph_hop=graph_hop_in,
                    graph_edge=graph_edge_in,
                    tree_mask=tree_mask_in,
                    graph_length_groups=graph_length_groups,
                    return_stage_features=return_stage_features,
                )
            if return_stage_features:
                rgb_feature, graph_feature, x = block_output
                last_stage_features = (
                    rgb_feature,
                    graph_feature,
                    x,
                )
            else:
                x = block_output

        if self.output_mode == "features":
            if self.concat_last_stage_features:
                if last_stage_features is None:
                    raise RuntimeError(
                        "Last-stage features were not captured"
                    )
                return torch.cat(last_stage_features, dim=-1)
            return x

        relative_pose = self.relative_pose_head(x)
        relative_pose = relative_pose - relative_pose[:, :, 0:1]
        root_position = self.root_position_head(x[:, :, 0])
        return root_position.unsqueeze(2) + relative_pose

    def init_streaming_state(
        self, *, max_length: int | None = None, joint_mask: torch.Tensor | None = None,
        graph_hop: torch.Tensor | None = None, graph_edge: torch.Tensor | None = None
    ):
        from models.kv_cache import TemporalKVCache

        if not all(hasattr(block.temporal, "forward_streaming") for block in self.blocks):
            raise RuntimeError("KV streaming currently supports temporal_mode='global' only")
        graph_length_groups = (
            build_joint_length_groups(joint_mask)
            if self.use_graph_temporal_inner and joint_mask is not None
            else None
        )
        relation_caches = None
        if self.use_graph_temporal_inner and graph_hop is not None and graph_edge is not None:
            relation_caches = [
                blk.graph.build_streaming_relation_cache(graph_hop, graph_edge)
                for blk in self.blocks
            ]
        return {
            "caches": [TemporalKVCache(max_length=max_length) for _ in self.blocks],
            "graph_length_groups": graph_length_groups,
            "graph_relation_caches": relation_caches,
        }

    def forward_streaming(
        self, ref_query, cond_img, joint_mask=None, graph_hop=None,
        graph_edge=None, tree_mask=None, state=None, return_stage_features=False,
    ):
        if cond_img.ndim != 4 or cond_img.shape[1] != 1:
            raise ValueError("cond_img must have shape [B,1,P,img_dim]")
        if state is None:
            state = self.init_streaming_state()
        caches = state["caches"]
        if len(caches) != len(self.blocks):
            raise ValueError("streaming state does not match model depth")
        B, _, _, _ = cond_img.shape
        _, J, D = ref_query.shape
        x = ref_query.unsqueeze(1)
        img_feat = masked_pointwise(cond_img, active_token_rows(joint_mask), self.img_proj)
        graph_hop_in = graph_hop if self.use_graph_temporal_inner else None
        graph_edge_in = graph_edge if self.use_graph_temporal_inner else None
        tree_mask_in = tree_mask if self.use_graph_temporal_inner else None
        graph_length_groups = state.get("graph_length_groups")
        if graph_length_groups is None and self.use_graph_temporal_inner:
            graph_length_groups = build_joint_length_groups(joint_mask)
        last_stage_features = None
        relation_caches = state.get("graph_relation_caches")
        for block_index, (blk, cache) in enumerate(zip(self.blocks, caches)):
            return_stage_features = self.concat_last_stage_features and block_index == len(self.blocks) - 1
            block_output = blk.forward_streaming(
                x=x, img_feat=img_feat, joint_mask=joint_mask,
                graph_hop=graph_hop_in, graph_edge=graph_edge_in,
                tree_mask=tree_mask_in, graph_length_groups=graph_length_groups,
                cache=cache, return_stage_features=return_stage_features,
                relation_cache=(relation_caches[block_index] if relation_caches is not None else None),
            )
            if return_stage_features:
                rgb_feature, graph_feature, x = block_output
                last_stage_features = (rgb_feature, graph_feature, x)
            else:
                x = block_output
        if self.output_mode == "features":
            if self.concat_last_stage_features:
                return torch.cat(last_stage_features, dim=-1), state
            return x, state
        relative_pose = self.relative_pose_head(x)
        relative_pose = relative_pose - relative_pose[:, :, 0:1]
        root_position = self.root_position_head(x[:, :, 0])
        return root_position.unsqueeze(2) + relative_pose, state
