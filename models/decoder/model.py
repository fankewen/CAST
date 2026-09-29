"""Canonical Decoder implementation.

This module is intentionally the single Decoder implementation used by
video2motion.  It keeps the reset/rest geometry conditioning and has no
reset-memory encoder or memory cross-attention.  VisualGate is composed by
``BaseModel`` around this base model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from models.attention_layout import checkpoint, masked_pointwise

from .attention_blocks import *  # noqa: F401,F403
from models.graph_attention_layout import build_joint_length_groups
from utils.rotation import rot6d_to_rotmat_tensor, rotmat_to_rot6d_tensor

class RestEncoder(nn.Module):
    def __init__(
        self,
        q_dim=256,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        use_grad_checkpoint=False,
    ):
        super().__init__()
        self.offset_proj = nn.Linear(3, q_dim)
        self.use_grad_checkpoint = use_grad_checkpoint

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(q_dim),
                "graph": GraphMultiHeadAttention(
                    q_dim,
                    num_heads,
                    dropout=dropout,
                    use_tree_mask=(layer_idx % 2 == 0),
                ),
                "ffn_norm": nn.LayerNorm(q_dim),
                "ffn": nn.Sequential(
                    nn.Linear(q_dim, q_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(q_dim * 4, q_dim),
                    nn.Dropout(dropout),
                )
            })
            for layer_idx in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(q_dim)

    def _run_block(self, blk, x, joint_mask, ancestor_mask, graph_hop, graph_edge):
        h = masked_pointwise(x, joint_mask, blk["norm"])
        h = blk["graph"](
            h, h, h,
            graph_hop, graph_edge,
            mask=joint_mask,
            tree_mask=ancestor_mask,
        )
        x = x + h
        x = x * joint_mask.unsqueeze(-1).float()

        x = masked_pointwise(x, joint_mask, lambda h: blk["ffn"](blk["ffn_norm"](h)), residual=True)
        x = x * joint_mask.unsqueeze(-1).float()
        return x

    def forward(
        self,
        offset,         # [B,J,3]
        joint_mask,     # [B,J]
        ancestor_mask,     # [B,J,J]
        graph_hop,      # [B,J,J]
        graph_edge,     # [B,J,J]
    ):
        x = self.offset_proj(offset)
        x = x * joint_mask.unsqueeze(-1).float()

        for blk in self.layers:
            if self.use_grad_checkpoint and self.training:
                def custom_forward(x_in, block=blk):
                    return self._run_block(block, x_in, joint_mask, ancestor_mask, graph_hop, graph_edge)
                x = checkpoint(custom_forward, x, use_reentrant=False)
            else:
                x = self._run_block(blk, x, joint_mask, ancestor_mask, graph_hop, graph_edge)

        x = self.out_norm(x)
        x = x * joint_mask.unsqueeze(-1).float()
        return x

class RotDecoder(nn.Module):
    def __init__(
        self,
        q_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        temporal_window=2,
        temporal_mode="global",
        causal=False,
        causal_chunk_size=1,
        dual_local_window=4,
        dual_global_pool=2,
        use_rest_film=True,
        use_grad_checkpoint=False,
        output_dim=6,
    ):
        super().__init__()
        self.use_rest_film = use_rest_film
        self.use_grad_checkpoint = use_grad_checkpoint
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError("RotDecoder output_dim must be positive")


        self.blocks = nn.ModuleList([
            RotDecoderBlock(
                q_dim=q_dim,
                num_heads=num_heads,
                dropout=dropout,
                temporal_window=temporal_window,
                temporal_mode=temporal_mode,
                causal=causal,
                causal_chunk_size=causal_chunk_size,
                dual_local_window=dual_local_window,
                dual_global_pool=dual_global_pool,
                use_tree_mask=(layer_idx % 2 == 0),
                use_rest_film=use_rest_film,
            )
            for layer_idx in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(q_dim),
            nn.Linear(q_dim, q_dim),
            nn.GELU(),
            nn.Linear(q_dim, self.output_dim),
        )

    def _run_block(
        self,
        blk,
        x,
        rest_t,
        joint_mask,
        ancestor_mask,
        graph_hop,
        graph_edge,
        temporal_joint_mask,
        graph_length_groups,
    ):
        return blk(
            x=x,
            rest_t=rest_t,
            joint_mask=joint_mask,
            ancestor_mask=ancestor_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
            temporal_joint_mask=temporal_joint_mask,
            graph_length_groups=graph_length_groups,
        )

    def forward(
        self,
        query_feat,      # [B,T,J,D]
        rest_embed,      # [B,J,D]
        joint_mask,      # [B,J]
        ancestor_mask,      # [B,J,J]
        graph_hop,       # [B,J,J]
        graph_edge,      # [B,J,J]
        frame_mask=None,  # [B,T]
        return_features=False,
    ):
        B, T, J, D = query_feat.shape
        rest_t = rest_embed.unsqueeze(1).expand(-1, T, -1, -1)
        x = query_feat

        temporal_joint_mask = joint_mask.unsqueeze(1).expand(-1, T, -1)
        if frame_mask is not None:
            temporal_joint_mask = temporal_joint_mask & frame_mask.bool().unsqueeze(-1)
        x = x * temporal_joint_mask.unsqueeze(-1).float()
        graph_length_groups = build_joint_length_groups(temporal_joint_mask)

        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training:
                def custom_forward(x_in, block=blk):
                    return self._run_block(
                        block, x_in, rest_t,
                        joint_mask, ancestor_mask, graph_hop, graph_edge,
                        temporal_joint_mask,
                        graph_length_groups,
                    )
                x = checkpoint(custom_forward, x, use_reentrant=False)
            else:
                x = self._run_block(
                    blk, x, rest_t,
                    joint_mask, ancestor_mask, graph_hop, graph_edge,
                    temporal_joint_mask,
                    graph_length_groups,
                )

        out = self.head(x.reshape(B * T, J, D)).reshape(
            B, T, J, self.output_dim
        )
        out = out * temporal_joint_mask.unsqueeze(-1).float()
        if return_features:
            return out, x
        return out

    def init_streaming_state(self, *, max_length=None, joint_mask=None,
                             graph_hop=None, graph_edge=None):
        from models.kv_cache import TemporalKVCache

        if not all(hasattr(blk.temporal, "forward_streaming") for blk in self.blocks):
            raise RuntimeError("streaming RotDecoder requires global temporal blocks")
        graph_length_groups = (
            build_joint_length_groups(joint_mask) if joint_mask is not None else None
        )
        relation_caches = None
        if graph_hop is not None and graph_edge is not None:
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
        self, query_feat, rest_embed, joint_mask,
        ancestor_mask, graph_hop, graph_edge, state, return_features=False,
    ):
        if query_feat.ndim != 4 or query_feat.shape[1] != 1:
            raise ValueError("query_feat must have shape [B,1,J,D]")
        B, _, J, D = query_feat.shape
        rest_t = rest_embed.unsqueeze(1)
        x = query_feat
        jm = joint_mask.unsqueeze(1)
        x = x * jm.unsqueeze(-1).to(x.dtype)
        groups = state.get("graph_length_groups")
        if groups is None:
            groups = build_joint_length_groups(joint_mask)
        relation_caches = state.get("graph_relation_caches")
        for block_index, (blk, cache) in enumerate(zip(self.blocks, state["caches"])):
            x = blk.forward_streaming(
                x=x, rest_t=rest_t, joint_mask=joint_mask,
                ancestor_mask=ancestor_mask, graph_hop=graph_hop, graph_edge=graph_edge,
                cache=cache, graph_length_groups=groups,
                relation_cache=(relation_caches[block_index] if relation_caches is not None else None),
            )
        out = self.head(x.reshape(B, J, D)).reshape(B, 1, J, self.output_dim)
        out = out * jm.unsqueeze(-1).to(out.dtype)
        if return_features:
            return out, x
        return out


def _normalize_quaternion_or_identity(
    quaternion: torch.Tensor,
    min_norm: float = 1e-4,
) -> torch.Tensor:
    """Normalize a quaternion without an unbounded gradient near zero."""
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    normalized = quaternion / norm.clamp_min(min_norm)
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1.0
    return torch.where(norm > min_norm, normalized, identity)


class RestGeometryEncoder(nn.Module):
    """Encode reset geometry and joint semantics for decoder AdaLN.

    Inputs are the dataset's original-scale parent-local offset and reset local
    rotation. ``metric_scale`` is deliberately
    excluded from model conditioning.  The optional surface-anchor loss may
    apply 2 m normalization outside the model, but changing this value can
    never change a forward prediction. Graph blocks propagate this static
    geometry/semantic condition over the skeleton once.
    """

    def __init__(
        self,
        q_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        use_grad_checkpoint: bool,
    ):
        super().__init__()
        self.input_proj = nn.Linear(3 + 6, q_dim)
        self.use_grad_checkpoint = use_grad_checkpoint
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "graph_norm": nn.LayerNorm(q_dim),
                        "graph": GraphMultiHeadAttention(
                            q_dim,
                            num_heads,
                            dropout=dropout,
                            use_tree_mask=(layer_index % 2 == 0),
                        ),
                        "ffn_norm": nn.LayerNorm(q_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(q_dim, q_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(q_dim * 4, q_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                for layer_index in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(q_dim)

    @staticmethod
    def _run_block(
        block,
        x,
        joint_mask,
        ancestor_mask,
        graph_hop,
        graph_edge,
        graph_length_groups,
    ):
        h = masked_pointwise(x, joint_mask, block["graph_norm"])
        h = block["graph"](
            h,
            h,
            h,
            graph_hop,
            graph_edge,
            mask=joint_mask,
            tree_mask=ancestor_mask,
            length_groups=graph_length_groups,
        )
        x = x + h
        x = x * joint_mask.unsqueeze(-1).to(x.dtype)
        x = masked_pointwise(x, joint_mask, lambda h: block["ffn"](block["ffn_norm"](h)), residual=True)
        return x * joint_mask.unsqueeze(-1).to(x.dtype)

    def forward(
        self,
        *,
        offset,
        ref_rot6d,
        joint_mask,
        ancestor_mask,
        graph_hop,
        graph_edge,
    ):
        geometry = torch.cat(
            (
                offset,
                ref_rot6d.to(offset.dtype),
            ),
            dim=-1,
        )
        x = self.input_proj(geometry)
        x = x * joint_mask.unsqueeze(-1).to(x.dtype)
        graph_length_groups = build_joint_length_groups(joint_mask)
        for block in self.layers:
            if self.use_grad_checkpoint and self.training:
                def custom_forward(x_in, current_block=block):
                    return self._run_block(
                        current_block,
                        x_in,
                        joint_mask,
                        ancestor_mask,
                        graph_hop,
                        graph_edge,
                        graph_length_groups,
                    )
                x = checkpoint(
                    custom_forward,
                    x,
                    use_reentrant=False,
                )
            else:
                x = self._run_block(
                    block,
                    x,
                    joint_mask,
                    ancestor_mask,
                    graph_hop,
                    graph_edge,
                    graph_length_groups,
                )
        return (
            masked_pointwise(x, joint_mask, self.out_norm)
            * joint_mask.unsqueeze(-1).to(x.dtype)
        )


class Decoder(nn.Module):
    """Decoder with rest conditioning, but no reset-memory encoder/cross-attn."""

    def __init__(
        self,
        q_dim: int = 256,
        rest_layers: int = 2,
        decoder_layers: int = 4,
        num_heads: int = 8,
        temporal_window: int | None = 2,
        temporal_mode: str = "global",
        causal: bool = False,
        causal_chunk_size: int = 1,
        dual_local_window: int = 4,
        dual_global_pool: int = 2,
        temporal_dropout: float = 0.1,
        decoder_rest_film: bool = True,
        use_grad_checkpoint: bool = False,
        external_query_dim: int | None = None,
        predict_residual_rotation: bool = False,
        rotation_representation: str = "rot6d",
    ):
        super().__init__()
        self.external_query_dim = external_query_dim
        self.predict_residual_rotation = bool(predict_residual_rotation)
        if rotation_representation not in {"rot6d", "quaternion"}:
            raise ValueError(
                "rotation_representation must be 'rot6d' or 'quaternion'"
            )
        self.rotation_representation = rotation_representation
        self.rest_encoder = RestGeometryEncoder(
            q_dim=q_dim,
            num_heads=num_heads,
            num_layers=rest_layers,
            dropout=temporal_dropout,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        if isinstance(external_query_dim, bool) or not isinstance(
            external_query_dim, int
        ) or external_query_dim <= 0:
            raise ValueError("external_query_dim must be a positive integer")
        self.external_query_dim = external_query_dim
        self.external_query_proj = nn.Sequential(
            nn.LayerNorm(external_query_dim),
            nn.Linear(external_query_dim, q_dim),
        )

        # FiLM -> temporal attention/FFN -> graph attention/FFN.
        self.decoder = RotDecoder(
            q_dim=q_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=temporal_dropout,
            temporal_window=temporal_window,
            temporal_mode=temporal_mode,
            causal=causal,
            causal_chunk_size=causal_chunk_size,
            dual_local_window=dual_local_window,
            dual_global_pool=dual_global_pool,
            use_rest_film=decoder_rest_film,
            use_grad_checkpoint=use_grad_checkpoint,
            output_dim=4 if rotation_representation == "quaternion" else 6,
        )

        if self.predict_residual_rotation:
            final = self.decoder.head[-1]
            nn.init.zeros_(final.weight)
            with torch.no_grad():
                if rotation_representation == "quaternion":
                    final.bias.copy_(
                        torch.tensor([1.0, 0.0, 0.0, 0.0])
                    )
                else:
                    final.bias.copy_(
                        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                    )

    def forward(
        self,
        batch: dict,
        query_input: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        joint_mask = batch["joint_mask"].bool()
        ancestor_mask = batch["ancestor_mask"].bool()
        graph_hop = batch["graph_hop"]
        graph_edge = batch["graph_edge"]
        static_rot_joint_mask = batch["static_rot_joint_mask"].bool()
        frame_mask = batch.get("frame_valid_mask")

        if query_input.shape[-1] != self.external_query_dim:
            raise ValueError(
                f"query_input width must be {self.external_query_dim}, "
                f"got {query_input.shape[-1]}"
            )

        ref_rot6d = batch["ref_rot6d_a"]
        offset = batch["offset_a"]

        rest_embed = self.rest_encoder(
            offset=offset,
            ref_rot6d=ref_rot6d,
            joint_mask=joint_mask,
            ancestor_mask=ancestor_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
        )

        q_feat = self.external_query_proj(query_input)
        temporal_mask = joint_mask.unsqueeze(1)
        if frame_mask is not None:
            temporal_mask = temporal_mask & frame_mask.bool().unsqueeze(-1)
        q_feat = q_feat * temporal_mask.unsqueeze(-1).to(q_feat.dtype)

        pred_rotation, decoder_feat = self.decoder(
            query_feat=q_feat,
            rest_embed=rest_embed,
            joint_mask=joint_mask,
            ancestor_mask=ancestor_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
            frame_mask=frame_mask,
            return_features=True,
        )

        reset_matrix = rot6d_to_rotmat_tensor(ref_rot6d.float()).unsqueeze(1)
        static_mask = static_rot_joint_mask.unsqueeze(1).unsqueeze(-1)
        if self.rotation_representation == "rot6d":
            pred_rot6d = pred_rotation
            if self.predict_residual_rotation:
                residual_matrix = rot6d_to_rotmat_tensor(pred_rot6d.float())
                pred_rot6d = rotmat_to_rot6d_tensor(
                    reset_matrix @ residual_matrix
                )
        else:
            raise ValueError
        reset_rotation = ref_rot6d.unsqueeze(1).expand_as(pred_rot6d)
        pred_rot6d = pred_rot6d * (~static_mask) + reset_rotation * static_mask
        if frame_mask is not None:
            valid = frame_mask.bool().unsqueeze(-1).unsqueeze(-1)
            pred_rot6d = pred_rot6d * valid

        result = {
            "pred_rot6d": pred_rot6d,
            # Preserve the direct head output for numerical diagnostics. The
            # training objective still consumes decoded SO(3) rotations.
            "pred_rotation_raw": pred_rotation,
            "rest_embed": rest_embed,
            "q_feat": q_feat,
            # Final tokens after all Decoder decoder blocks, immediately
            # before the shared rotation output head.
            "decoder_feat": decoder_feat,
        }
        return result

    @torch.no_grad()
    def init_streaming_state(self, batch, *, max_length=None):
        joint_mask = batch["joint_mask"].bool()
        rest_embed = self.rest_encoder(
            offset=batch["offset_a"], ref_rot6d=batch["ref_rot6d_a"],
            joint_mask=joint_mask, ancestor_mask=batch["ancestor_mask"].bool(),
            graph_hop=batch["graph_hop"], graph_edge=batch["graph_edge"],
        )
        return {
            "rest_embed": rest_embed,
            "decoder": self.decoder.init_streaming_state(
                max_length=max_length, joint_mask=joint_mask,
                graph_hop=batch["graph_hop"], graph_edge=batch["graph_edge"]
            ),
        }

    @torch.no_grad()
    def forward_streaming(self, batch, state, *, query_input):
        if query_input.ndim != 4 or query_input.shape[1] != 1:
            raise ValueError("query_input must have shape [B,1,J,external_query_dim]")
        joint_mask = batch["joint_mask"].bool()
        if query_input.shape[-1] != self.external_query_dim:
            raise ValueError(f"query_input width must be {self.external_query_dim}")
        q_feat = self.external_query_proj(query_input)
        q_feat = q_feat * joint_mask[:, None, :, None].to(q_feat.dtype)
        pred_rotation, decoder_feat = self.decoder.forward_streaming(
            query_feat=q_feat, rest_embed=state["rest_embed"],
            joint_mask=joint_mask,
            ancestor_mask=batch["ancestor_mask"].bool(), graph_hop=batch["graph_hop"],
            graph_edge=batch["graph_edge"], state=state["decoder"], return_features=True,
        )
        reset_matrix = rot6d_to_rotmat_tensor(batch["ref_rot6d_a"].float()).unsqueeze(1)
        static_mask = batch["static_rot_joint_mask"].bool().unsqueeze(1).unsqueeze(-1)
        pred_rot6d = pred_rotation
        if self.predict_residual_rotation:
            pred_rot6d = rotmat_to_rot6d_tensor(reset_matrix @ rot6d_to_rotmat_tensor(pred_rot6d.float()))
        reset_rotation = batch["ref_rot6d_a"].unsqueeze(1).expand_as(pred_rot6d)
        pred_rot6d = pred_rot6d * (~static_mask) + reset_rotation * static_mask
        return {
            "pred_rot6d": pred_rot6d,
            "pred_rotation_raw": pred_rotation,
            "rest_embed": state["rest_embed"],
            "q_feat": q_feat,
            "decoder_feat": decoder_feat,
        }, state
