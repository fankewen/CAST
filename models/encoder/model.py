### model.py ###
import torch
import torch.nn as nn

from .ref_encoder import RefQueryEncoder
from .temporal_trunk import EncoderTemporalModel


class Encoder(nn.Module):
    """
    Full Encoder v3 model: reference-frame query encoder + temporal trunk.

    The reference encoder fuses the reference pose (+ optional joint text
    embedding) into per-joint query features.
    The temporal trunk consumes those queries together with per-frame
    image embeddings and predicts per-frame 3D joint positions.

    The reset reference pose is used only to construct attention queries. It
    is never copied or added directly to the predicted positions.
    """

    def __init__(
        self,
        num_layers=12,
        q_dim=256,
        img_dim=1024,
        num_joints=150,
        num_heads=8,
        ref_layers=4,
        temporal_window=2,
        temporal_mode="global",
        causal=False,
        causal_chunk_size=1,
        dual_local_window=4,
        dual_global_pool=2,
        use_graph_ref_inner=False,
        use_graph_temporal_inner=False,
        use_checkpoint=False,
        dropout=0.1,
        output_mode="pose",
        concat_last_stage_features=False,
        use_mask_frame_token=False,
    ):
        super().__init__()

        self.use_mask_frame_token = bool(use_mask_frame_token)
        if self.use_mask_frame_token:
            # One learned visual placeholder is broadcast over patch tokens;
            # the temporal/graph positional structure remains unchanged.
            self.mask_frame_token = nn.Parameter(torch.zeros(1, 1, 1, img_dim))
            nn.init.normal_(self.mask_frame_token, mean=0.0, std=0.02)

        self.ref_encoder = RefQueryEncoder(
            q_dim=q_dim,
            num_heads=num_heads,
            num_layers=ref_layers,
            use_graph_ref_inner=use_graph_ref_inner,
            dropout=dropout,
        )

        self.temporal_model = EncoderTemporalModel(
            num_layers=num_layers,
            q_dim=q_dim,
            img_dim=img_dim,
            num_joints=num_joints,
            num_heads=num_heads,
            temporal_window=temporal_window,
            temporal_mode=temporal_mode,
            causal=causal,
            causal_chunk_size=causal_chunk_size,
            dual_local_window=dual_local_window,
            dual_global_pool=dual_global_pool,
            use_graph_temporal_inner=use_graph_temporal_inner,
            use_checkpoint=use_checkpoint,
            dropout=dropout,
            output_mode=output_mode,
            concat_last_stage_features=concat_last_stage_features,
        )
        self.output_feature_dim = self.temporal_model.output_feature_dim

    def forward(self, batch):
        image_embed = batch["image_embed"]               # [B,F,P,img_dim]
        observed_mask = batch.get("frame_observed_mask")
        if observed_mask is not None:
            observed_mask = observed_mask.bool()
            if observed_mask.shape != image_embed.shape[:2]:
                raise ValueError(
                    "frame_observed_mask must have shape [B,T], got "
                    f"{tuple(observed_mask.shape)} for {tuple(image_embed.shape)}"
                )
            # Keep temporal slots and positional encoding intact. Hidden
            # frames receive a learned [MASK_FRAME] embedding instead of a
            # zero vector, so the model can distinguish masking from silence.
            if self.use_mask_frame_token:
                token = self.mask_frame_token.to(dtype=image_embed.dtype)
                image_embed = torch.where(
                    observed_mask[:, :, None, None], image_embed,
                    token.expand(image_embed.shape[0], image_embed.shape[1], image_embed.shape[2], -1),
                )
            else:
                image_embed = image_embed * observed_mask[:, :, None, None].to(image_embed.dtype)
        ref_pos = batch["ref_position"]                  # [B,J,3]
        joint_mask = batch["joint_mask"].bool()          # [B,J]
        graph_hop = batch["graph_hop"]                   # [B,J,J]
        graph_edge = batch["graph_edge"]                 # [B,J,J]
        ancestor_mask = batch["ancestor_mask"].bool()    # [B,J,J]
        frame_mask = batch.get("frame_valid_mask")       # [B,F], optional

        ref_query = self.ref_encoder(
            ref_position=ref_pos,
            joint_mask=joint_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
            tree_mask=ancestor_mask,
        )

        F = image_embed.shape[1]

        joint_mask_t = joint_mask.unsqueeze(1).expand(-1, F, -1)
        if frame_mask is not None:
            joint_mask_t = joint_mask_t & frame_mask.bool().unsqueeze(-1)
        graph_hop_t = graph_hop.unsqueeze(1).expand(-1, F, -1, -1)
        graph_edge_t = graph_edge.unsqueeze(1).expand(-1, F, -1, -1)
        ancestor_mask_t = ancestor_mask.unsqueeze(1).expand(-1, F, -1, -1)

        pose_pred = self.temporal_model(
            ref_query=ref_query,
            cond_img=image_embed,
            joint_mask=joint_mask_t,
            graph_hop=graph_hop_t,
            graph_edge=graph_edge_t,
            tree_mask=ancestor_mask_t,
        )

        pose_pred = pose_pred * joint_mask_t.unsqueeze(-1).float()
        return pose_pred

    @torch.no_grad()
    def init_streaming_state(self, batch, *, max_length=None):
        """Encode static reference inputs once and allocate one cache per block."""
        if not all(hasattr(block.temporal, "forward_streaming") for block in self.temporal_model.blocks):
            raise RuntimeError("streaming requires temporal_mode='global' and causal=True")
        if not all(getattr(block.temporal.attn, "causal", False) for block in self.temporal_model.blocks):
            raise RuntimeError("KV cache requires a causal model")
        joint_mask = batch["joint_mask"].bool()
        ref_query = self.ref_encoder(
            ref_position=batch["ref_position"],
            joint_mask=joint_mask,
            graph_hop=batch["graph_hop"],
            graph_edge=batch["graph_edge"],
            tree_mask=batch["ancestor_mask"].bool(),
        )
        return {
            "ref_query": ref_query,
            "temporal": self.temporal_model.init_streaming_state(
                max_length=max_length, joint_mask=joint_mask,
                graph_hop=batch.get("graph_hop"), graph_edge=batch.get("graph_edge")
            ),
        }

    @torch.no_grad()
    def forward_streaming(self, batch, state):
        image_embed = batch["image_embed"]
        if image_embed.ndim != 4 or image_embed.shape[1] != 1:
            raise ValueError("streaming batch['image_embed'] must be [B,1,P,D]")
        joint_mask = batch["joint_mask"].bool()
        out, _ = self.temporal_model.forward_streaming(
            ref_query=state["ref_query"],
            cond_img=image_embed,
            joint_mask=joint_mask,
            graph_hop=batch.get("graph_hop"),
            graph_edge=batch.get("graph_edge"),
            tree_mask=batch.get("ancestor_mask"),
            state=state["temporal"],
        )
        return out * joint_mask[:, None, :, None].to(out.dtype), state
