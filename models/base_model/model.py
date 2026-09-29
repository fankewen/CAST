from models.attention_layout import with_attention_layouts
from models.attention_layout import set_attention_layout_cache
import torch.nn as nn
import torch
import torch.nn.functional as F
import inspect
import copy
from typing import Optional, Dict, Any
from utils.config_utils import instantiate_from_config
from models.base_model.bone_direction_head import (
    canonical_fallback_axis,
    positions_to_bone_directions,
    reconstruct_positions_from_bone_directions,
)
from utils.skeleton_observability import DEFAULT_EDGE_EPSILON


def _rot6d_to_rotmat_columns(rot_6d: torch.Tensor) -> torch.Tensor:
    """Decode the repository's concat(first two matrix columns) convention."""

    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]
    eps = 1e-6
    a1_norm = torch.linalg.vector_norm(a1, dim=-1, keepdim=True)
    fallback_a1 = torch.zeros_like(a1)
    fallback_a1[..., 0] = 1.0
    b1 = torch.where(
        a1_norm > eps, a1 / a1_norm.clamp_min(eps), fallback_a1
    )
    a2_orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    a2_norm = torch.linalg.vector_norm(
        a2_orthogonal, dim=-1, keepdim=True
    )
    fallback_axis = F.one_hot(
        torch.argmin(torch.abs(b1), dim=-1), num_classes=3
    ).to(dtype=b1.dtype)
    fallback_b2 = fallback_axis - (fallback_axis * b1).sum(
        dim=-1, keepdim=True
    ) * b1
    fallback_b2 = F.normalize(fallback_b2, dim=-1, eps=eps)
    b2 = torch.where(
        a2_norm > eps,
        a2_orthogonal / a2_norm.clamp_min(eps),
        fallback_b2,
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _rotmat_to_rot6d_columns(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


@torch.compiler.disable
def _streaming_grace_correction(module, **kwargs):
    """Keep the custom in-place GRACE correction outside Dynamo as one break."""
    return module(**kwargs)


class AuxiliaryPoseHead(nn.Sequential):
    """Predict 3D joint positions directly from encoder features."""

    def __init__(self, feature_dim: int):
        super().__init__(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        nn.init.normal_(self[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self[-1].bias)


class BaseModel(nn.Module):
    def __init__(
        self,
        image_backbone_cfg: dict,
        # -------- encoder --------
        encoder_cfg: dict,
        # -------- decoder --------
        decoder_cfg: dict,
        rotation_query_source: str = "pose",
        auxiliary_pose_head: bool = False,
        auxiliary_pose_detach_features: bool = False,
        pose_fusion_cfg: Optional[dict] = None,
        bone_direction_head_cfg: Optional[dict] = None,
        grace_cfg: Optional[dict] = None,
        grace_source: str = "bone_direction",
        zero_rest_edge_fallback_axis: str = "x",
        include_zero_rest_edges: bool = False,
        edge_epsilon: float = DEFAULT_EDGE_EPSILON,
    ):
        super().__init__()

        edge_epsilon = float(edge_epsilon)
        if not 0.0 < edge_epsilon < float("inf"):
            raise ValueError("edge_epsilon must be finite and positive")
        self.edge_epsilon = edge_epsilon
        self.zero_rest_edge_fallback_axis = canonical_fallback_axis(
            zero_rest_edge_fallback_axis
        )
        if not isinstance(include_zero_rest_edges, bool):
            raise TypeError("include_zero_rest_edges must be a boolean")
        self.include_zero_rest_edges = include_zero_rest_edges

        valid_query_sources = {"pose", "video_features", "gt_pose_oracle"}
        if rotation_query_source not in valid_query_sources:
            raise ValueError(
                "rotation_query_source must be one of "
                f"{sorted(valid_query_sources)}, got {rotation_query_source!r}"
            )
        self.rotation_query_source = rotation_query_source

        # The oracle is an explicitly isolated Decoder experiment.  Do not
        # instantiate (or accidentally optimize) the RGB backbone/Encoder
        # path: exact GT pose is injected by the trainer under a dedicated key.
        if rotation_query_source == "gt_pose_oracle":
            self.image_backbone = None
            self.encoder = None
            self.encoder_accepts_attention_kwargs = False
        else:
            # Frozen DINOv3 is a real model submodule and runs online from RGB.
            # Its weights are loaded from the configured checkpoint, never updated.
            self.image_backbone = instantiate_from_config(image_backbone_cfg)

            # stage1: video -> pose
            self.encoder = instantiate_from_config(encoder_cfg)
            self.encoder_accepts_attention_kwargs = (
                "attention_kwargs"
                in inspect.signature(self.encoder.forward).parameters
            )

        # stage2: pose -> rot
        self.decoder = instantiate_from_config(decoder_cfg)
        if (
            rotation_query_source == "gt_pose_oracle"
            and getattr(self.decoder, "external_query_dim", None) is not None
        ):
            raise ValueError(
                "gt_pose_oracle requires Decoder's internal pose encoder; "
                "remove decoder_cfg.params.external_query_dim"
            )
        self.auxiliary_pose_detach_features = bool(
            auxiliary_pose_detach_features
        )
        if auxiliary_pose_head and rotation_query_source != "video_features":
            raise ValueError(
                "auxiliary_pose_head requires "
                "rotation_query_source='video_features'"
            )
        if auxiliary_pose_head:
            feature_dim = self.encoder.output_feature_dim
            self.auxiliary_pose_head = AuxiliaryPoseHead(feature_dim)
        else:
            self.auxiliary_pose_head = None
        if pose_fusion_cfg is not None:
            if rotation_query_source != "video_features":
                raise ValueError(
                    "pose_fusion_cfg requires "
                    "rotation_query_source='video_features'"
                )
            if self.auxiliary_pose_head is None:
                raise ValueError(
                    "pose_fusion_cfg requires auxiliary_pose_head=true"
                )
            self.pose_fusion = instantiate_from_config(pose_fusion_cfg)
            feature_dim = int(self.encoder.output_feature_dim)
            fusion_feature_dim = int(
                getattr(self.pose_fusion, "feature_dim", -1)
            )
            if fusion_feature_dim != feature_dim:
                raise ValueError(
                    "Pose-fusion/video feature widths differ: "
                    f"{fusion_feature_dim}/{feature_dim}"
                )
        else:
            self.pose_fusion = None

        self.bone_direction_head = None
        if bone_direction_head_cfg is not None:
            if rotation_query_source != "video_features":
                raise ValueError(
                    "bone_direction_head_cfg requires "
                    "rotation_query_source='video_features'"
                )
            bone_cfg = copy.deepcopy(bone_direction_head_cfg)
            bone_params = bone_cfg.setdefault("params", {})
            bone_params["edge_epsilon"] = self.edge_epsilon
            bone_params["zero_rest_edge_fallback_axis"] = (
                self.zero_rest_edge_fallback_axis
            )
            bone_params["include_zero_rest_edges"] = (
                self.include_zero_rest_edges
            )
            self.bone_direction_head = instantiate_from_config(bone_cfg)
            direction_feature_dim = int(
                getattr(self.bone_direction_head, "feature_dim", -1)
            )
            feature_dim = int(self.encoder.output_feature_dim)
            if direction_feature_dim != feature_dim:
                raise ValueError(
                    "Bone-direction/video feature widths differ: "
                    f"{direction_feature_dim}/{feature_dim}"
                )

        grace_source = str(grace_source).lower()
        if grace_source not in {"bone_direction", "pred_position"}:
            raise ValueError(
                "grace_source must be 'bone_direction' or "
                "'pred_position'"
            )
        self.grace_source = grace_source
        self.grace = None
        if grace_cfg is not None:
            configured_representation = str(
                decoder_cfg.get("params", {}).get(
                    "rotation_representation", "rot6d"
                )
            ).lower()
            if configured_representation != "rot6d":
                raise ValueError(
                    "observable swing correction currently requires "
                    "decoder rotation_representation='rot6d'"
                )
            if rotation_query_source != "video_features":
                raise ValueError(
                    "grace_cfg requires "
                    "rotation_query_source='video_features'"
                )
            if (
                grace_source == "bone_direction"
                and self.bone_direction_head is None
            ):
                raise ValueError(
                    "bone_direction correction requires bone_direction_head_cfg"
                )
            if (
                grace_source == "pred_position"
                and self.auxiliary_pose_head is None
            ):
                raise ValueError(
                    "pred_position correction requires auxiliary_pose_head=true"
                )
            grace_cfg = copy.deepcopy(grace_cfg)
            correction_params = grace_cfg.setdefault("params", {})
            correction_params["zero_rest_edge_fallback_axis"] = (
                self.zero_rest_edge_fallback_axis
            )
            self.grace = instantiate_from_config(
                grace_cfg
            )
            correction_feature_dim = int(
                getattr(self.grace, "feature_dim", -1)
            )
            feature_dim = int(self.encoder.output_feature_dim)
            if correction_feature_dim != feature_dim:
                raise ValueError(
                    "Rotation-correction/video feature widths differ: "
                    f"{correction_feature_dim}/{feature_dim}"
                )

    @with_attention_layouts
    def forward(
        self,
        batch,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        pose_source_mode: str = "pred",
        pose_mix_prob: float = 1.0,
        detach_pred_pose_for_rot: bool = False,
        return_reconstructed_positions: bool = True,
    ):
        if not isinstance(return_reconstructed_positions, bool):
            raise TypeError("return_reconstructed_positions must be a bool")

        model_batch = dict(batch)
        pred_position = None
        pose_fusion_output = None
        bone_direction_output = None
        grace_output = None

        if self.rotation_query_source == "gt_pose_oracle":
            if "oracle_pose_position" not in model_batch:
                raise KeyError(
                    "gt_pose_oracle requires "
                    "batch['oracle_pose_position']; the video2motion trainer only "
                    "injects this key for an explicit oracle config"
                )
            oracle_position = model_batch["oracle_pose_position"]
            if oracle_position.ndim != 4 or oracle_position.shape[-1] != 3:
                raise ValueError(
                    "oracle_pose_position must have shape [B,T,J,3], got "
                    f"{tuple(oracle_position.shape)}"
                )
            pose_for_rot = (
                oracle_position - oracle_position[:, :, 0:1]
            )
            decoder_out = self.decoder(
                batch=model_batch,
                pose_input=pose_for_rot,
            )
            used_pred_ratio = 0.0
            mode = "gt_pose_oracle"
        else:
            # Validation can reuse frozen DINO features across overlapping
            # centered windows. Temporal modules are still evaluated from
            # scratch for every window; only the immutable per-frame image
            # encoding is cached by the caller.
            if "image_embed" in model_batch:
                image_embed = model_batch["image_embed"]
            elif "image_rgb" in model_batch:
                image_embed = self.image_backbone(
                    model_batch["image_rgb"],
                    frame_valid_mask=model_batch.get(
                        "frame_observed_mask",
                        model_batch.get("frame_valid_mask"),
                    ),
                )
            else:
                raise KeyError(
                    "BaseModel requires batch['image_rgb'] or "
                    "precomputed batch['image_embed']"
                )
            model_batch["image_embed"] = image_embed
            if (
                attention_kwargs is None
                or not self.encoder_accepts_attention_kwargs
            ):
                video_output = self.encoder(model_batch)
            else:
                video_output = self.encoder(
                    model_batch,
                    attention_kwargs=attention_kwargs,
                )

            if self.rotation_query_source == "video_features":
                need_auxiliary_pose = (
                    self.auxiliary_pose_head is not None
                    and (
                        return_reconstructed_positions
                        or self.pose_fusion is not None
                        or self.grace_source == "pred_position"
                    )
                )
                if need_auxiliary_pose:
                    # This is an auxiliary branch only. Decoder continues to
                    # consume the full Encoder feature tensor below.
                    pose_feature = (
                        video_output.detach()
                        if self.auxiliary_pose_detach_features
                        else video_output
                    )
                    pred_position = self.auxiliary_pose_head(pose_feature)
                    pred_position = pred_position - pred_position[:, :, 0:1]
                    valid = model_batch["joint_mask"].bool().unsqueeze(1)
                    frame_valid = model_batch.get("frame_valid_mask")
                    if frame_valid is not None:
                        valid = valid & frame_valid.bool().unsqueeze(-1)
                    pred_position = (
                        pred_position
                        * valid.unsqueeze(-1).to(pred_position.dtype)
                    )
                rotation_query = video_output
                if self.pose_fusion is not None:
                    if pred_position is None:
                        raise RuntimeError(
                            "Pose fusion requires an auxiliary pose prediction"
                        )
                    pose_fusion_output = self.pose_fusion(
                        visual_features=video_output,
                        predicted_position=pred_position,
                        batch=model_batch,
                    )
                    rotation_query = pose_fusion_output["fused_query"]
                if self.bone_direction_head is not None:
                    bone_direction_output = self.bone_direction_head(
                        visual_features=video_output,
                        batch=model_batch,
                        return_reconstructed_positions=return_reconstructed_positions,
                    )
                decoder_out = self.decoder(
                    batch=model_batch,
                    query_input=rotation_query,
                )
                if self.grace is not None:
                    if self.grace_source == "bone_direction":
                        direction_source = bone_direction_output
                        if direction_source is None:
                            raise RuntimeError(
                                "bone-direction correction has no prediction"
                            )
                    else:
                        if pred_position is None:
                            raise RuntimeError(
                                "XYZ swing correction has no pose prediction"
                            )
                        direction_source = positions_to_bone_directions(
                            pred_position,
                            model_batch["parent_a"].long(),
                            model_batch["offset_a"].to(pred_position.dtype),
                            model_batch["joint_mask"].bool(),
                            model_batch.get("frame_valid_mask"),
                            edge_epsilon=self.edge_epsilon,
                        )
                        direction_source = {
                            "pred_bone_direction": direction_source["direction"],
                            "pred_bone_confidence": direction_source["confidence"],
                            "pred_bone_valid_mask": direction_source["valid_mask"],
                            "pred_bone_position": None,
                        }
                        if return_reconstructed_positions:
                            direction_source["pred_bone_position"] = (
                                reconstruct_positions_from_bone_directions(
                                    direction_source["pred_bone_direction"],
                                    model_batch["offset_a"].to(pred_position.dtype),
                                    model_batch["parent_a"].long(),
                                    model_batch["joint_mask"].bool(),
                                    model_batch.get("frame_valid_mask"),
                                    edge_epsilon=self.edge_epsilon,
                                )
                            )
                        bone_direction_output = direction_source
                    baseline_matrix = decoder_out.get(
                        "pred_rotation_matrix"
                    )
                    if baseline_matrix is None:
                        baseline_matrix = _rot6d_to_rotmat_columns(
                            decoder_out["pred_rot6d"].float()
                        )
                    grace_output = self.grace(
                        baseline_local_rotation=baseline_matrix,
                        target_bone_direction=direction_source[
                            "pred_bone_direction"
                        ],
                        bone_confidence=direction_source[
                            "pred_bone_confidence"
                        ],
                        batch=model_batch,
                        visual_features=(
                            video_output
                            if getattr(
                                self.grace, "gate_mode", "confidence"
                            )
                            == "visual"
                            else None
                        ),
                        return_diagnostics=return_reconstructed_positions,
                    )
                used_pred_ratio = 1.0
                mode = "video_features"
            else:
                pred_position = video_output
                pred_root_position = pred_position[:, :, 0]
                pred_pose_for_rot = (
                    pred_position - pred_root_position.unsqueeze(2)
                )

                mode = pose_source_mode.lower()
                if mode == "pred":
                    pose_for_rot = (
                        pred_pose_for_rot.detach()
                        if detach_pred_pose_for_rot
                        else pred_pose_for_rot
                    )
                    used_pred_ratio = 1.0
                else:
                    if "position" not in model_batch:
                        raise KeyError(
                            f"pose_source_mode={mode!r} requires "
                            "batch['position'] during training"
                        )
                    gt = model_batch["position"]
                    gt_pose_for_rot = gt - gt[:, :, 0:1]
                    pred_used = (
                        pred_pose_for_rot.detach()
                        if detach_pred_pose_for_rot
                        else pred_pose_for_rot
                    )
                    if mode == "gt":
                        pose_for_rot = gt_pose_for_rot
                        used_pred_ratio = 0.0
                    elif mode == "mix":
                        selector = (
                            torch.rand(gt.shape[0], device=gt.device)
                            < float(pose_mix_prob)
                        ).view(-1, 1, 1, 1)
                        pose_for_rot = torch.where(
                            selector, pred_used, gt_pose_for_rot
                        )
                        used_pred_ratio = float(
                            selector.float().mean().item()
                        )
                    else:
                        raise ValueError(
                            f"Unknown pose_source_mode: {pose_source_mode}"
                        )
                decoder_out = self.decoder(
                    batch=model_batch,
                    pose_input=pose_for_rot,
                )

        baseline_rot6d = decoder_out["pred_rot6d"]
        baseline_matrix = decoder_out.get("pred_rotation_matrix")
        if baseline_matrix is None:
            baseline_matrix = _rot6d_to_rotmat_columns(
                baseline_rot6d.float()
            )
        final_rotation_matrix = baseline_matrix

        # Integration order: VisualGate corrected rotation / existing base
        # rotation -> anchor direction branch -> analytic solver -> final
        # rotation.
        if grace_output is not None:
            final_rotation_matrix = grace_output[
                "corrected_local_rotation"
            ]
        final_rot6d = _rotmat_to_rot6d_columns(final_rotation_matrix)

        output = {
            # Joint zero is supervised camera/global root rotation; remaining
            # joints are supervised parent-local rotations.
            "pred_rot6d": final_rot6d,
            "rest_embed": decoder_out.get("rest_embed"),
            "q_feat": decoder_out.get("q_feat"),
            "mem_feat": decoder_out.get("mem_feat"),
            "pose_source_info": {
                "pred_prob": float(
                    pose_mix_prob if mode == "mix" else used_pred_ratio
                ),
                "used_pred_ratio": used_pred_ratio,
                "used_gt_ratio": 1.0 - used_pred_ratio,
            },
        }
        any_correction = (
            grace_output is not None
        )
        if not any_correction:
            output["pred_rotation_raw"] = decoder_out.get(
                "pred_rotation_raw"
            )
            if decoder_out.get("pred_quaternion") is not None:
                output["pred_quaternion"] = decoder_out["pred_quaternion"]
        else:
            output["pred_base_rotation_raw"] = decoder_out.get(
                "pred_rotation_raw"
            )
            if decoder_out.get("pred_quaternion") is not None:
                output["pred_base_quaternion"] = decoder_out[
                    "pred_quaternion"
                ]
            output["pred_base_rot6d"] = baseline_rot6d
            output["pred_base_rotation_matrix"] = baseline_matrix
            output["pred_corrected_rot6d"] = final_rot6d
            output["pred_corrected_rotation_matrix"] = final_rotation_matrix

        if final_rotation_matrix is not None:
            output["pred_rotation_matrix"] = final_rotation_matrix
        if grace_output is not None:
            output["observable_swing_applied"] = True
            for key, value in grace_output.items():
                if key not in {
                    "corrected_local_rotation",
                    "corrected_global_rotation",
                }:
                    output[key] = value
        if bone_direction_output is not None:
            output.update(bone_direction_output)
        if pred_position is not None:
            output["pred_position"] = pred_position
        if pose_fusion_output is not None:
            for key, value in pose_fusion_output.items():
                if key != "fused_query":
                    output[key] = value
        return output

    @torch.no_grad()
    def prepare_inference_optimizations(self):
        """Build the packed inference projections after the weights are loaded.

        Streaming inference uses a single packed projection per self-attention
        module, reducing launch and intermediate-tensor overhead without
        changing any weights or numerical operations.
        """
        prepared = 0
        for module in self.modules():
            prepare = getattr(module, "prepare_inference_fusion", None)
            if prepare is not None:
                prepare()
                prepared += 1
        return prepared

    @torch.no_grad()
    def prepare_streaming_static_state(self, batch, *, max_length=81):
        """Precompute every fixed-skeleton input before video frames arrive."""
        if self.rotation_query_source != "video_features":
            raise RuntimeError("streaming API currently targets rotation_query_source='video_features'")
        static = dict(batch)
        joint_mask = static["joint_mask"].bool()
        encoder_state = {
            "ref_query": None,
            "temporal": self.encoder.temporal_model.init_streaming_state(
                max_length=max_length,
                joint_mask=joint_mask,
                graph_hop=static.get("graph_hop"),
                graph_edge=static.get("graph_edge"),
            ),
        }
        decoder_state = self.decoder.init_streaming_state(static, max_length=max_length)
        state = {
            "encoder": encoder_state,
            "decoder": decoder_state,
            # Streaming calls do not use the full-forward decorator; keep a
            # persistent metadata cache for active rows/packed layouts.
            "attention_layout_cache": {},
        }
        # Eager prefill is allowed to populate metadata; benchmark/serving
        # code freezes this cache before compiling the steady-state step.
        set_attention_layout_cache(state["attention_layout_cache"], readonly=False)
        # Skeleton metadata is immutable for a realtime stream.  Keep the
        # already-cast CUDA tensors in the state so every frame does not
        # rebuild a batch dictionary or repeat bool/long conversions.
        static_batch = {}
        for key, value in static.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key in {"joint_mask", "ancestor_mask", "static_rot_joint_mask"}:
                value = value.bool()
            elif key in {"parent_a"}:
                value = value.long()
            static_batch[key] = value
        static_batch.pop("image_embed", None)
        static_batch.pop("image_rgb", None)
        static_batch.pop("frame_valid_mask", None)
        state["static_batch"] = static_batch
        if self.grace is not None:
            state["grace_topology"] = (
                self.grace.build_streaming_topology_cache(static)
            )
        return state

    @torch.no_grad()
    def bind_streaming_reference(self, state):
        """Compute the static reference query from the fixed skeleton."""
        static = state["static_batch"]
        joint_mask = static["joint_mask"].bool()
        state["encoder"]["ref_query"] = self.encoder.ref_encoder(
            ref_position=static["ref_position"],
            joint_mask=joint_mask,
            graph_hop=static["graph_hop"],
            graph_edge=static["graph_edge"],
            tree_mask=static["ancestor_mask"].bool(),
        )
        return state

    @torch.no_grad()
    def init_streaming_state(self, batch, *, max_length=81):
        """Initialize a causal stream."""
        state = self.prepare_streaming_static_state(batch, max_length=max_length)
        return self.bind_streaming_reference(state)

    @torch.no_grad()
    def forward_streaming(
        self,
        batch,
        state,
    ):
        """Consume one frame and update caches.

        This is the latency-critical inference path.  It intentionally omits
        FK reconstruction and diagnostic buffers; those remain in ``forward``
        and offline validation paths.
        """
        static_batch = state.get("static_batch")
        if static_batch is None:
            model_batch = dict(batch)
        else:
            model_batch = dict(static_batch)
            for key in ("image_embed", "image_rgb", "frame_valid_mask"):
                if key in batch:
                    model_batch[key] = batch[key]
        if "image_embed" not in model_batch:
            if "image_rgb" not in model_batch:
                raise KeyError("forward_streaming requires image_embed or image_rgb")
            model_batch["image_embed"] = self.image_backbone(model_batch["image_rgb"])
        image_embed = model_batch["image_embed"]
        if image_embed.ndim != 4 or image_embed.shape[1] != 1:
            raise ValueError("streaming image_embed must be [B,1,P,D]")
        video_output, _ = self.encoder.forward_streaming(
            model_batch, state["encoder"]
        )
        rotation_query = video_output
        pred_position = None
        bone_direction_output = None
        need_auxiliary_pose = (
            self.auxiliary_pose_head is not None
            and (
                self.pose_fusion is not None
                or self.grace_source == "pred_position"
            )
        )
        if need_auxiliary_pose:
            pred_position = self.auxiliary_pose_head(video_output)
            pred_position = pred_position - pred_position[:, :, 0:1]
        if self.pose_fusion is not None:
            pose_fusion_output = self.pose_fusion(
                visual_features=video_output, predicted_position=pred_position, batch=model_batch
            )
            rotation_query = pose_fusion_output["fused_query"]
        if self.bone_direction_head is not None:
            bone_direction_output = self.bone_direction_head(
                visual_features=video_output,
                batch=model_batch,
                return_reconstructed_positions=False,
            )
        decoder_out, _ = self.decoder.forward_streaming(
            model_batch, state["decoder"], query_input=rotation_query
        )
        baseline_rot6d = decoder_out["pred_rot6d"]
        baseline_matrix = _rot6d_to_rotmat_columns(baseline_rot6d.float())
        final_matrix = baseline_matrix
        grace_output = None
        if self.grace is not None:
            if self.grace_source == "bone_direction":
                if bone_direction_output is None:
                    raise RuntimeError("bone-direction correction has no prediction")
                direction_source = bone_direction_output
            else:
                if pred_position is None:
                    raise RuntimeError("XYZ swing correction has no pose prediction")
                direction_source = positions_to_bone_directions(
                    pred_position, model_batch["parent_a"].long(), model_batch["offset_a"].to(pred_position.dtype),
                    model_batch["joint_mask"].bool(), model_batch.get("frame_valid_mask"), edge_epsilon=self.edge_epsilon,
                )
                direction_source = {
                    "pred_bone_direction": direction_source["direction"],
                    "pred_bone_confidence": direction_source["confidence"],
                    "pred_bone_valid_mask": direction_source["valid_mask"],
                }
            grace_output = _streaming_grace_correction(
                self.grace,
                baseline_local_rotation=baseline_matrix,
                target_bone_direction=direction_source["pred_bone_direction"],
                bone_confidence=direction_source["pred_bone_confidence"],
                batch=model_batch,
                visual_features=video_output if getattr(self.grace, "gate_mode", "confidence") == "visual" else None,
                return_diagnostics=False,
                topology_cache=state.get("grace_topology"),
            )
            final_matrix = grace_output["corrected_local_rotation"]
        output = {
            "pred_rot6d": _rotmat_to_rot6d_columns(final_matrix),
            "pred_rotation_raw": decoder_out.get("pred_rotation_raw"),
            "rest_embed": decoder_out.get("rest_embed"),
            "q_feat": decoder_out.get("q_feat"),
            "mem_feat": None,
            "pred_position": pred_position,
            "_stream_visual_features": video_output,
        }
        if grace_output is not None:
            output.update({k: v for k, v in grace_output.items() if k not in {"corrected_local_rotation", "corrected_global_rotation"}})
        if bone_direction_output is not None:
            output.update(bone_direction_output)
        return output, state

    @staticmethod
    def _is_external_frozen_key(key: str) -> bool:
        return key.startswith((
            "image_backbone.backbone.",
        ))

    def trainable_state_dict(self):
        """Exclude externally loaded frozen DINO weights from checkpoints."""
        return {
            key: value
            for key, value in self.state_dict().items()
            if not self._is_external_frozen_key(key)
        }

    def load_trainable_state_dict(
        self,
        state_dict,
        allowed_missing_prefixes=(),
    ):
        allowed_missing_prefixes = tuple(
            str(prefix) for prefix in allowed_missing_prefixes
        )
        result = self.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key
            for key in result.missing_keys
            if not self._is_external_frozen_key(key)
            and not any(
                key.startswith(prefix) for prefix in allowed_missing_prefixes
            )
        ]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(
                "Video2Motion checkpoint mismatch: "
                f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
            )
