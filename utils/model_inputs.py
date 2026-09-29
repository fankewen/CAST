"""Assembling the tensors a forward pass consumes."""





import torch


TARGET_KEYS = {
    "_fk_layout",
    "position",
    "fk_position_target",
    "rot6d_a",
    "root_crop_camera_xyz",
    "camera_root_position",
    "camera_extrinsics",
    "loss_frame_mask",
    # Never accept this privileged input from an ordinary collated batch.
    # model_forward_inputs injects it only for the explicit oracle config.
    "oracle_pose_position",
    # Static supervision metadata must never enter the model.
}


ORACLE_UNUSED_VISUAL_KEYS = {
    "image_rgb", "full_image_rgb", "image_embed"
}


def model_inputs_only(batch):
    """Remove animation GT before the batch reaches model.forward."""
    return {key: value for key, value in batch.items() if key not in TARGET_KEYS}


def normalize_model_skeleton_inputs(batch):
    """Scale only static skeleton geometry by the dataset metric scale."""
    metric_scale = batch.get("metric_scale")
    if not isinstance(metric_scale, torch.Tensor):
        raise KeyError(
            "Model skeleton normalization requires batch['metric_scale']"
        )
    metric_scale = metric_scale.float().reshape(-1)
    if not torch.isfinite(metric_scale).all() or not bool((metric_scale > 0).all()):
        raise ValueError("batch['metric_scale'] must be finite and positive")

    inputs = model_inputs_only(batch)
    inputs.pop("metric_scale", None)
    for key in ("ref_position", "offset_a"):
        value = inputs.get(key)
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                f"Model skeleton normalization requires tensor batch[{key!r}]"
            )
        if value.shape[0] != metric_scale.numel():
            raise ValueError(
                f"{key} batch dimension does not match metric_scale: "
                f"{tuple(value.shape)} vs {tuple(metric_scale.shape)}"
            )
        scale_shape = (metric_scale.numel(),) + (1,) * (value.ndim - 1)
        inputs[key] = value * metric_scale.to(
            device=value.device, dtype=value.dtype
        ).view(scale_shape)
    return inputs


def uses_gt_pose_oracle(cfg):
    return (
        str(
            cfg.get("model", {})
            .get("params", {})
            .get("rotation_query_source", "pose")
        ).lower()
        == "gt_pose_oracle"
    )


def model_forward_inputs(batch, cfg):
    """Build leakage-safe model inputs, with one explicit oracle exception."""
    inputs = normalize_model_skeleton_inputs(batch)
    if uses_gt_pose_oracle(cfg):
        if "position" not in batch:
            raise KeyError(
                "GT-pose oracle config requires batch['position'] target"
            )
        # RGB is scientifically irrelevant to this oracle. Dropping it here
        # also guarantees that an accidental visual dependency fails loudly.
        for key in ORACLE_UNUSED_VISUAL_KEYS:
            inputs.pop(key, None)
        # Rename the privileged target so the model cannot silently consume
        # ordinary GT keys and normal configs remain fully leakage-isolated.
        inputs["oracle_pose_position"] = batch["position"]
    return inputs
