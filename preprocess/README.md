# Preprocessed dataset cache

The caches use a right-handed OpenCV camera frame: x points right, y points
down, and z points forward. **All model-facing positions, root orientations,
and rest-skeleton coordinates use this camera-frame convention.** Non-root
rotations and rest-skeleton transforms are parent-local. Positions and
translations are in meters. Named camera transforms map between this frame and
the source scene; Mixamo also preserves source-world rest transforms in
`bind_transforms.npz`.

The [Zoo](zoo/README.md) and [Mobjaverse](mobjaverse/README.md) pipelines turn
animated source assets into the RGB, mask, skeleton, and motion cache consumed
by CAST. [Mixamo](mixamo/README.md) builds a separate paired cross-skeleton
evaluation cache. Each guide explains how to build its cache. The source
datasets and generated caches are distributed separately.

## Zoo and Mobjaverse cache layout

```text
cache/
  <skeleton-id>/
    meta.json
    static.npy
    rot6d.npy
    world_pos.npy
    canonical_root_pos.npy
    images/shard-*.tar
```

One `<skeleton-id>` directory stores one shared rest skeleton and one or more
animations or camera views. The `.npy` motion arrays concatenate all entries
in `meta.json["motions"]` along time. Let `T` be
`meta.json["total_frames"]` and `J` be the number of joints.

| File | Contents |
| --- | --- |
| `static.npy` | A Python dictionary of rest-skeleton data; load with `np.load(path, allow_pickle=True).item()`. |
| `rot6d.npy` | `float32 [T, J, 6]` local joint rotations. Each 6D value concatenates the first two **columns** of a rotation matrix. Joint 0 is oriented in the camera frame; other joints are relative to their parents. |
| `world_pos.npy` | `float32 [T, J, 3]` absolute joint positions in the OpenCV camera frame, in meters. |
| `canonical_root_pos.npy` | `float32 [T, 3]` camera-frame root positions in meters, equal to `world_pos[:, 0]`. |
| `images/shard-*.tar` | JPEG RGB frames paired with PNG foreground masks. |

Position arrays and rest offsets are stored in meters. The `metric_scale` field
below is an additional scale for a 2 m rest-skeleton convention; it is not
already applied to the saved motion arrays or images.

## Shared rest-skeleton keys in `static.npy`

| Key | Meaning |
| --- | --- |
| `joint_names` | Joint names in the order used by every motion array. |
| `parents` | Parent index for each joint; root joint 0 has parent `-1`. Parents precede their children. |
| `rest_translations` | `[J, 3]` local rest-pose offsets in meters. |
| `rest_rotations_quat` | `[J, 4]` local rest-pose rotations as `xyzw` quaternions. |
| `valid_joint_mask` | `[J]` mask selecting joints used for the rest-skeleton extent. |
| `metric_scale` | `2 / d`, where `d` is the axis-aligned bounding-box diagonal of valid rest joints after rest-pose forward kinematics, in meters. Multiply meter-valued skeleton coordinates by this factor for the 2 m convention. |

## Zoo and Mobjaverse keys in `meta.json`

| Key | Meaning |
| --- | --- |
| `coordinate_system`, `units` | Camera-frame axis convention and meter units. |
| `skeleton_id`, `joints`, `joint_names` | Identity, joint count, and joint order for this directory. |
| `total_frames` | Total number of frames in the concatenated arrays. |
| `array_files`, `array_shapes` | Filenames and expected shapes of the three motion arrays. |
| `motions` | Ordered list of animations or camera views stored in this directory. |

Each entry in `motions` describes one animation and camera view:

| Key | Meaning |
| --- | --- |
| `offset`, `frames` | Slice `[offset : offset + frames]` in each concatenated motion array. |
| `source_motion`, `clip_name` | Source animation or asset identifier, and the name used for its tar members. |
| `image.tar` | Relative path to the tar archive containing this entry's RGB and mask frames. |
| `images` | Image and mask counts and resolution. |
| `fps` | Playback rate in frames per second. |
| `camera` | Camera intrinsics and pose for this entry. |

Paired archive members are named `<clip_name>.<frame:06d>.jpg` and
`<clip_name>.<frame:06d>.png`, with frame numbers starting at zero for each
entry.

Each motion also has a `camera` dictionary. `fx`, `fy`, `cx`, `cy`, `width`, and
`height` describe the rendered camera intrinsics. `location`, `right`, `up`,
and `forward` specify the camera pose in the normalized source scene;
`camera_to_world` and `world_to_camera` convert between that scene and camera
coordinates. Zoo stores multiple views of a source animation. Mobjaverse stores
a fixed camera per asset and records its source split and category in each
motion entry.

## Mixamo evaluation cache

Mixamo uses the same `static.npy`, `rot6d.npy`, `world_pos.npy`, and
`canonical_root_pos.npy` definitions above. Its directory key is a character
`skeleton_id`; the cache also stores each character's original rest mesh and
reference motion for cross-skeleton evaluation:

```text
cache/
  <skeleton-id>/
    meta.json
    static.npy
    rot6d.npy
    world_pos.npy
    canonical_root_pos.npy
    bind_transforms.npz
    target_rest.blend
    reference/<clip>.npz
    rgba/<clip>/
    rgb/<clip>/
    mask/<clip>/
    images/oblique15_el18.tar
  pairs.jsonl
  validation.json
```

| File or key | Meaning |
| --- | --- |
| `static.npy` → `skeleton_id` | Character identity stored with the shared rest skeleton. |
| `meta.json` → `skeleton_id`, `motions` | Character identity and ordered clips. Each motion's `offset` and `frames` select a slice of the concatenated arrays; `fps`, `clip_name`, `camera`, and `image.tar` describe its rendered frames. |
| `bind_transforms.npz` | `world_rest` is the FBX source-world rest pose, `camera_from_world` maps it to the camera frame, and `rest_local` is parent-local. |
| `target_rest.blend` | Original skinned character mesh in its rest pose, with packed textures. |
| `reference/<clip>.npz` | Imported FBX pose: camera-frame `global_transforms`, parent-local `local_transforms`, and a per-frame `camera_from_world` transform. |
| `images/oblique15_el18.tar` | RGB JPEG and mask PNG members named `<clip>.<frame:06d>.jpg/.png`, numbered from zero for each clip. |
| `pairs.jsonl` | One directed source-to-target record per character pair and clip. `source_rgb` and `source_mask` name the input frames; `target_static` and `target_reference` name the target rig and reference pose. |
| `validation.json` | Completed character, clip, frame, and directed-pair counts. |

See the [Mixamo guide](mixamo/README.md) for the FBX input layout and the
one-command build procedure.
