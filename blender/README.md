# CAST Live for Blender

## Installation

Use Blender 4.2 or later. Prepare an external Python environment using the
[main installation guide](../docs/installation.md), including the backbone,
SAM2, and the custom CUDA operator. Keep the full CAST checkout on the Blender
computer: the addon starts its inference engine from this checkout. Install
dependencies into that external environment, then select the same interpreter
in the panel's **Python** field.

Download [CAST_B_dinov3_causal.pt](https://huggingface.co/CAST-mocap/CAST/resolve/main/CAST_B_dinov3_causal.pt)
and select [CAST_B_dinov3_causal.yaml](../configs/experiment/CAST_B_dinov3_causal.yaml).
For CAST-L, use `configs/experiment/CAST_L_dinov3_causal.yaml` with your own matching
causal checkpoint. Supply the DINO backbone and SAM2 weights separately.

From the repository root, with that environment activated, build the ZIP:

```text
python blender/build_addon.py --output /path/to/cast_live.zip
```

Use a new output path if the ZIP already exists.

1. In Blender, open **Edit > Preferences > Add-ons > Install from Disk** and
   select the ZIP. Replace any previously installed single-file version of
   **CAST Live** first.
2. Enable **CAST Live**.
3. Open the 3D Viewport sidebar: press **N**, then select **CAST Live**.
4. Configure the panel and follow [Typical run](#typical-run).

The imported file must contain exactly one armature. For FBX and GLB, the addon
selects the newly imported rig even when other armatures are already present.
For `.blend`, it uses the armature in the opened scene. Set an active scene
camera before adapting the rig. Playback stays bound to this prepared armature;
if it is removed, import the character again before running a video.

## The panel

**Engine** - where the engine runs and how it is reached:

| field | meaning |
|---|---|
| `Repo` | repository root the engine runs from |
| `Python` | interpreter that has torch (the engine is started with it) |
| `GPU` | GPU index for inference, set through `CUDA_VISIBLE_DEVICES` |
| `Ctrl`, `Pose` | control port (5598) and pose port (5599) |

**Weights** - what `1. Load Weights` loads, all in one step:

| field | meaning |
|---|---|
| `CAST cfg` | the CAST model config the checkpoint was trained with (`.yaml`) |
| `CAST ckpt` | the CAST weights (`.pt`); the RGB backbone is not inside it |
| `SAM cfg` | SAM 2.1 config *name*, resolved inside the SAM2 install |
| `SAM ckpt` | SAM 2.1 weights |

Set the RGB backbone source and weights paths in the CAST configuration.

**Steps** - the buttons, in order:

| button | what it does |
|---|---|
| `1. Load Weights` | loads the CAST and SAM weights; wait for `models ready` before importing a character |
| `2. Import Character` | imports an FBX / GLB / `.blend` and prepares the rig; wait for `model ready` before running a video |
| `3. Run Video` | pick a clip (`mp4`/`mov`/`mkv`/`avi`) and feed it at `FPS`; while the feed is paused this loads another clip and waits for `Continue` |
| `Pause`, `Continue` | hold the feed where it is / carry on from that frame |
| `Stop` | stops the feed, keeps the model loaded |
| `Shutdown Engine` | stops the feed and terminates the engine process |

**Feed** - how the clip is fed:

| field | meaning |
|---|---|
| `FPS` | feed rate, default 30 |
| `After end` | `Repeat last frame` (default) or `Loop video` |
| `SAM res` | SAM encoder resolution, 512 default, 1024 official |
| `AMP` | `bf16` (default) or `fp16` |

**Status** - current step message, joint count, loaded SAM model and resolution,
poses received vs applied, and the last engine log lines.

## Typical run

1. Set `Repo`, `Python`, `GPU` once; they are stored in the scene.
2. Fill the Weights box (`CAST cfg`, `CAST ckpt`, `SAM cfg`, `SAM ckpt`).
3. Click `1. Load Weights` and wait for `models ready`.
4. `2. Import Character` and pick the character (`.blend` also works: the scene is
   opened from it). Wait for `model ready` and check the reported joint count.
5. `3. Run Video` and pick the clip: the armature starts moving.
6. `Pause` / `Continue` to look at a frame, `3. Run Video` to switch clip,
   `Stop` to end the feed, `Shutdown Engine` to free the card.

## Additional Settings

Set `Cache length` (default 81) and the engine-script override in
**Scene > Custom Properties** if needed.

Stop other pose receivers using port 5599 before starting the addon, or select
an unused **Pose** port.
