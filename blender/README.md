# CAST Live for Blender

The CAST Live addon runs video-driven motion capture in Blender and applies
the predicted motion to the imported character's armature. Inference runs in
a separate Python process with access to the GPU.

```text
video frame -> SAM 2.1 (subject box) -> crop -> causal mocap model -> poses -> armature
```

## Files

| Path | Purpose |
| --- | --- |
| [cast_live.py](cast_live.py) | Blender addon for local video capture and armature animation |
| [cast_engine_server.py](cast_engine_server.py) | GPU inference process started by the local addon |
| [setup_panel_scene.py](setup_panel_scene.py) | Enables the local addon in a Blender scene |

## Loading the addon

1. Copy `blender/cast_live.py` into Blender's `scripts/addons` directory, or use
   *Preferences > Add-ons > Install from file* and pick that file.
2. Enable **CAST Live** in *Preferences > Add-ons*.
3. Open the 3D viewport sidebar: press **N**, then pick the **CAST Live** tab.
4. `2. Import Character` brings your own character in (FBX / GLB / `.blend`).
   `blender/setup_panel_scene.py` only enables the addon and gives the timeline
   room to keep counting; it changes nothing else.

The imported file must contain exactly one armature. For FBX and GLB, the addon
selects the newly imported rig even when other armatures are already present.
For `.blend`, it uses the armature in the opened scene. Set an active scene
camera before adapting the rig. Playback stays bound to this prepared armature;
if it is removed, import the character again before running a video.

Torch never runs inside Blender: the panel starts
`blender/cast_engine_server.py` as a separate process and talks to it over a
local socket, so the UI never blocks inference and the other way round.

## The panel

**Engine** - where the engine runs and how it is reached:

| field | meaning |
|---|---|
| `Repo` | repository root the engine runs from |
| `Python` | interpreter that has torch (the engine is started with it) |
| `GPU` | card for the engine, as `CUDA_VISIBLE_DEVICES`; SAM shares it by design |
| `Ctrl`, `Pose` | control port (5598) and pose port (5599) |

**Weights** - what `1. Load Weights` loads, all in one step:

| field | meaning |
|---|---|
| `CAST cfg` | the CAST model config the checkpoint was trained with (`.yaml`) |
| `CAST ckpt` | the CAST weights (`.pt`); the RGB backbone is not inside it |
| `SAM cfg` | SAM 2.1 config *name*, resolved inside the SAM2 install |
| `SAM ckpt` | SAM 2.1 weights |

The two file paths have a browse button, and a path that does not exist is
reported before the engine is started. The RGB backbone needs no field of its
own: the CAST config names its target, its code directory and its weights.

**Steps** - the buttons, in order:

| button | what it does |
|---|---|
| `1. Load Weights` | starts the engine process and loads the `CAST ckpt` and `SAM ckpt` from the Weights box (SAM's CUDA graph takes ~30 s); no skeleton is involved yet |
| `2. Import Character` | imports an FBX / GLB / .blend with Blender's own importers and immediately adapts the armature + camera to CAST and compiles the streaming step (~1 min) |
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
3. `1. Load Weights`, wait for `models ready: CAST image_size=256, SAM sam2.1_hiera_tiny.pt@512`.
4. `2. Import Character` and pick the character (`.blend` also works: the scene is
   opened from it). One click imports it, adapts the rig and compiles the step;
   wait for `model ready (joints=57, cache=81)`.
5. `3. Run Video` and pick the clip: the armature starts moving.
6. `Pause` / `Continue` to look at a frame, `3. Run Video` to switch clip,
   `Stop` to end the feed, `Shutdown Engine` to free the card.

## Notes

* One GPU, strictly serial: SAM, then crop, then the model, one frame at a time.
  A frame costs the sum of its stages.
* `1. Load Weights` loads SAM too, so the first frame of the feed costs the same
  as the hundredth.
* The first frame of a clip becomes the streaming reference, so a new clip
  starts a new temporal window.
* `Cache length` (81) and the engine-script override are scene properties whose
  defaults live in `register()` of `blender/cast_live.py` and are not drawn in
  the panel; set those two in *Scene > Custom Properties*.
* `ops/` has to be built in place once:
  `python ops/setup.py build_ext --inplace`.
