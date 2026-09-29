# Mixamo preprocessing

The motion arrays and rest skeleton use the OpenCV camera coordinate system:
right is +X, down is +Y, and forward is +Z. Positions and translations are in
meters; non-root joint rotations and rest offsets are parent-local. Camera
matrices explicitly convert between this frame and the FBX scene.

This pipeline builds the six-character Mixamo cross-skeleton **evaluation**
cache. It contains rendered RGB/mask frames, source animation, target rest rigs,
and directed source-to-target pairs. It is separate from the Zoo and Mobjaverse
training caches described in [the preprocessing overview](../README.md).

## Requirements

- Blender 3.6.3 with its bundled Python/NumPy. The renderer uses Blender 3.6's
  EEVEE API.
- Python 3 with NumPy and Pillow for the cache packaging step.
- `xvfb-run` for headless Linux rendering.

## Source FBX files

Download animated, **skinned** FBX files from
[Mixamo](https://www.mixamo.com/) and arrange them as six character
directories. Each character needs the same ten animation names:
`Dancing`, `Drop Kick`, `Fall Flat`, `Flying Kick`, `Freehang Climb`, `Jogging`,
`Praying`, `Situps`, `Skinning Test`, and `Zombie Stand Up`.

```text
raw_fbx/
  Big_Vegas/
    Dancing.fbx
    Drop Kick.fbx
    ...
  Brian/
  Mousey/
  Mutant/
  The_Boss/
  Ty/
```

Each character's FBX clips must contain the same rest rig and a valid animated
frame range. The reference exports are 30 FPS. The files are named by their
animation; spaces become underscores in cache clip names, for example
`Drop Kick.fbx` becomes `Drop_Kick`.

## Build the cache

From the repository root, run:

```bash
bash preprocess/mixamo/build_cache.sh \
  /path/to/raw_fbx \
  /path/to/blender-3.6.3-linux-x64/blender \
  /path/to/mixamo-work
```

Use `MIXAMO_RENDER_WORKERS` and `MIXAMO_BLENDER_THREADS` to change the default
three simultaneous Blender jobs and four threads per job. The command exports
all six skeletons and their animations, renders every frame, then packages the
images and constructs directed evaluation pairs. Completed clips are skipped
when the command is resumed. Output is under `mixamo-work/cache/`; per-job logs
are under `mixamo-work/logs/`.

## Cache layout

```text
cache/
  Big_Vegas/                 # likewise Brian, Mousey, Mutant, The_Boss, Ty
    static.npy
    meta.json
    rot6d.npy
    world_pos.npy
    canonical_root_pos.npy
    bind_transforms.npz
    target_rest.blend
    reference/<clip>.npz
    images/oblique15_el18.tar
    rgba/<clip>/
    rgb/<clip>/
    mask/<clip>/
  pairs.jsonl
  validation.json
```

`rot6d.npy` is `[T, J, 6]` local joint rotation, `world_pos.npy` is
`[T, J, 3]` camera-frame joint position, and `canonical_root_pos.npy` is
`[T, 3]` camera-frame root position. `meta.json` gives each clip's `offset` and
`frames` slice within those concatenated arrays. `static.npy` identifies the
character by `skeleton_id` and contains the shared rest rig. Its `metric_scale` is
`2 /` the rest-joint AABB diagonal in meters, and is not applied to the stored
positions.

`reference/<clip>.npz` stores the imported FBX joint transforms and per-frame
camera transforms for the target character. It is the Mixamo retargeted
reference, not a captured motion ground truth. `bind_transforms.npz` and
`target_rest.blend` preserve the original rest rig and skinned mesh. Each tar
contains paired `<clip>.<frame:06d>.jpg` RGB and
`<clip>.<frame:06d>.png` masks, with frame numbers starting at zero.

Each `pairs.jsonl` record names a source character, a different target
character, a clip, and paths for the source RGB/mask and target rest/reference.
The reference six-character export has 60 animations, 6,528 frames, and 300
directed pairs. A different FBX frame range changes the frame total.

## Camera and lighting

The renderer uses 512×512 images, a 50 mm lens on a 36 mm sensor, and a camera
at azimuth 15° and elevation 18°. It follows the mesh center while keeping a
clip-wide framing distance. Three disk area lights have base energies
`(44, 28.6, 44) ×` the character-wide mesh maximum edge length squared. The
character-wide extent includes every clip and the rest pose; light positions
and sizes use that same scale. The world light uses RGB `(0.17, 0.17, 0.17)`
and strength `0.385`. The inference `mixamo` preset uses the same three-light
arrangement and world settings.

## License

The preprocessing code follows the repository [MIT license](../../LICENSE).
Mixamo FBX assets are not included. Obtain and use them under
[Adobe's Mixamo terms](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html).

## Acknowledgments

We thank Adobe for providing the Mixamo characters and animation library.
