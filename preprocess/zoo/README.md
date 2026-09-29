# Zoo preprocessing

Download the FBX files from the
[Truebones FBX/.BVH Zoo](https://truebones.gumroad.com/l/skZMC).

This pipeline builds the [CAST cache](../README.md) from those FBX files. The
input root contains one directory per character, with a character
FBX and its animation FBXs in that directory. Pass the FBX download root to the
script; raw BVH files are not accepted as input. The pipeline generates the
BVHs it needs from the FBXs and processes all animations in one run.

## Requirements

Use a Bash shell with Blender and FFmpeg available. Install the Python packages
from this directory:

```bash
python3 -m pip install -r requirements.txt
```

The required MoCapAnything components are included under `third_party/`.

## Build

Run from `preprocess/zoo`, passing the Zoo root, the Blender executable, and a
new working directory:

```bash
bash build_cache.sh \
  /path/to/Truebone_Z-OO \
  /path/to/blender \
  /path/to/new-work-directory
```

The script prepares the FBXs, renders RGB views and masks, builds view caches,
then merges and groups them by skeleton. The finished cache is in
`<work-directory>/cache/`. Set `DISPLAY` if Blender requires an X display.
`ZOO_RGB_WORKERS`, `ZOO_VERTEX_WORKERS`, `ZOO_MASK_WORKERS`, and
`ZOO_CACHE_WORKERS` control the corresponding worker counts.

## License

The bundled MoCapAnything V2 components are licensed under MIT; see
[`third_party/LICENSE`](third_party/LICENSE). The bundled PyTorch3D
`transforms3d.py` retains its BSD license; see the
[LICENSE-3RD-PARTY](../../LICENSE-3RD-PARTY). The Truebone Zoo FBX files
are obtained separately and are not included in this repository.

## Acknowledgments

We thank the creators of Truebone Zoo for the FBX animations and the authors
of [MoCapAnything V2](https://github.com/phongdaot/MocapAnything) for sharing
the FBX/BVH and rendering code adapted by this pipeline from commit
`c5ec0711400929416d9fddc05545f2c622cded64`.
