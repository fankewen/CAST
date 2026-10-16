# Third-party notices and redistribution boundaries

This archive contains deployable source code, not model weights or customer assets.
No upstream endorsement is implied. The publisher must verify ownership and authority
to publicly release its own modifications and any separately supplied checkpoints.

## Bundled original CAST code

The supplied CAST source carried the MIT license, copyright (c) 2026 Dao Thien Phong.
The original notice is preserved verbatim in `LICENSE` and `licenses/CAST-MIT.txt`.
That notice is not a license grant for unrelated model weights, external code or user assets.
Deployment files are adapted for relative configuration, external assets, safer defaults
and independent operation; bundled original attribution is retained.

## Bundled geometry utility

`utils/transforms3d.py` at the repository/archive root contains PyTorch3D-derived geometry utilities with
the original Meta copyright header. Its BSD license is retained in
`licenses/PyTorch3D-BSD.txt`. This file is an exception to the root MIT label.

## External DINOv3 — not included

DINOv3 source and pretrained weights are not bundled. The DINOv3 License is copied
to `licenses/DINOv3-external-LICENSE.md` for review, not relicensed as MIT.
The fixed source revision is listed in `third_party/dependencies.json`.
Use and redistribution of DINO Materials and their derivatives must comply with
the applicable DINOv3 agreement. Do not describe the full model stack as uniformly MIT.
Obtaining external code is an explicit local action after review of these terms.

## External SAM2 — not included

SAM2 source and weights are external resources. An Apache-2.0 license copy is retained
in `licenses/SAM2-external-Apache-2.0.txt` for reference; see the selected upstream
revision for its complete notices. Public download availability alone does not grant
rights over arbitrary input images or datasets.

## Installed dependencies

PyTorch, torchvision, FlashAttention, xFormers, FastAPI, Uvicorn, NumPy, Pillow and
other requirements are installed separately with their respective notices. This ZIP
does not redistribute their packages or CUDA binaries. If publishing a built container,
audit all installed components and preserve their licenses in that artifact as well.

## Models, rigs, data, and service histories

No CAST/DINO/SAM/detector weights, training data, FBX/blend files, rig caches,
camera frames, service histories, credentials, compiled caches, or Git history are included.
`docs/MODELS.md` and the download/prepare scripts are instructions, not a transfer of
rights to those resources. The motion checkpoint owner must provide its own licensing
terms before any public weight release. The deployment publisher should review
organizational release authorization before making this archive public.
