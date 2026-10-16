# Models and Configurations

## Model Families

Remote inference supports CAST-B and CAST-L with DINOv3 and frame-wise causal
attention. Select the family that matches your causal checkpoint.

| Model | Configuration | Preparation argument |
| --- | --- | --- |
| CAST-B | [configs/CAST_B.yaml](../configs/CAST_B.yaml) | `--model CAST-B` |
| CAST-L | [configs/CAST_L.yaml](../configs/CAST_L.yaml) | `--model CAST-L` |

These configurations use the repository's encoder, decoder, base model, and
GRACE modules. The model architecture follows the corresponding CAST-B or
CAST-L configuration; the remote settings enable single-frame causal streaming
and resolve backbone paths from environment variables.

Use a checkpoint trained for causal inference. The non-causal checkpoints used
for offline evaluation require causal fine-tuning before deployment. Model
preparation checks all trained tensor keys, shapes, and values; it preserves the
checkpoint's encoder, decoder, and GRACE parameter names.

```bash
python tools/prepare_checkpoint.py \
  --model CAST-L --checkpoint /path/to/CAST_L_dinov3_causal.pt \
  --model-id cast-l --activate
```

For CAST-B, use `--model CAST-B` with its matching causal checkpoint and a
distinct model ID. Preparation records the family, source hash, and training
step in the manifest. Activation takes effect at the next API start.

## Required Assets

| Asset | Default local path | Setup |
| --- | --- | --- |
| DINOv3 ViT-L/16 pretrained weights | `assets/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | Obtain under the DINOv3 terms |
| SAM2.1 Hiera Tiny | `assets/weights/sam2/sam2.1_hiera_tiny.pt` | `python tools/fetch_public_weights.py --confirm-model-terms` |
| TorchVision SSDLite320 MobileNetV3 COCO | `assets/weights/person/ssdlite320_mobilenet_v3_large_coco-a79551df.pth` | Same download command |
| CAST-B or CAST-L causal checkpoint | `assets/models/<id>/runtime.pt` after preparation | `python tools/prepare_checkpoint.py` |
| Character rig and camera | Keep character files and bind transforms on the client | [Blender exporter](BLENDER.md#1-export-the-rig) |

The service loads all three pretrained components and the motion model at
startup, including the person detector when using generic tracking mode.
Weights are supplied separately from the source release.

## External Source Revisions

`third_party/dependencies.json` records repository URLs, commits, and license
information. Fetch sources with:

```bash
python tools/fetch_dependencies.py --accept-dinov3-license
```

- DINOv3: `346f38fee679c56a6888f91c51670fae61d364e0`.
- SAM2: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

The script fetches source code only and refuses to overwrite modified or
incompatible checkouts. Repeat numerical, CUDA, and protocol checks when
updating revisions.

## Weight Integrity

Public download URLs and SHA-256 values are recorded in the download tool.
An existing file with an incorrect hash causes an error.

```text
SAM2.1 Tiny:
7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
SSDLite320 MobileNetV3:
a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2
```

For DINOv3 access, follow the instructions in its upstream repository. CAST
preparation records the actual source checkpoint hash and training step.

## Licenses

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the license texts in
`licenses/`. Model weights, datasets, characters, and images retain their own
terms. DINOv3 is governed by its separate license.
