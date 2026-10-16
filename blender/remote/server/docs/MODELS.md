# Models and Configurations

## Model Families

Remote inference supports CAST-B and CAST-L with DINOv3 and frame-wise causal
attention. Select the family that matches your causal checkpoint.

| Model | Configuration | Preparation argument |
| --- | --- | --- |
| CAST-B | [CAST_B_dinov3_causal.yaml](../../../../configs/experiment/CAST_B_dinov3_causal.yaml) | `--model CAST-B` |
| CAST-L | [CAST_L_dinov3_causal.yaml](../../../../configs/experiment/CAST_L_dinov3_causal.yaml) | `--model CAST-L` |

Re-run preparation with a new model ID after changing model settings.

Download [CAST_B_dinov3_causal.pt](https://huggingface.co/CAST-mocap/CAST/resolve/main/CAST_B_dinov3_causal.pt)
for CAST-B. For CAST-L, provide your own matching causal checkpoint.

```bash
python tools/prepare_checkpoint.py \
  --model CAST-B --checkpoint /path/to/CAST_B_dinov3_causal.pt \
  --model-id cast-b --activate
```

For CAST-L, use `--model CAST-L` with its matching causal checkpoint and a
distinct model ID. Restart the API after activation to load the selected model.

## Required Assets

| Asset | Default local path | Setup |
| --- | --- | --- |
| DINOv3 ViT-L/16 pretrained weights | `<repository>/third_party/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | Obtain under the DINOv3 terms |
| SAM2.1 Hiera Tiny | `assets/weights/sam2/sam2.1_hiera_tiny.pt` | `python tools/fetch_public_weights.py --confirm-model-terms` |
| TorchVision SSDLite320 MobileNetV3 COCO | `assets/weights/person/ssdlite320_mobilenet_v3_large_coco-a79551df.pth` | Same download command |
| CAST-B or CAST-L causal checkpoint | `assets/models/<id>/runtime.pt` after preparation | `python tools/prepare_checkpoint.py` |
| Character rig and camera | Keep character files and bind transforms on the client | [Blender exporter](../../client/README.md) |

Prepare every weight listed above, including the person detector when using
generic tracking mode.

## Dependency Paths

Use the DINOv3 and SAM2 installations described in the [main repository installation guide](../../../../docs/installation.md).
Use `<repository>/third_party/dinov3` and `<repository>/sam2` by default. Set
`CAST_DINOV3_REPO` or `CAST_SAM2_REPO` only when your shared installation uses
another location.

## Weight Integrity

Public download URLs and SHA-256 values are recorded in
[`fetch_public_weights.py`](../tools/fetch_public_weights.py).
If the downloader reports a checksum mismatch, verify the configured weight
path and move the incorrect file aside before downloading again.

## Licenses

See the repository's [LICENSE](../../../../LICENSE) and
[LICENSE-3RD-PARTY](../../../../LICENSE-3RD-PARTY) for license notices.
