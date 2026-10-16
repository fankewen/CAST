# CAST Remote Inference

## Requirements

Linux, Python 3.10, an NVIDIA GPU and driver compatible with CUDA 12.8,
CUDA Toolkit 12.8, Git, and a C/C++ compiler. Activate your Python environment
before installation.
See the [deployment guide](docs/DEPLOYMENT.md) for details.

## Installation

Clone the complete repository and follow the [main installation guide](../../../docs/installation.md),
including the DINOv3 backbone and SAM2 installation. Use the root
`scripts/setup_backbones.sh` for backbone sources. DINOv3 lives in
`third_party/dinov3`, its weights in `third_party/weights`, and SAM2 in `sam2/`,
all relative to the repository root.

Then run from the repository root:

```bash
python -m pip install -r blender/remote/server/requirements.txt
python blender/remote/server/tools/init_env.py
cd blender/remote/server
```

`init_env.py` creates a local `.env` with random
`CAST_API_KEY` and `CAST_ADMIN_PASSWORD` values. Configure the GPU and resource
paths there; see [directories and configuration](docs/DEPLOYMENT.md#directories-and-configuration).

## Model Setup

Download the SAM2 and person detection weights:

```bash
python tools/fetch_public_weights.py --confirm-model-terms
```

Obtain the DINOv3 ViT-L/16 weights and a CAST causal checkpoint as described in
the [model guide](docs/MODELS.md). The default DINOv3 path is
`<repository>/third_party/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
Set `CAST_DINOV3_WEIGHTS` in `.env` to use a different path.

Select CAST-B or CAST-L to match your causal checkpoint, then prepare and
activate it. For CAST-L:

```bash
python tools/prepare_checkpoint.py \
  --model CAST-L --checkpoint /path/to/CAST_L_dinov3_causal.pt \
  --model-id cast-l --activate
```

`--activate` selects the model for the next API start.
Restart a running API to load a newly activated model.

## Start and Connect

Run these commands from `blender/remote/server/`:

```bash
python tools/manage.py api start
python tools/manage.py admin start
curl --noproxy '*' http://127.0.0.1:8080/healthz
```

The API listens on `http://127.0.0.1:8080` and the management console on
`http://127.0.0.1:8081` by default. Sign in to the console with
`CAST_ADMIN_USER` and `CAST_ADMIN_PASSWORD` from `.env`.
Clients authenticate with `CAST_API_KEY`.

To connect from another machine, create an SSH tunnel:

```bash
ssh -N -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081 user@gpu-host
```

Use `http://127.0.0.1:8080` as the client API URL and open
`http://127.0.0.1:8081` in a browser. Network access guidance is in the [security guide](docs/SECURITY.md).

Before a character is registered,
`rig_ready=false` in `/healthz` is expected. See the
[Blender client guide](../client/README.md) for integration examples and the
[API reference](docs/API.md) for request and response formats.

To stop the services:

```bash
python tools/manage.py api stop
python tools/manage.py admin stop
```

## Documentation

- [Deployment and maintenance](docs/DEPLOYMENT.md): settings, logs, and model changes.
- [Models and configurations](docs/MODELS.md): weights, paths, and architecture matching.
- [Blender client integration](../client/README.md): rig export, frame submission, and pose application.
- [HTTP API](docs/API.md): sessions, frame sequences, resets, and error recovery.
- [Validation](docs/VALIDATION.md): GPU checks and end-to-end validation.
- [Security](docs/SECURITY.md): credentials, network access, and image history.

See the repository's [LICENSE](../../../LICENSE) and
[LICENSE-3RD-PARTY](../../../LICENSE-3RD-PARTY) for code licensing and dependency notices.
