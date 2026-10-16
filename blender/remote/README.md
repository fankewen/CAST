# CAST Remote Inference

Run CAST on a GPU server and send skeletons and images from Blender over HTTP.
The server handles person detection, SAM segmentation, and causal inference,
then returns joint rotations. Blender is only required on the client machine.
An independent management console provides service status, resource metrics,
logs, and image history.

The service uses the current CAST checkout's `models/`, `inference/`, `utils/`,
`data/`, `visualization/`, and `ops/` directly. There is no separate model-code
snapshot to synchronize. Deploy the complete checkout or an archive produced
by `tools/build_release.py`; copying only `blender/remote/` is not supported.

## Requirements

Linux, Python 3.10, an NVIDIA GPU and driver compatible with CUDA 12.8,
CUDA Toolkit 12.8, Git, and a C/C++ compiler. Activate your Python environment
before installation. The installer installs PyTorch 2.7.0+cu128 and builds the
CUDA operators using that environment's `python`.
See the [deployment guide](docs/DEPLOYMENT.md) for details.

## Installation

From the repository root:

```bash
cd blender/remote
python tools/init_env.py
python tools/fetch_dependencies.py --accept-dinov3-license

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
bash tools/install.sh
```

Set `CUDA_HOME` to the Toolkit path on your server. `init_env.py` generates
`.env` with random values for `CAST_API_KEY` and `CAST_ADMIN_PASSWORD`.
Configure the GPU and resource paths in `.env`; see
[directories and configuration](docs/DEPLOYMENT.md#directories-and-configuration)
for the available settings.

## Model Setup

Download the SAM2 and person detection weights:

```bash
python tools/fetch_public_weights.py --confirm-model-terms
```

Obtain the DINOv3 ViT-L/16 weights and a CAST causal checkpoint as described in
the [model guide](docs/MODELS.md). The default DINOv3 path is
`assets/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
Set `CAST_DINOV3_WEIGHTS` in `.env` to use a different path.

Select CAST-B or CAST-L to match your causal checkpoint, then prepare and
activate it. For CAST-L:

```bash
python tools/prepare_checkpoint.py \
  --model CAST-L --checkpoint /path/to/CAST_L_dinov3_causal.pt \
  --model-id cast-l --activate
```

The tool checks tensor keys, shapes, and values before saving the runtime model
under `assets/models/`. `--activate` selects the model for the next API start.
Restart a running API to load a newly activated model.

## Start and Connect

Run these commands from `blender/remote/`:

```bash
python tools/run_tests.py
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
`http://127.0.0.1:8081` in a browser. HTTPS reverse proxy and Docker setup are
covered in the [deployment guide](docs/DEPLOYMENT.md).

A client creates a session with its JSON skeleton, waits for warm-up to finish,
then submits JPEG frames in capture order. Before a character is registered,
`rig_ready=false` in `/healthz` is expected. See the
[Blender client guide](docs/BLENDER.md) for integration examples and the
[API reference](docs/API.md) for request and response formats.

To stop the services:

```bash
python tools/manage.py api stop
python tools/manage.py admin stop
```

## Directory Layout

| Path | Purpose |
| --- | --- |
| `service.py` / `run_origin.py` | Inference API and entry point |
| `admin/` | Management console |
| `client/` | Blender rig export, HTTP client, and pose driver |
| `../../models/`, `../../inference/`, `../../utils/`, `../../data/`, `../../visualization/`, `../../ops/` | Shared current CAST runtime and CUDA operators |
| `../rig_adapter.py` | Rig export and pose application shared with the local addon |
| `configs/` | CAST-B and CAST-L configurations for causal inference |
| `tools/` / `tests/` | Installation, model conversion, process management, packaging, and tests |
| `assets/` / `var/` | Local model assets and runtime data created during setup and operation |

## Documentation

- [Deployment and maintenance](docs/DEPLOYMENT.md): settings, logs, model changes, and Docker.
- [Models and configurations](docs/MODELS.md): weights, paths, and architecture matching.
- [Blender client integration](docs/BLENDER.md): rig export, frame submission, and pose application.
- [HTTP API](docs/API.md): sessions, frame sequences, resets, and error recovery.
- [Validation](docs/VALIDATION.md): tests without weights, GPU checks, and end-to-end validation.
- [Security](docs/SECURITY.md): credentials, network access, and image history.

See [LICENSE](LICENSE) for the code license and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for external dependency and
model terms.
