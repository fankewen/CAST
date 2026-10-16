# Deployment and Maintenance

Run commands from `blender/remote/` with your Python environment activated.
Keep the complete CAST repository layout on the server. Model imports and CUDA
builds use the repository root, not a deployment-specific copy of the code.

## Linux Setup

Follow the [README](../README.md) for installation and model preparation.
The reference environment uses Python 3.10, PyTorch 2.7.0+cu128,
TorchVision 0.22.0, xFormers 0.0.30, FlashAttention 2.8.3, and Uvicorn 0.34.2.
`tools/install.sh` installs dependencies into the active Python environment.
External source revisions are pinned in `third_party/dependencies.json`.

Install a compatible NVIDIA driver, CUDA Toolkit 12.8 with `nvcc`, Git, and a
C/C++ compiler before running the installer. FlashAttention is built from
source; allow sufficient build time and host memory.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=4
bash tools/install.sh
python tools/check_acceleration.py
```

Set the Toolkit path and GPU index for your machine. The installer detects the
visible GPU architecture unless `TORCH_CUDA_ARCH_LIST` is already set.
`check_acceleration.py` executes FlashAttention, xFormers attention, and CAST
CUDA operators. The inference service uses BF16. Memory requirements depend on
the model, rig, compilation, and graph capture.

## Directories and Configuration

```text
.env                     Local settings and generated credentials
third_party/{sam2,dinov3} External source checkouts
assets/weights/          External pretrained weights
assets/models/<id>/      Validated runtime.pt, config.yaml, and manifest.json
assets/active_model.json Model selected for the next API start
var/cache/               Compilation caches, separated by model ID
var/runtime/             Local PID records and validated rig data
var/logs/                Service, administrator, and HTTP audit logs
var/admin/local/         SQLite database on a local disk
```

`.env` accepts `KEY=VALUE` entries without shell evaluation or interpolation.
Paths support `~`; relative paths resolve from `blender/remote/`.
Process environment variables take precedence over `.env`.
Each instance needs its own state directory. If state is on a network
filesystem, set `CAST_ADMIN_STATE_DIR` to a separate local disk directory for
SQLite.

| Variable | Default or purpose |
| --- | --- |
| `CAST_GPU` | `0`; an existing `CUDA_VISIBLE_DEVICES` takes precedence |
| `CAST_HOST` / `CAST_PORT` | `127.0.0.1` / `8080` |
| `CAST_ADMIN_HOST` / `CAST_ADMIN_PORT` | `127.0.0.1` / `8081` |
| `CAST_STATE_DIR` / `CAST_ASSETS_DIR` | `var` / `assets` |
| `CAST_ADMIN_STATE_DIR` | `var/admin/local` |
| `CAST_INTERNAL_API_URL` | API address used by the management console |
| `CAST_PUBLIC_API_URL` | API address displayed to clients |
| `CAST_DINOV3_REPO` / `CAST_DINOV3_WEIGHTS` | DINOv3 source and pretrained weights |
| `CAST_SAM2_REPO` | SAM2 source directory |
| `CAST_SAM_CHECKPOINT` / `CAST_PERSON_CHECKPOINT` | SAM2 and person detection weights |
| `CAST_FULL_STEP_GRAPH` | `1`; set to `0` to diagnose graph-related issues |
| `CAST_PIPELINE_CAPACITY` | `10`; supported range is 1 through 10 |
| `CAST_HTTP_ACTIVE_LIMIT` / `CAST_HTTP_TCP_LIMIT` | `64` / `256`; leave room for health requests above the active limit |
| `CAST_HISTORY_ENABLED` | `0`; set to `1` to sample incoming images |

## Service Management

```bash
python tools/manage.py api start
python tools/manage.py api status
python tools/manage.py admin start
tail -f var/logs/service.log
curl --noproxy '*' http://127.0.0.1:8080/healthz
```

The manager tracks processes from this checkout and checks their hostname,
start time, and entry point. A successful `start` means the process was created;
check the logs and `/healthz` for model readiness.

To run in the foreground, use `bash run_service.sh` or
`python admin/run_admin.py`. Foreground processes have no manager PID record,
so process metrics may be unavailable. Run one Uvicorn worker per instance.

Drain and close client sessions before stopping the services:

```bash
python tools/manage.py api stop
python tools/manage.py admin stop
```

## Changing Models

1. Select `CAST-B` or `CAST-L` to match the causal checkpoint.
2. Prepare the checkpoint under a new model ID:

   ```bash
   python tools/prepare_checkpoint.py --model CAST-L \
     --checkpoint /path/to/CAST_L_dinov3_causal.pt \
     --model-id cast-l-updated --activate
   ```

3. The tool validates tensor keys, shapes, loaded values, saved values, and
   hashes. Activation selects the model for the next start.
4. Drain and close the session, restart the API, check the selected model in
   `/healthz`, and register the rig again.
5. To roll back, restore the previous trusted active manifest and restart during
   maintenance. Preparation preserves an existing pointer as
   `previous_active_model.json` in the new model directory.

## Updating Source Code

Drain client sessions and stop the API before updating the checkout. Deploy the
new CAST source revision as a whole, retain external assets and instance state,
rebuild the root `ops/` extension if its sources changed, and run
`python tools/run_tests.py` and `python tools/check_acceleration.py` before
restarting. Follow the GPU validation guide after model/runtime changes. Existing
prepared checkpoints remain external assets; prepare them again if the model
architecture or configuration changes. Do not copy an old runtime directory over
the new checkout.

## Docker

Generate `.env`, fetch the external source repositories, and prepare the model
assets first. The image builds CUDA operators; assets and credentials are
provided at deployment time.
Compose uses the repository root as the build context and
`blender/remote/Dockerfile` as the recipe. Its `Dockerfile.dockerignore` includes
only the shared runtime and remote-service sources, excluding credentials,
assets, state, and build artifacts.

```bash
# Example build arguments for Hopper; choose the architecture for your GPU.
docker compose build --build-arg CUDA_ARCH=9.0 --build-arg FLASH_ARCH=90 api
docker compose run --rm --no-deps api python tools/check_acceleration.py
docker compose up -d
docker compose logs -f api
```

Docker Compose requires GPU support and the NVIDIA container runtime. The
default ports bind to host loopback. Assets are mounted read-only, state is
writable, and SQLite uses a local volume. Custom absolute paths in model
configurations must be accessible inside the container. `CAST_GPU` refers to a
GPU visible inside the container.

The admin container has a separate process namespace and no GPU access by
default. API process and device metrics may therefore be unavailable. Use the
health response, shared audit logs, and image history for service diagnostics.
Read container stdout with `docker compose logs`. After building, follow the
[validation guide](VALIDATION.md) inside the target environment.

## Benchmarking and Diagnostics

Use an idle test instance:

```bash
python tools/benchmark.py --rig /path/to/rig.json \
  --video /path/to/video.mp4 --window 10 --frames 300 --target-mode human
```

Results are saved to `var/logs/benchmark-local.json`. Timing includes JPEG
encoding, HTTP, and inference, but excludes video preloading and Blender
playback. Initial visual reference binding is included; rig compilation is
excluded. The tool caches up to 300 video frames and loops them for longer runs.
Inspect `valid_motion_frames` alongside throughput.

The upload window hides network latency. To investigate slow inference,
compare receive, queue, decode, detector, SAM, and model timings under the same
model, rig, and video. Check acceleration with `tools/check_acceleration.py`.

For HTTP 503 responses, use the request ID, UTC time, and sequence number to
correlate [origin audit records](API.md#http-connection-admission).
For crop problems, inspect `tracking.detector_bbox`, `sam_bbox`, and
`crop_source`, or enable short-term image history as described in
[security settings](SECURITY.md#image-history).
