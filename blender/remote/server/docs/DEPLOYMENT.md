# Deployment and Maintenance

Run commands from `blender/remote/server/` with your Python environment activated.
Keep the complete CAST repository layout on the server.

## Linux Setup

Install the shared CAST environment using the [main installation guide](../../../../docs/installation.md),
including the root CUDA operators, DINOv3 sources/weights, and SAM2. Then follow
the [server README](../README.md) to add the HTTP dependencies and prepare the
service settings and checkpoint.

After installation, follow the [GPU operator checks](VALIDATION.md#gpu-operator-checks)
in that same environment.

## Directories and Configuration

```text
.env                     Local settings and generated credentials
assets/weights/          External pretrained weights
assets/models/<id>/      Validated runtime.pt, config.yaml, and manifest.json
assets/active_model.json Model selected for the next API start
var/cache/               Compilation caches, separated by model ID
var/runtime/             Local PID records and validated rig data
var/logs/                Service, administrator, and HTTP audit logs
var/admin/local/         SQLite database on a local disk
```

DINOv3 sources/weights are shared from `<repository>/third_party/`; SAM2 is
shared from `<repository>/sam2/`, as installed by the main repository guide.

`.env` accepts `KEY=VALUE` entries without shell evaluation or interpolation.
Paths support `~`; relative paths resolve from `blender/remote/server/`.
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
| `CAST_DINOV3_REPO` / `CAST_DINOV3_WEIGHTS` | Defaults to the root `third_party/dinov3` and `third_party/weights/` |
| `CAST_SAM2_REPO` | Defaults to the root `sam2/` source directory |
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

After starting, check the logs and `/healthz` for model readiness.

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

3. Drain and close the session, restart the API, check the selected model in
   `/healthz`, and register the rig again.
4. To roll back, restore the previous trusted active manifest and restart during
   maintenance. Preparation preserves an existing pointer as
   `previous_active_model.json` in the new model directory.

## Updating Source Code

Drain client sessions and stop the API before updating the checkout. Deploy the
new CAST source revision as a whole, retain external assets and instance state,
rebuild the root `ops/` extension if its sources changed, and run
`python tools/check_acceleration.py` before
restarting. Follow the GPU validation guide after model/runtime changes. Prepare
checkpoints again if the model architecture or configuration changes.

## Benchmarking and Diagnostics

Use an idle test instance and follow the
[video validation procedure](VALIDATION.md#blender-and-video-validation)
for benchmark commands, output fields, and measurement boundaries.

To investigate slow inference,
compare receive, queue, decode, detector, SAM, and model timings under the same
model, rig, and video. Check acceleration with `tools/check_acceleration.py`.

For HTTP 503 responses, use the request ID, UTC time, and sequence number to
correlate [origin audit records](API.md#http-connection-admission).
For crop problems, inspect `tracking.detector_bbox`, `sam_bbox`, and
`crop_source`, or enable short-term image history as described in
[security settings](SECURITY.md#image-history).
