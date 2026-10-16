# Validation

Run the following commands from `blender/remote/server/`.

## GPU Operator Checks

After installation, run:

```bash
python tools/check_acceleration.py
```

This executes FlashAttention, xFormers attention, and the CAST CUDA operators.
Rebuild and repeat the checks after changing the GPU, PyTorch, or CUDA version.

## Blender and Video Validation

Prepare the models and start the service. Export a rig using the
[Blender client guide](../../client/README.md), then check the service with a real video:

```bash
curl --noproxy '*' http://127.0.0.1:8080/healthz
python tools/benchmark.py \
  --rig /path/to/rig.json --video /path/to/video.mp4 \
  --window 10 --frames 300 --target-mode human
```

Use an idle test instance. Results are saved to `var/logs/benchmark-local.json`.
Compare `frames` with `valid_motion_frames`, and inspect throughput and request
latencies. Timing includes JPEG encoding, HTTP, inference, and initial visual
reference binding; it excludes video preloading, rig compilation, and Blender
playback. The tool caches up to 300 video frames and loops them for longer runs.
Use the service logs or API responses to investigate tracking status; the
benchmark report does not include per-frame tracking details.

In Blender, check coordinate axes, bone lengths, and poses while standing,
turning, and raising each arm. Verify recovery after changing characters,
disconnecting the network, and stopping playback. See the
[validation checklist](../../client/README.md) for details.
