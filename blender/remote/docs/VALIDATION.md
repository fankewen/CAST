# Validation

Run the following commands from `blender/remote/`.

## Tests Without Model Weights

```bash
python tools/run_tests.py
```

The suite covers frame ordering, queue capacity, HTTP connection admission,
rig validation, human crop selection, image history, model manifests,
administrator authentication, and source packaging. It uses temporary
directories and loopback ports and requires neither weights nor a running
inference service.

## GPU Operator Checks

After installation, run:

```bash
python tools/check_acceleration.py
```

This executes FlashAttention, xFormers attention, and the CAST CUDA operators.
Rebuild and repeat the checks after changing the GPU, PyTorch, or CUDA version.

## Blender and Video Validation

Prepare the models and start the service. Export a rig using the
[Blender client guide](BLENDER.md), then check the service with a real video:

```bash
curl --noproxy '*' http://127.0.0.1:8080/healthz
python tools/benchmark.py \
  --rig /path/to/rig.json --video /path/to/video.mp4 \
  --window 10 --frames 300 --target-mode human
```

Results are saved to `var/logs/benchmark-local.json`. Check the valid motion
frame count, request timings, and tracking status together. Reported throughput
includes JPEG encoding and HTTP, but excludes Blender playback.

In Blender, check coordinate axes, bone lengths, and poses while standing,
turning, and raising each arm. Verify recovery after changing characters,
disconnecting the network, and stopping playback. See the
[validation checklist](BLENDER.md#validation-checklist) for details.
