# Image to Motion API / v1.2.0

## Session Lifecycle

1. Poll `GET /healthz` until `ready=true`. This endpoint requires no API key;
   restrict access to its detailed status when exposing the service.
2. Create a session with `POST /v1/sessions` and a JSON rig. Wait for skeleton
   preparation, streaming compilation, and fixed-rig CUDA Graph validation.
3. Send raw JPEG or PNG bytes, starting at `seq=0`, with the returned `epoch`.
4. Stop submitting, wait for every pending future, then reset or delete the
   session. Reset returns a new epoch and restarts the sequence at zero.

All endpoints except `/healthz` require `X-API-Key`. Each process owns one GPU
causal stream. A second active session receives HTTP 409. The idle session TTL
is 120 seconds.

## Rig Registration and Warm-Up

Send `POST /v1/sessions` with `Content-Type: application/json`:

```json
{
  "target_mode": "human",
  "rig": {
    "rig_id": "synthetic-example",
    "joints": 2,
    "joint_names": ["root", "child"],
    "parents": [-1, 0],
    "rest_translations": [[0, 0, 3], [0, 0.3, 0]],
    "rest_rotations_quat": [[0, 0, 0, 1], [0, 0, 0, 1]],
    "metric_scale": 6.6666667,
    "coordinate_system": "opencv_right_handed_ydown_zforward_meters",
    "units": "meters"
  }
}
```

This two-joint example illustrates the format. The rig is required and must
satisfy the following constraints:

- `joints` is an integer from 1 through 150. Joint names are unique. Index zero
  is the only root; every other parent index is smaller than its joint index.
- `rest_translations` has shape `[J,3]` in meters. The root uses OpenCV camera
  coordinates; child translations are relative to their parents.
- `rest_rotations_quat` has shape `[J,4]` with normalized **xyzw** quaternions
  under the same hierarchy. Convert Blender's wxyz order before uploading.
- `metric_scale` is positive. The exporter computes `2 / diagonal` from the
  camera-space rest skeleton's axis-aligned bounding box.
- All numeric values are finite. Bind transforms remain on the client.
- `target_mode="human"` uses person detection to guide SAM. `"generic"` uses
  single-subject SAM tracking without semantic person detection.

The response includes `session_id`, `epoch`, `joint_names`, `joints`,
`next_seq=0`, `pipeline_capacity`, `model`, `acceleration`, and
`root_translation_predicted=false`. Model metadata identifies `CAST-B` or
`CAST-L` through `model_family`. The default pipeline capacity is 10.

Repeated sessions using the same rig can reuse skeleton preparation, while
temporal state is cleared for each stream. Changing rigs triggers new graph
capture and validation. Allow time for compilation; the example clients use a
900-second request timeout.

## Frame Submission

```text
POST /v1/sessions/{session_id}/frames?seq=0&epoch={epoch}&timestamp_ms=0
X-API-Key: <local-secret>
Content-Type: image/jpeg

<raw JPEG bytes; not multipart or base64 JSON>
```

Use `image/png` for PNG. Requests accept up to 8 MiB and `1920*1080` pixels.
The optional `timestamp_ms` must be finite. Input dimensions describe the
camera image; preprocessing produces a 256-pixel square model input.
Bounding boxes use xyxy coordinates in the original image's pixels.

Example response:

```json
{
  "session_id": "...",
  "seq": 0,
  "epoch": "...",
  "timestamp_ms": 0,
  "motion": {"rot6d": [[1,0,0,0,1,0],[1,0,0,0,1,0]]},
  "motion_valid": true,
  "bbox_xyxy": [100, 20, 500, 700],
  "tracking": {"mode":"human", "status":"tracking", "crop_source":"detector_fallback"},
  "timing_ms": {"receive_ms":10, "queue_wait_ms":2, "server_pipeline_ms":45}
}
```

These values illustrate the format. Timing fields also cover decoding, person
detection, SAM, cropping, model execution, host transfer, and history capture.
`server_pipeline_ms` measures origin receive, queue, and processing time;
client network round trips and Blender playback are measured separately.

`motion.rot6d` has shape `[J,6]` in the registered joint order. Each entry
concatenates the first two **columns** of a local rotation matrix. The root uses
OpenCV camera coordinates; child rotations are parent-relative. Root translation
is not predicted. Apply rest transforms, forward kinematics, and coordinate
conversion as described in the [Blender guide](BLENDER.md).

When no person is detected, the sequence is consumed and HTTP 200 is returned
with `motion_valid=false`, `bbox_xyxy=null`, and the previous or rest pose.
The background is not bound as a reference and motion KV state does not advance.
Tracking recovery binds a new visual reference. Handle a null bounding box and
exclude invalid motion responses when reporting effective motion frame rate.

## Pipeline and Recovery

- Several frames may upload, decode, or wait concurrently. The GPU advances
  strictly in sequence order on one worker.
- Capacity includes uploading, decoding, ready, and running frames. Sequence
  numbers must lie in `[next_seq, next_seq+capacity)`.
- Keep the client window at or below `pipeline_capacity`. A larger window can
  hide network latency; smaller windows reduce queued frames.
- Allocate sequence numbers only when submitting. Keep the latest unsent image
  when the window is full; preserve every already submitted sequence.
- `GET /v1/sessions/{id}/status` returns pending frames, next sequence, epoch,
  completed frame count, last completed sequence, and failure state.
- `POST /v1/sessions/{id}/reset` has no body and requires a drained pipeline.
  A successful reset invalidates responses from the previous epoch.
- `DELETE /v1/sessions/{id}` also requires a drained pipeline.

| Status | Meaning and recovery |
| --- | --- |
| 401 | Invalid API key; check local configuration |
| 413 / 415 / 422 | Invalid size, format, rig, or parameter; correct the input |
| 429 | Pipeline capacity reached or GPU busy during a control request; stop increasing the window |
| 409 | Sequence or epoch mismatch, missing frame, occupied session, or undrained pipeline; inspect status and drain |
| 408 | Upload exceeded 10 seconds; the frame was not consumed and dependent frames fail |
| 500 | Inference failed; drain and reset, or recreate the session |
| 503 | HTTP/TCP admission, shutdown, or gateway failure; identify the responding layer |

A missing sequence times out after five seconds with
`missing seq=...; gap timeout`. A network timeout can occur after a frame has
already been consumed. Stop submission, wait for outstanding requests, inspect
status, and drain before resetting. POST requests have no general replay cache.

## HTTP Connection Admission

`run_origin.py` pins Uvicorn 0.34.2 and uses independent admission controls:

- Up to 64 active non-health requests by default, plus four health slots.
- Up to 256 TCP connections by default. At capacity, completed idle keep-alive
  connections are reclaimed before rejecting a new connection.
- Request headers time out after 10 seconds; idle keep-alive lasts five seconds.
- Health slots share the TCP pool and remain subject to its connection limit.

Origin responses include `X-CAST-Origin`, `X-CAST-Request-ID`, `X-CAST-Policy`,
and `X-CAST-Response-Layer`. Rejections also include a reason.
`var/logs/http_origin.jsonl` records UTC time, sequence, epoch, request and
connection IDs, optional `x-rio-seq`, and connection or task counts.
Forwarded headers are untrusted diagnostic data.

For a 503 without a matching origin record, use the response headers and time
to investigate the reverse proxy or gateway.
