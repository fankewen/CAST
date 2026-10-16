# Blender Client Integration

`client/` provides rig export, an HTTP client, and a pose driver. Blender
captures images, encodes JPEG, uploads frames, and applies returned rotations.
The server receives rig JSON and images; character files and bind transforms
remain on the client machine.
The rig implementation is shared with the local addon in `blender/rig_adapter.py`.
Keep this repository layout when running the exporter; it locates the shared
module automatically. An integrating Blender addon should include this module
and `client/pipeline_client.py`, without server or model dependencies.

## 1. Export the Rig

Import the character into Blender, select the armature, set `scene.camera`, and
save the scene. Then run:

```bash
blender character.blend --background --python client/export_rig_json.py -- --output /path/to/rig/rig.json
```

The saved scene must have a selected armature and an active camera. An addon can
also call `build_rig_asset()` and convert the result to JSON as the exporter does.
The output contains `rig.json`, `static.npy`, and `bind_transforms.npz`.
Upload the JSON rig; retain the other two files locally.

Rigs support one root and up to 150 joints in topological parent order. Exclude
non-bone control objects. The driver expects rigid rest transforms and uniform
scale; prepare rigs with constraints, drivers, or nonuniform scale accordingly.
Normalize actual dimensions to meters before export and verify the armature
and camera transforms. Re-export and warm up after changing the rest pose,
scale, or camera reference.

## 2. Create a Session in a Worker Thread

`client/pipeline_client.py` uses only the Python standard library.

```python
import json
from pipeline_client import PipelinedMotionClient

client = PipelinedMotionClient(api_url, api_key, window=10, timeout=900)
meta = client.start(json.loads(rig_json_text), target_mode="human")
# Rig preparation, compilation, and graph capture finish before start returns.
```

Read the API key from local user configuration or credential storage. The first
valid subject image binds the visual reference. Session warm-up does not
require sending a black frame.

## 3. Submit and Consume Frames

- Use one client per camera and session. Keep `window` at or below 10 and the
  capacity returned by the server.
- Call `submit` serially in capture order. Run JPEG encoding and HTTP work in
  background threads; keep every `bpy` operation on Blender's main thread.
- Store futures in a deque and consume them in submission order. Completion
  order can differ from animation order.
- `BufferError` means no frame was submitted and no sequence was allocated.
  Replace an older unsent image if necessary, while preserving submitted frames.
- Validate the response sequence, epoch, joint count, and finite values.
  With `motion_valid=false`, hold the current pose or display tracking status.
- Pass completed, ordered motion to a main-thread queue or Blender timer.
  Keep blocking `future.result()` calls outside the UI thread.
- Before stopping or changing characters, stop capture, drain futures, and
  close the client. Export and warm up the new rig in a new session.
- Follow the [API recovery procedure](API.md#pipeline-and-recovery) after
  ambiguous network failures.

## 4. Apply Rotations

```text
rot6d[j] = [R00, R10, R20, R01, R11, R21]
joint order = registered joint_names
root rotation = OpenCV camera coordinates: x right, y down, z forward
child rotation = parent-relative local rotation
root translation = client rest/bind strategy; not predicted
```

Use `RigDriver` from `blender/rig_adapter.py` (relative to the repository root).
It orthogonalizes the 6D rotation,
combines local rotations with rest translations, performs hierarchical forward
kinematics, converts camera coordinates into Blender armature coordinates,
and applies the inverse rest transform to obtain `matrix_basis`.

```python
from rig_adapter import RigDriver

driver = RigDriver(armature_object, bind_transforms_path)
# Apply poses only on Blender's main thread.
if result["motion_valid"]:
    driver.apply_rot6d(result["motion"]["rot6d"])
```

Preserve the registered bone order and matrix column convention. Directly
assigning converted quaternions to `pose_bone.rotation_quaternion` skips the
required rest and coordinate transforms.

## Validation Checklist

1. Confirm that registration returns the exported joint order and successful
   rig and graph preparation.
2. Check standing, turning, and raising each arm for mirroring, axis offsets,
   and changes in bone length.
3. Verify `bbox_xyxy=null` and invalid motion with no person in view, followed
   by tracking recovery when a person returns. Tracking does not provide stable
   identity selection among multiple people.
4. Exercise `window=10` over a high-latency connection and verify continuous
   sequences, then test disconnect, drain, reset, and new-epoch handling.
5. Report capture, HTTP completion, valid motion, and UI playback rates
   separately, together with resolution, JPEG quality, network latency, and rig
   joint count.
6. Confirm image retention requirements before enabling history; capture is
   disabled by default.
