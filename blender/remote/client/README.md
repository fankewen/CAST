# CAST Remote Live Client

## Installation

Use Blender 4.2 or later and a separate Python 3.10+ environment. Camera preview
requires Tkinter. First, have the server operator complete the
[server setup](../server/README.md) and provide a reachable API URL and API key.

With your external Python environment activated, run from the repository root:

```powershell
python -m pip install -r blender/remote/client/requirements.txt
python blender/remote/client/build_addon.py --output "$env:TEMP/cast_remote_live.zip"
python -c "import sys; print(sys.executable)"
```

These are Windows PowerShell commands. The last command prints the interpreter
path to use in Blender. For another shell, replace the output path with a new
local path. Choose a different output path if the ZIP already exists.

1. In Blender, open **Edit > Preferences > Add-ons > Install from Disk** and
   select the ZIP.
2. Enable **CAST Remote Live**.
3. Open the 3D Viewport sidebar: press **N**, then select **CAST Remote**.
4. Set **Python** to the interpreter path printed above, enter the server URL
   and API key, and follow [Blender Workflow](#blender-workflow).

Leave **Worker** at its automatically configured path. Set **Python** to the
external `python.exe` with the dependencies above installed.

## Configuration

The default configuration is `.cast_remote/config.json` under the user's home directory, typically `%USERPROFILE%\.cast_remote\config.json` on Windows. Enabling the addon creates the file if missing; initialization never overwrites an existing configuration. Set `CAST_REMOTE_CLIENT_HOME` before starting Blender to change the default data directory, or select a configuration file in the panel's Config field.

Use [config.example.json](config.example.json) as the reference:

| Setting | Purpose |
| --- | --- |
| `server_url` | HTTP(S) inference API URL; defaults to `http://127.0.0.1:8080` for an SSH tunnel |
| `api_key` | The server's `CAST_API_KEY`; the environment variable of that name can also supply it |
| `target_mode` | `human` detects a person in the camera image; `generic` processes the full frame for other subjects. Select by the input subject, independently of the output character's species |
| `camera` | Camera index or an OpenCV-supported video source |
| `fps` | Capture and submission rate, default 24; adjustable live through Input FPS |
| `capture_width` / `capture_height` | Requested camera resolution |
| `square_mode` / `square_size` | Center cropping (`crop`) or padding (`pad`) and the resulting square image size |
| `pipeline_depth` | Maximum in-flight HTTP requests, default 4, range 1–10; must not exceed server capacity |
| `request_timeout_seconds` | Timeout for session creation, warm-up, and frame requests; increase for slow initial model warm-up |
| `input_port` / `pose_port` / `control_port` | Local frame, pose, and control ports, default 5602 / 5599 / 5601; each must be distinct |
| `lens_correct` / `lens_profile` | Optional lens correction and calibration profile; relative paths resolve from the configuration directory |

An empty panel URL uses the configuration file; entering a URL updates that file. A nonempty panel API Key overrides the configuration and environment key and is not saved in the `.blend` file. Input Video, Input FPS, Pose Port, and Control Port use the panel's current values at startup. Before changing the server, input mode, or configuration directory, click Stop and then import the FBX again.

Find capture and transfer logs beside the configuration file in
`camera_daemon.log` and `worker.log`. Back up `fbx_user_settings.json` and
`fbx_user_settings_models/` to preserve per-character pose settings.

## Blender Workflow

1. Set Python, Config, and the connection parameters.
2. Click **Import FBX**, select the character, and wait for `remote skeleton ready`.
3. Enable Camera Preview if needed, click **Start Camera**, and wait for stable capture.
4. Click **Start Transfer**. Frames are submitted in order and results are applied in submission order. When the server reports `motion_valid=false`, the previous pose is retained.
5. **Pause / Continue** pauses or resumes frame submission while retaining the session. **Reset Cache** drains in-flight requests before clearing temporal state. **Stop** ends transfer, releases the remote session, and closes the camera process started by the addon. Import the character again to start a new session.

After a network failure, wait for reconnection. Old frames are not replayed.
Check `worker.log` if transfer does not resume. If initial rig preparation times
out, increase `request_timeout_seconds`, click **Stop**, and import the FBX again.

Advanced pose settings include skeleton axis conventions, root mode, rotation offsets, fixed bones, crab leg synchronization, and rotation smoothing. Settings are saved per FBX. Reimport the FBX after changing Skeleton Input.

Use one configuration directory per client instance. Close manually launched
preview windows yourself, and avoid opening the same camera from two processes.
Stop other pose receivers using port 5599, or choose an unused **Pose Port**.

## Standalone Camera and Calibration

To launch the camera outside the Blender panel:

```powershell
python blender/remote/client/camera_standalone.py --config "$env:USERPROFILE/.cast_remote/config.json"
```

Add `--no-preview` for capture without a window. To calibrate the lens, use a
recording containing real straight lines:

```powershell
python blender/remote/client/calibrate_from_video.py recording.mp4 --camera 0 --out "$env:USERPROFILE/.cast_remote/camera_profile.json"
```

Then set `lens_correct: true`. Check the corrected preview before using it for inference.

## Export a Rig for Benchmarking

For standalone rig export, select the target armature and set an active camera in the `.blend` file, then run from the repository root:

```text
blender character.blend --background --python blender/remote/client/export_rig_json.py -- --output /path/to/rig/rig.json
```

## Validation

Verify the complete workflow with Blender, a camera, and a GPU server: import different rigs, preview capture, drive continuous motion, pause/resume, reset the cache, recover from a connection loss, stop, and import again.
