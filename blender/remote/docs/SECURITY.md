# Security and Privacy

## Access and Configuration

- The API and management console bind to loopback by default. `init_env.py`
  generates random API and administrator credentials, preserves existing
  credentials, and keeps their values out of terminal output.
- API requests use `X-API-Key`; the console uses a separate login cookie. The
  console provides read-only monitoring.
- Input endpoints accept JPEG, PNG, and validated rig JSON. Model weights and
  configurations are prepared locally by the operator.
- Checkpoint loading uses `torch.load(weights_only=True)`.
- Public deployments require TLS, access controls, connection and request
  limits, and dependency updates. Run the origin HTTP tests when upgrading the
  pinned Uvicorn/H11 stack.

Set `CAST_ADMIN_SECURE_COOKIE=1` behind an HTTPS reverse proxy. Keep credentials
out of URLs, logs, and public issues. Restrict access to asset and state
directories; `umask 077` is suitable for the startup shell. Model configurations
can construct Python objects, so only use trusted configurations, weights, and
active manifests.

## Image History

`CAST_HISTORY_ENABLED=0` disables image capture by default. Before enabling it,
confirm consent and retention requirements for the input images.

History samples subsequent input at most once per second by default, with
24-hour, 3,600-record, and 2 GiB retention limits. A bounded background queue
writes samples. Stored data may include original images, person boxes, SAM
masks, crops, model inputs, poses, and rig labels. Frames without a detected
person may have no crop.

Image access requires administrator login. Disabling capture stops new samples;
it does not erase existing data. The operator manages backups, access auditing,
and cleanup. Keep `var/` out of source distributions.

Resource monitoring samples every five seconds and retains metrics for seven
days. These samples describe resource usage rather than valid motion frame rate.

## Logs

Origin audit records contain sequence numbers, epochs, connection IDs, times,
status, and peer or forwarding diagnostics. Image bodies and API keys should
remain out of logs. The console masks known credentials and common token
fields; review diagnostic exports before sharing them.

Audit logs rotate automatically. Configure rotation and retention for process
manager stdout logs. Share only the necessary redacted excerpts.

## Source Packaging

```bash
python tools/build_release.py --output /path/outside/source/CAST_remote.zip
```

The builder uses a source allowlist, scans candidate text, and generates a
SHA-256 manifest. Credentials, runtime state, assets, environments, external
checkouts, compiled binaries, weights, images, and logs are excluded.
The archive preserves the repository layout: shared runtime code is at its
root, and the service is under `blender/remote/`. Enter that directory after
extracting the archive. Packaging always reads the current shared source files;
it does not maintain or generate a second runtime tree in the working checkout.
Review source contents and distribution rights before publishing. Generate
deployment credentials on the target machine.
