# Security and Privacy

## Access and Configuration

- Keep the default loopback addresses when connecting through an SSH tunnel.
- Run `python tools/init_env.py` to create `.env`, then read the generated
  API key and administrator password locally. Preserve an existing `.env`.
- Send the API key in `X-API-Key`. Sign in to the management console with
  `CAST_ADMIN_USER` and `CAST_ADMIN_PASSWORD` from `.env`.
- Public deployments require TLS, access controls, connection and request
  limits, and dependency updates. When upgrading the pinned Uvicorn/H11 stack,
  verify the [HTTP connection admission behavior](API.md#http-connection-admission)
  on an isolated instance before restarting the production service.

Set `CAST_ADMIN_SECURE_COOKIE=1` behind an HTTPS reverse proxy. Keep credentials
out of URLs, logs, and public issues. Restrict access to asset and state
directories; `umask 077` is suitable for the startup shell. Use trusted
configurations, weights, and active manifests.

## Image History

`CAST_HISTORY_ENABLED=0` disables image capture by default. Before enabling it,
confirm consent and retention requirements for the input images.

History samples subsequent input at most once per second by default, with
24-hour, 3,600-record, and 2 GiB retention limits. Stored data may include original
images, person boxes, SAM masks, crops, model inputs, poses, and rig labels.
Frames without a detected person may have no crop.

Image access requires administrator login. Disabling capture stops new samples;
it does not erase existing data. The operator manages backups, access auditing,
and cleanup. Keep `var/` out of source distributions.

Resource monitoring samples every five seconds and retains metrics for seven
days. Use the benchmark tool to measure valid motion frame rate.

## Logs

Origin audit records contain sequence numbers, epochs, connection IDs, times,
status, and peer or forwarding diagnostics. Image bodies and API keys should
remain out of logs. The console masks known credentials and common token
fields; review diagnostic exports before sharing them.

Audit logs rotate automatically. Configure rotation and retention for process
manager stdout logs. Share only the necessary redacted excerpts.

## Repository Deployment

Clone and update the complete CAST repository on the server. Keep credentials,
model weights, input data, and runtime state outside Git. Generate `.env` on
the target machine with `tools/init_env.py`; do not copy another instance's
credentials or logs into a commit.
