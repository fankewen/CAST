# CAST Remote Inference

| Directory | Runs on | Documentation |
| --- | --- | --- |
| `server/` | Linux server with an NVIDIA GPU | [Server setup](server/README.md), [deployment](server/docs/DEPLOYMENT.md), [model weights](server/docs/MODELS.md) |
| `client/` | Computer running Blender and the camera | [Client installation and usage](client/README.md) |

## Getting Started

1. Follow the [server guide](server/README.md) to install dependencies, prepare weights, and start the API. Obtain the API URL and `CAST_API_KEY`.
2. Follow the [client guide](client/README.md) to install the external Python dependencies, build and install `cast_remote_live.zip`, and configure the connection.
3. In Blender's **CAST Remote** panel, import an FBX and wait for skeleton initialization. Start the camera, then start transfer.

See the [HTTP API](server/docs/API.md), [server validation guide](server/docs/VALIDATION.md), and [client validation checklist](client/README.md#validation) for protocol details and verification steps.
