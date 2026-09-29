#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "Usage: bash build_cache.sh MIXAMO_FBX_ROOT BLENDER WORK_DIR" >&2
  exit 2
fi

raw_root=$1
blender=$2
work=$3
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}

# Export the six rest skeletons, motion arrays, and original reference poses.
# Render all FBX frames and package RGB/mask frames with directed pairs.
"$python" "$here/scripts/run_render.py" \
  --raw-root "$raw_root" \
  --blender "$blender" \
  --cache "$work/cache" \
  --log-dir "$work/logs" \
  --workers "${MIXAMO_RENDER_WORKERS:-3}" \
  --threads "${MIXAMO_BLENDER_THREADS:-4}"
