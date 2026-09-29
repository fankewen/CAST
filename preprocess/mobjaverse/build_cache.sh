#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: bash build_cache.sh MOBJAVERSE_ROOT WORK_DIR" >&2
  exit 2
fi

source_root=$1
work=$2
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
scripts="$here/scripts"
python=${PYTHON:-python3}

if [[ ! -f "$source_root/render.py" || ! -d "$source_root/src" || ! -d "$source_root/datalist" || ! -d "$source_root/mobjaverse" ]]; then
  echo "MOBJAVERSE_ROOT must contain render.py, src/, datalist/, and mobjaverse/." >&2
  exit 2
fi

if [[ -e "$work/cache" ]]; then
  echo "Cache output already exists: $work/cache. Choose a new work directory." >&2
  exit 2
fi

mkdir -p -- "$work"

# Render every frame of every listed asset as RGB and mask images.
"$python" "$scripts/render_official_full_frames.py" \
  --source-root "$source_root" \
  --output-root "$work/render" \
  --egl-device-id "${MOBJ_EGL_DEVICE_ID:-0}"

# Group assets by skeleton and build the final cache.
"$python" "$scripts/build_fullframe_cache.py" \
  --source-root "$source_root" \
  --render-root "$work/render" \
  --output-root "$work/cache" \
  --workers "${MOBJ_CACHE_WORKERS:-16}"
