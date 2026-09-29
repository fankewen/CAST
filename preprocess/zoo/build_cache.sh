#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "Usage: bash build_cache.sh TRUEBONE_ZOO_ROOT BLENDER WORK_DIR" >&2
  exit 2
fi

zoo_root=$1
blender=$2
work=$3

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
scripts="$here/scripts"
third_party="$here/third_party"
scene="$third_party/preprocess/blank.blend"
python=${PYTHON:-python3}

# Convert character and animation FBXs into source skeletons and motions.
"$python" "$scripts/prepare_zoo_fbx.py" \
  --zoo-root "$zoo_root" \
  --blender "$blender" \
  --output-root "$work/source"

# Render an RGB video for each source animation.
"$python" "$scripts/render_rgb_videos.py" \
  --zoo-root "$work/source" \
  --blender "$blender" \
  --scene "$scene" \
  --workers "${ZOO_RGB_WORKERS:-4}"

# Extract the animated mesh vertices needed for mask rendering.
"$python" "$scripts/extract_mesh_vertices.py" \
  --source-root "$work/source" \
  --third-party-root "$third_party" \
  --output-root "$work/vertices" \
  --workers "${ZOO_VERTEX_WORKERS:-4}"

# Render foreground masks aligned with the RGB views.
"$python" "$scripts/render_masks_multigpu.py" \
  --vertices-root "$work/vertices" \
  --output-root "$work/masks" \
  --blender "$blender" \
  --scene "$scene" \
  --workers "${ZOO_MASK_WORKERS:-4}" --threads 4

# Package RGB frames, masks, cameras, and motion for each view.
"$python" "$scripts/build_view_cache.py" \
  --source-root "$work/source" \
  --mask-root "$work/masks" \
  --vertices-root "$work/vertices" \
  --third-party-root "$third_party" \
  --output-root "$work/cache-views" \
  --workers "${ZOO_CACHE_WORKERS:-8}"

# Combine the views of each animation into one cache entry.
"$python" "$scripts/merge_multiview_cache.py" \
  --source-root "$work/cache-views" \
  --output-root "$work/cache-merged"

# Group matching skeletons into the final cache.
"$python" "$scripts/deduplicate_skeleton_caches.py" \
  --source-root "$work/cache-merged" \
  --output-root "$work/cache"
