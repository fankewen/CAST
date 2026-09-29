#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Select one checkpoint; its configuration has the same filename stem.
# model=CAST_B_dinov2
# model=CAST_B_dinov2_no_grace
# model=CAST_B_dinov3
# model=CAST_B_dinov3_no_grace
model=CAST_L_dinov3

# Cache produced by preprocess/mixamo/build_cache.sh; contains pairs.jsonl.
mixamo_cache="/path/to/mixamo-work/cache"
checkpoint_dir="$repo_root/checkpoints"
results_dir="$repo_root/results/eval/cross_skeleton"

# Evaluate all directed source-video to target-skeleton Mixamo pairs.
python -m inference.video2motion_cross_skeleton \
  --cache-root "$mixamo_cache" \
  --config "configs/experiment/${model}.yaml" \
  --checkpoint "${checkpoint_dir}/${model}.pt" \
  --output-dir "${results_dir}/${model}" \
  --stride 20
