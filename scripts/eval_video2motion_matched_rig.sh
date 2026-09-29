#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

checkpoint_dir="$repo_root/checkpoints"
results_dir="$repo_root/results/eval/matched_rig"

# Select one checkpoint; its configuration has the same filename stem.
# model=CAST_B_dinov2
# model=CAST_B_dinov2_no_grace
# model=CAST_B_dinov3
# model=CAST_B_dinov3_no_grace
model=CAST_L_dinov3

# Evaluate the complete Zoo and Mobjaverse validation animations.
python -m inference.video2motion_matched_rig \
  --config "configs/experiment/${model}.yaml" \
  --checkpoint "${checkpoint_dir}/${model}.pt" \
  --output-dir "${results_dir}/${model}" \
  --stride 81 \
  --validation-source zoo \
  --validation-source mobj_seen \
  --validation-source mobj_unseen \
  # --visualize # Uncomment to write one RGB and projected-skeleton video per animation.
