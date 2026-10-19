#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Select one model configuration.
# model=CAST_B_dinov2
# model=CAST_B_dinov2_no_grace
model=CAST_B_dinov3
# model=CAST_B_dinov3_no_grace
# model=CAST_L_dinov3
# model=CAST_B_dinov3_causal
# model=CAST_L_dinov3_causal
# model=CAST_B_dinov3_mask_completion

gpus="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="$gpus"
IFS=',' read -r -a gpu_ids <<< "$gpus"

python -u -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${#gpu_ids[@]}" \
  -m train.video2motion \
  --config "configs/experiment/${model}.yaml"
