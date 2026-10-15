#!/usr/bin/env bash
set -euo pipefail
ulimit -n "${NOFILE_LIMIT:-1048576}"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PATH="$(dirname "$PYTHON_BIN"):${PATH}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CONFIG="${CONFIG:-$repo_root/configs/experiment/CAST_B_dinov3.yaml}"
MASTER_PORT="${MASTER_PORT:-29647}"
NPROC="$(awk -F',' '{print NF}' <<< "${GPUS// /}")"
LOG_DIR="${LOG_DIR:-$repo_root/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log}"

TRAIN_ARGS=(--config "${CONFIG}")
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="${GPUS// /}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  echo "Config: ${CONFIG}"
  echo "GPUs: ${GPUS}"
  echo "NPROC: ${NPROC}"
  echo "Python: ${PYTHON_BIN}"
  echo "Log: ${LOG_FILE}"
  "${PYTHON_BIN}" -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    -m train.video2motion \
    "${TRAIN_ARGS[@]}"
} 2>&1 | tee -a "${LOG_FILE}"
