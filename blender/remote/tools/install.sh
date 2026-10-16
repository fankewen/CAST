#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?Set CUDA_HOME to an existing compatible CUDA 12.8 toolkit}"
"$CUDA_HOME/bin/nvcc" --version
P="${CAST_PYTHON:-python}"
"$P" -c 'import sys; assert sys.version_info[:2] == (3,10), "Tested target is Python 3.10"'
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/var/cache/pip}"
export TMPDIR="${TMPDIR:-$PWD/var/tmp}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
"$P" -m pip install 'pip>=24,<27' setuptools==80.9.0 wheel packaging==25.0 ninja==1.13.0
"$P" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$P" -m pip install -c constraints.txt -r requirements.txt
"$P" -m pip install xformers==0.0.30 --no-deps --index-url https://download.pytorch.org/whl/cu128
# Local source build avoids upstream wheel glibc/C++ ABI mismatches.
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$($P -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')}"
"$P" -m pip install flash-attn==2.8.3 --no-deps --no-build-isolation
[[ -d third_party/sam2/sam2 ]] || { echo 'Run tools/fetch_dependencies.py first' >&2; exit 1; }
SAM2_BUILD_ALLOW_ERRORS=0 "$P" -m pip install -e third_party/sam2 --no-build-isolation --no-deps
P_ABS="$("$P" -c 'import sys; print(sys.executable)')"
(cd ../.. && "$P_ABS" ops/setup.py build_ext --inplace)
"$P" tools/check_acceleration.py
"$P" -m pip check
