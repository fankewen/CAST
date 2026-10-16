# Build external code first (tools/fetch_dependencies.py); weights are mounted.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
WORKDIR /app
ENV CUDA_HOME=/usr/local/cuda PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ ninja-build iproute2 && rm -rf /var/lib/apt/lists/*
COPY blender/remote/requirements.txt blender/remote/constraints.txt ./
RUN python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 \
 && python -m pip install -c constraints.txt -r requirements.txt \
 && python -m pip install xformers==0.0.30 --no-deps --index-url https://download.pytorch.org/whl/cu128
ARG CUDA_ARCH=9.0
ARG FLASH_ARCH=90
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} FLASH_ATTN_CUDA_ARCHS=${FLASH_ARCH} FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 NVCC_THREADS=2
RUN python -m pip install flash-attn==2.8.3 --no-deps --no-build-isolation
COPY models/ models/
COPY data/ data/
COPY inference/ inference/
COPY utils/ utils/
COPY visualization/ visualization/
COPY ops/ ops/
COPY LICENSE LICENSE-3RD-PARTY ./
COPY blender/cast_engine_server.py blender/rig_adapter.py blender/
COPY blender/remote/ blender/remote/
WORKDIR /app/blender/remote
RUN test -d third_party/sam2/sam2 && test -d third_party/dinov3/dinov3 \
 && SAM2_BUILD_ALLOW_ERRORS=0 python -m pip install -e third_party/sam2 --no-deps --no-build-isolation \
 && cd /app && python ops/setup.py build_ext --inplace
ENV CAST_HOST=0.0.0.0 CAST_STATE_DIR=/state CAST_ASSETS_DIR=/assets CAST_ADMIN_STATE_DIR=/state/admin/local
EXPOSE 8080 8081
CMD ["python", "-u", "run_origin.py"]
