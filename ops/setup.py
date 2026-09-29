"""Build the CAST custom CUDA operators."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


OPS_DIR = Path(__file__).resolve().parent

cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]

setup(
    name="cast-cuda-ops",
    version="0.1.0",
    packages=["ops"],
    package_dir={"ops": "ops"},
    ext_modules=[
        CUDAExtension(
            name="ops.cast_fused_depth_geometry",
            sources=[
                str(OPS_DIR / "fused_depth_geometry_bindings.cpp"),
                str(OPS_DIR / "fused_depth_geometry_kernels.cu"),
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
