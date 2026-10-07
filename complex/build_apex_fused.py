#!/usr/bin/env python3
"""Build the Apex fused weight-gradient extension against this environment."""

from __future__ import annotations

import os
import shutil
import sysconfig
from pathlib import Path

from torch.utils.cpp_extension import load


apex = Path(os.environ["APEX_SOURCE"])
build = Path(os.environ["CUA_APEX_BUILD_DIR"])
build.mkdir(parents=True, exist_ok=True)
module = load(
    name="fused_weight_gradient_mlp_cuda",
    sources=[
        str(apex / "csrc/megatron/fused_weight_gradient_dense.cpp"),
        str(apex / "csrc/megatron/fused_weight_gradient_dense_cuda.cu"),
        str(apex / "csrc/megatron/fused_weight_gradient_dense_16bit_prec_cuda.cu"),
    ],
    extra_include_paths=[str(apex / "csrc")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    build_directory=str(build),
    verbose=True,
)
target = Path(sysconfig.get_paths()["purelib"]) / Path(module.__file__).name
shutil.copy2(module.__file__, target)
print(target)
