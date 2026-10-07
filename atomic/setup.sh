#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_DIR=${RUNTIME_DIR:-$VERSION_DIR/.runtime}
mkdir -p "$RUNTIME_DIR"
RUNTIME_DIR=$(cd "$RUNTIME_DIR" && pwd)
BRIDGE="$RUNTIME_DIR/Megatron-Bridge"
TE="$RUNTIME_DIR/TransformerEngine"
APEX="$RUNTIME_DIR/Apex"
VENV="$RUNTIME_DIR/venv"
BRIDGE_COMMIT=5cb3444c43f7499cf3872b2d46870cf8bc2e00ce
TE_COMMIT=b9d690e042b1c4e455214e7dab65d6d3512c05d6
APEX_COMMIT=575968bc1f9127ccd61003a681472a83af4ff1a1

for command_name in git g++ make python3; do
  command -v "$command_name" >/dev/null || {
    echo "Missing $command_name. Install Git, C++ build tools, Python 3, and a CUDA toolkit with nvcc." >&2
    exit 3
  }
done
if [[ -z "${CUDA_HOME:-}" ]]; then
  command -v nvcc >/dev/null || { echo "Set CUDA_HOME to a CUDA toolkit containing bin/nvcc" >&2; exit 3; }
  CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
fi
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "CUDA_HOME/bin/nvcc is missing" >&2; exit 3; }
export MAX_JOBS=${MAX_JOBS:-8}

UV_BIN=${UV_BIN:-$(command -v uv || true)}
if [[ -z "$UV_BIN" ]]; then
  python3 -m venv "$RUNTIME_DIR/bootstrap"
  "$RUNTIME_DIR/bootstrap/bin/python" -m pip install --disable-pip-version-check uv==0.12.16
  UV_BIN="$RUNTIME_DIR/bootstrap/bin/uv"
fi
[[ -x "$UV_BIN" ]] || { echo "uv 0.12.16 could not be installed" >&2; exit 3; }

fingerprint=$(sha256sum "$VERSION_DIR/setup.sh" "$VERSION_DIR/runtime-requirements.txt" \
  "$VERSION_DIR/bridge-training.patch" "$VERSION_DIR/build_apex_fused.py" | sha256sum | cut -c1-64)
if [[ -f "$RUNTIME_DIR/.ready" && -x "$VENV/bin/python" && $(cat "$RUNTIME_DIR/.ready") == "$fingerprint" ]]; then
  echo "RUNTIME_READY $RUNTIME_DIR"
  exit 0
fi

if [[ ! -d "$BRIDGE/.git" ]]; then
  git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "$BRIDGE"
  git -C "$BRIDGE" checkout "$BRIDGE_COMMIT"
  git -C "$BRIDGE" submodule update --init --recursive
fi
[[ $(git -C "$BRIDGE" rev-parse HEAD) == "$BRIDGE_COMMIT" ]] || {
  echo "Megatron-Bridge checkout differs from pinned commit" >&2; exit 3;
}
if git -C "$BRIDGE" apply --ignore-space-change --reverse --check "$VERSION_DIR/bridge-training.patch" 2>/dev/null; then
  :
else
  git -C "$BRIDGE" apply --ignore-space-change --check "$VERSION_DIR/bridge-training.patch"
  git -C "$BRIDGE" apply --ignore-space-change "$VERSION_DIR/bridge-training.patch"
fi
sed -i 's/\r$//' "$BRIDGE/src/megatron/bridge/data/energon/task_encoder_utils.py"

PYTHON312=${PYTHON312:-3.12}
if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV_BIN" venv --python "$PYTHON312" "$VENV"
fi
"$UV_BIN" pip install --python "$VENV/bin/python" torch==2.10.0
cuda_arch=$(
  "$VENV/bin/python" -c 'import torch; major, minor = torch.cuda.get_device_capability(0); print(f"{major}.{minor}")'
)
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$cuda_arch}
export NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-${cuda_arch/./}}
UV_EXTRA_INDEX_URL=https://pypi.nvidia.com "$UV_BIN" pip install \
  --python "$VENV/bin/python" -r "$VERSION_DIR/runtime-requirements.txt"
"$UV_BIN" pip install --python "$VENV/bin/python" setuptools wheel cmake ninja pybind11 \
  onnx==1.21.0 onnxscript==0.7.1 nvdlfw-inspect==0.2.2

if [[ -n "${TE_SOURCE:-}" ]]; then
  command -v rsync >/dev/null || { echo "rsync is required with TE_SOURCE" >&2; exit 3; }
  mkdir -p "$TE"
  rsync -a --exclude=/build/ --exclude='*.egg-info/' "$TE_SOURCE/" "$TE/"
elif [[ ! -d "$TE/.git" ]]; then
  git clone https://github.com/NVIDIA/TransformerEngine.git "$TE"
  git -C "$TE" checkout "$TE_COMMIT"
  git -C "$TE" submodule update --init --recursive
fi
if [[ -d "$TE/.git" ]]; then
  actual_te_commit=$(git -C "$TE" rev-parse HEAD)
else
  actual_te_commit=$(cat "$TE/.source_commit")
fi
[[ "$actual_te_commit" == "$TE_COMMIT" ]] || {
  echo "Transformer Engine checkout differs from pinned commit" >&2; exit 3;
}
"$UV_BIN" pip install --python "$VENV/bin/python" --no-build-isolation --no-deps "$TE"

if [[ ! -d "$APEX/.git" ]]; then
  git clone https://github.com/NVIDIA/apex.git "$APEX"
  git -C "$APEX" checkout "$APEX_COMMIT"
fi
[[ $(git -C "$APEX" rev-parse HEAD) == "$APEX_COMMIT" ]] || {
  echo "Apex checkout differs from pinned commit" >&2; exit 3;
}
export APEX_SOURCE="$APEX"
export CUA_APEX_BUILD_DIR="$RUNTIME_DIR/apex-build"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$VENV/bin/python" "$VERSION_DIR/build_apex_fused.py"

helper_suffix=$($VENV/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PATH="$VENV/bin:$PATH" make -C "$BRIDGE/3rdparty/Megatron-LM/megatron/core/datasets" "LIBEXT=$helper_suffix"
PYTHONPATH="$VERSION_DIR:$BRIDGE/src:$BRIDGE/3rdparty/Megatron-LM" \
  "$VENV/bin/python" -c "import torch, transformers, megatron.bridge, megatron.core, megatron.energon, transformer_engine, fused_weight_gradient_mlp_cuda; print(torch.__version__, transformers.__version__)"
printf '%s\n' "$fingerprint" > "$RUNTIME_DIR/.ready"
echo "RUNTIME_READY $RUNTIME_DIR"
