#!/usr/bin/env bash
set -euo pipefail

LANE=${1:?usage: bash run.sh func|gui [check|smoke|train]}
MODE=${2:-train}
case "$LANE" in func|gui) ;; *) echo "lane must be func or gui" >&2; exit 2 ;; esac
case "$MODE" in check|smoke|train) ;; *) echo "mode must be check, smoke, or train" >&2; exit 2 ;; esac
: "${INIT_CKPT:?set INIT_CKPT to a Megatron checkpoint directory}"
: "${WORK_DIR:?set WORK_DIR to a writable output directory}"
[[ -d "$INIT_CKPT" ]] || { echo "INIT_CKPT does not exist: $INIT_CKPT" >&2; exit 3; }
INIT_CKPT=$(cd "$INIT_CKPT" && pwd)
mkdir -p "$WORK_DIR"
WORK_DIR=$(cd "$WORK_DIR" && pwd)
if [[ -n "${DATA_ROOT:-}" ]]; then
  [[ -d "$DATA_ROOT" ]] || { echo "DATA_ROOT does not exist: $DATA_ROOT" >&2; exit 3; }
  DATA_ROOT=$(cd "$DATA_ROOT" && pwd)
fi
if [[ -n "${IMAGE_ROOT:-}" ]]; then
  [[ -d "$IMAGE_ROOT" ]] || { echo "IMAGE_ROOT does not exist: $IMAGE_ROOT" >&2; exit 3; }
  IMAGE_ROOT=$(cd "$IMAGE_ROOT" && pwd)
fi
if [[ -n "${MODEL_DIR:-}" ]]; then
  [[ -d "$MODEL_DIR" ]] || { echo "MODEL_DIR does not exist: $MODEL_DIR" >&2; exit 3; }
  MODEL_DIR=$(cd "$MODEL_DIR" && pwd)
fi
export INIT_CKPT WORK_DIR DATA_ROOT IMAGE_ROOT MODEL_DIR

VERSION_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_DIR=${RUNTIME_DIR:-$VERSION_DIR/.runtime}
mkdir -p "$RUNTIME_DIR"
RUNTIME_DIR=$(cd "$RUNTIME_DIR" && pwd)
export RUNTIME_DIR
bash "$VERSION_DIR/setup.sh"

export BRIDGE_DIR="$RUNTIME_DIR/Megatron-Bridge"
export PYTHON="$RUNTIME_DIR/venv/bin/python"
export PYTHONPATH="$VERSION_DIR:$BRIDGE_DIR/src:$BRIDGE_DIR/3rdparty/Megatron-LM"
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$RUNTIME_DIR/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR=${TMPDIR:-/tmp}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/cua_torchinductor_${USER:-user}}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/cua_triton_${USER:-user}}
export PYTHONDONTWRITEBYTECODE=1
bash "$VERSION_DIR/train.sh" "$LANE" "$MODE"
