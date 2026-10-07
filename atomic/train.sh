#!/usr/bin/env bash
set -euo pipefail

LANE=${1:?usage: bash train.sh func|gui [check|smoke|train]}
MODE=${2:-train}
case "$MODE" in check|smoke|train) ;; *) echo "mode must be check, smoke, or train" >&2; exit 2 ;; esac
case "$LANE" in
  func) CP=2; PORT=29771; FUSER=0; RECOVERY_INTERVAL=null; RECOVERY_TYPE=null; RECOVERY_DIR=null ;;
  gui) CP=4; PORT=29772; FUSER=1; RECOVERY_INTERVAL=25; RECOVERY_TYPE=global ;;
  *) echo "lane must be func or gui" >&2; exit 2 ;;
esac

: "${BRIDGE_DIR:?set BRIDGE_DIR to Megatron-Bridge}"
: "${INIT_CKPT:?set INIT_CKPT to base Megatron checkpoint}"
: "${WORK_DIR:?set WORK_DIR to output root}"
PYTHON=${PYTHON:-python}
VERSION_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_CACHE=${DATA_CACHE:-$WORK_DIR/data}
DATA_REVISION=${DATA_REVISION:-main}
MODEL_REVISION=${MODEL_REVISION:-main}
MODEL_DIR=${MODEL_DIR:-$WORK_DIR/models/Qwen3.5-4B}
if [[ -n "${DATA_ROOT:-}" ]]; then LOCAL_DATA=1; else LOCAL_DATA=0; fi
DATA_ROOT=${DATA_ROOT:-$DATA_CACHE/dataset/cua-training-data/atomic}
IMAGE_ROOT=${IMAGE_ROOT:-$DATA_ROOT}
SAVE=$WORK_DIR/atomic/$LANE/checkpoints
TB=$WORK_DIR/atomic/$LANE/tensorboard
TRAIN_ITERS=422
DECAY_ITERS=422
SAVE_INTERVAL=211
if [[ "$LANE" == gui ]]; then
  RECOVERY_DIR=$SAVE/non_persistent_recovery
fi
if [[ "$MODE" == smoke ]]; then
  TRAIN_ITERS=1
  DECAY_ITERS=1
  SAVE=null
  SAVE_INTERVAL=0
  RECOVERY_INTERVAL=null
  RECOVERY_TYPE=null
  RECOVERY_DIR=null
fi

[[ -d "$BRIDGE_DIR" && -d "$INIT_CKPT" ]] || {
  echo "Bridge or initialization checkpoint directory is missing" >&2; exit 3;
}
mkdir -p "$TB"
if [[ "$MODE" == train ]]; then mkdir -p "$SAVE"; fi
if [[ "$LOCAL_DATA" == 1 ]]; then
  "$PYTHON" "$VERSION_DIR/download_data.py" atomic \
    --local-root "$DATA_ROOT" --image-root "$IMAGE_ROOT"
else
  "$PYTHON" "$VERSION_DIR/download_data.py" atomic \
    --dest "$DATA_CACHE" --revision "$DATA_REVISION"
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  "$PYTHON" "$VERSION_DIR/download_model.py" \
    --dest "$MODEL_DIR" --revision "$MODEL_REVISION"
fi

export CUA_IMAGE_ROOT="$IMAGE_ROOT"
export CUA_DISABLE_JIT_FUSER_EARLY="$FUSER"
export PYTHONPATH="$VERSION_DIR:$BRIDGE_DIR/src:$BRIDGE_DIR/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
"$PYTHON" "$VERSION_DIR/runtime_check.py" \
  --data-root "$DATA_ROOT" --image-root "$IMAGE_ROOT" --model-dir "$MODEL_DIR" \
  --checkpoint "$INIT_CKPT" --lane "$LANE"
if [[ "$MODE" == check ]]; then exit 0; fi

cd "$BRIDGE_DIR"
"$PYTHON" -m torch.distributed.run --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port="$PORT" \
  "$VERSION_DIR/run_qwen35_cua_1ep.py" \
  --hf-path "$MODEL_DIR" --seq-length 32768 \
  --micro-batch-size 1 --global-batch-size 128 \
  "dataset.path=$DATA_ROOT/train.3x.$LANE.jsonl" \
  dataset.seq_length=32768 dataset.num_workers=2 \
  dataset.max_num_images=96 dataset.max_visual_tokens=24576 \
  dataset.min_pixels=200704 dataset.max_pixels=200704 \
  dataset.max_num_frames=60 dataset.pack_sequences_in_batch=false \
  "train.train_iters=$TRAIN_ITERS" train.global_batch_size=128 train.micro_batch_size=1 \
  validation.eval_iters=0 optimizer.lr=1e-5 optimizer.min_lr=1e-6 \
  optimizer.use_precision_aware_optimizer=true optimizer.main_grads_dtype=bfloat16 \
  optimizer.optimizer_cpu_offload=false scheduler.lr_warmup_iters=0 \
  "scheduler.lr_decay_iters=$DECAY_ITERS" scheduler.lr_decay_style=cosine \
  "checkpoint.pretrained_checkpoint=$INIT_CKPT" checkpoint.load=null \
  "checkpoint.save=$SAVE" "checkpoint.save_interval=$SAVE_INTERVAL" \
  "checkpoint.non_persistent_save_interval=$RECOVERY_INTERVAL" \
  "checkpoint.non_persistent_ckpt_type=$RECOVERY_TYPE" \
  "checkpoint.non_persistent_global_ckpt_dir=$RECOVERY_DIR" \
  checkpoint.async_save=false logger.log_interval=1 \
  "logger.tensorboard_dir=$TB" model.seq_length=32768 \
  model.tensor_model_parallel_size=1 model.pipeline_model_parallel_size=1 \
  "model.context_parallel_size=$CP" model.sequence_parallel=false \
  model.recompute_granularity=full model.recompute_method=uniform \
  model.recompute_num_layers=1 model.attention_backend=fused \
  model.calculate_per_token_loss=true model.cross_entropy_loss_fusion=false \
  ddp.average_in_collective=false ddp.grad_reduce_in_fp32=false \
  dist.disable_jit_fuser=true
