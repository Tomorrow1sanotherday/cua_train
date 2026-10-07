# Atomic (3x-full)

This is the 2026-09-07 one-epoch SFT recipe. Function and GUI are separate
53,917/53,918-trajectory datasets. The release keeps the original JSONL,
Energon byte-offset indexes, and image-relative paths. The image root is the
downloaded `atomic/` directory.

## Starting checkpoint

`INIT_CKPT` must point to a Megatron-format checkpoint imported from the
untrained Qwen3.5-VL-4B base model. The included
[`import_qwen35_vlm_checkpoint.py`](../shared/import_qwen35_vlm_checkpoint.py)
can produce it from a local HF-format model directory; run that import with
one GPU before using `train.sh`. `MODEL_DIR` points to the same model's local
HF configuration, tokenizer, and processor files.

## Train

```bash
export BRIDGE_DIR=/path/to/Megatron-Bridge
export MODEL_DIR=/path/to/Qwen3.5-4B
export INIT_CKPT=/path/to/base-megatron-checkpoint
export WORK_DIR=/path/to/cua-output
bash atomic/train.sh func
bash atomic/train.sh gui
```

The launcher downloads the `atomic/` data from `Furunhao/cua` into
`${DATA_CACHE:-$WORK_DIR/data}`, validates both JSONL SHA256 values, then runs
422 optimizer updates on eight GPUs. Global batch is 128, context parallelism
is 2 for Function and 4 for GUI, sequence length is 32,768, image processing
uses 200,704 pixels, and checkpoints are saved at iterations 211 and 422.

Use separate `WORK_DIR` outputs for repeated runs. GUI and Function each
start from the base checkpoint and must not initialize from each other.
