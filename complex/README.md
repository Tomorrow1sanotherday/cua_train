# Complex (Slides01 action-window v4)

This is the 2026-10-03 Slides01 SFT recipe. The current training JSONL comes
from `action_window_padded_initial_plus5_gt_extra_v4_20260930`, previously
exposed as `action_window_current`. Each source trajectory is expanded into
one sample per target action. Only the final action and EOS receive loss.

The release contains Function 47,163 real actions plus 5 zero-loss padding
samples, and GUI 206,406 real actions plus 26 padding samples. Each sample
keeps the initial screenshot, up to five recent screenshots, and optional GT,
for a maximum of seven images. The image root is the downloaded
`complex/sft_gui_sharegpt/` directory.

## Starting checkpoint

Function must start from the **Function** atomic `iter_0000422`; GUI must
start from the **GUI** atomic `iter_0000422`. Set `INIT_CKPT` to the matching
Megatron-format checkpoint. The original run imported each route's atomic
HF export into Megatron format, then reset optimizer, scheduler, RNG, and
training step. This release contains training data and code; it does not
include model weights.

The downloader, CUA adapter, and optional checkpoint importer are included
here. This directory does not import code from `atomic/` or the repository
root. To prepare `INIT_CKPT` from an atomic HF export, run this directory's
`import_qwen35_vlm_checkpoint.py` in the compatible Megatron-Bridge environment.

## Train

```bash
export BRIDGE_DIR=/path/to/Megatron-Bridge
export MODEL_DIR=/path/to/Qwen3.5-4B
export INIT_CKPT=/path/to/atomic-func-iter_0000422
export WORK_DIR=/path/to/cua-output
bash complex/train.sh func

export INIT_CKPT=/path/to/atomic-gui-iter_0000422
bash complex/train.sh gui
```

The launcher downloads `complex/` from `Furunhao/cua`, checks the conversion
manifest and both JSONL SHA256 values, then runs three epochs on eight GPUs.
Function uses global batch 64 and 737 updates/epoch; GUI uses global batch
32 and 6,451 updates/epoch. Both use CP4/DP2, 32,768 tokens, up to seven
1920x1080 source images, and one full checkpoint per epoch.

Train Probe rows remain in the training set. There is no validation split.

To use only this directory, copy `complex/` to a Linux training machine,
install `requirements.txt` in a compatible Megatron-Bridge environment,
provide the matching atomic final checkpoint as `INIT_CKPT`, and run
`bash train.sh func` or `bash train.sh gui` from inside the directory.
