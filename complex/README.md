# Complex: Slides01 action-window v4

This directory contains its own setup, launcher, data adapter, downloaders,
action-window encoder/provider, training entry, and tests. It does not import
code from `atomic/` or the repository root.

## Inputs and commands

- `INIT_CKPT`: required **matching lane** atomic `iter_0000422` in Megatron
  format. Function must use atomic Function; GUI must use atomic GUI. Model
  parameters continue, while optimizer, scheduler, RNG, and step reset.
- `WORK_DIR`: required writable directory for downloads and outputs.
- `DATA_ROOT`: optional directory with `action_window_current/` JSONL,
  `.idx`, `conversion_manifest.json`, and `sft_gui_sharegpt/` images. Without
  it, download the public complex release from `Furunhao/cua`.
- `MODEL_DIR`: optional HF-format Qwen3.5-4B model files. Without it, download
  public `Qwen/Qwen3.5-4B` for configuration and processor files.

```bash
export INIT_CKPT=/path/to/atomic-func-iter_0000422
export WORK_DIR=/path/to/output
export DATA_ROOT=/path/to/complex-data  # optional
bash run.sh func check
bash run.sh func smoke
bash run.sh func train
```

If the image tree is separate, set `IMAGE_ROOT` to the directory containing
`images_cursor_crosspage/`. The HF release already has the expected layout.

There are 47,163 Function and 206,406 GUI real actions per epoch, plus 5/26
zero-loss padding rows. Only the target action and EOS participate in loss.
Both lanes train for three epochs at sequence length 32,768, CP4/DP2, with
global batch 64/32 and 737/6,451 updates per epoch. One full checkpoint is
saved at each epoch boundary. `smoke` runs one update and saves nothing.

The host requires an NVIDIA driver, CUDA toolkit with `nvcc`, eight GPUs,
Git, compiler tools, and Python 3 with `venv` and pip. `run.sh` calls `setup.sh` to
install the runtime in `.runtime/`, then trains directly on the host. It does
not require Docker.
