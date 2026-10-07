# Atomic: 3x-full

This directory contains its own setup, launcher, data adapter, downloaders,
training entry, and tests. It does not need `complex/` or repository-root
Python code.

## Inputs and commands

- `INIT_CKPT`: required Megatron-format checkpoint imported from
  Qwen3.5-4B base. The optional `import_qwen35_vlm_checkpoint.py` converts
  local HF-format weights into Megatron format.
- `WORK_DIR`: required writable directory for downloads and outputs.
- `DATA_ROOT`: optional directory holding both `train.3x.{func,gui}.jsonl`,
  their `.jsonl.idx` files, and images. Without it, download from
  `Furunhao/cua` and verify the JSONL hashes.
- `MODEL_DIR`: optional HF-format Qwen3.5-4B directory. Without it, download
  public `Qwen/Qwen3.5-4B` for configuration and processor files.

```bash
export INIT_CKPT=/path/to/base-megatron-checkpoint
export WORK_DIR=/path/to/output
export DATA_ROOT=/path/to/atomic-data  # optional
bash run.sh func check
bash run.sh func smoke
bash run.sh func train
```

If JSONL and images live under separate roots, set `IMAGE_ROOT` to the
directory containing `images_cursor_*`. The HF release already puts them
under one atomic root.

Function and GUI train independently from base weights. Each uses 422
optimizer updates, global batch 128, sequence length 32,768, and save points
211 and 422. Function uses context parallelism 2; GUI uses 4 and an extra
rolling recovery checkpoint every 25 steps. `smoke` changes only the number
of updates and disables saving.

The host requires an NVIDIA driver, CUDA toolkit with `nvcc`, eight GPUs,
Git, compiler tools, and Python 3 with `venv` and pip. `run.sh` calls `setup.sh` to
install the remaining runtime in `.runtime/`, then runs `train.sh` directly
on the host. It does not require Docker.
