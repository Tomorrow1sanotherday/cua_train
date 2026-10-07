# CUA Training

Two independent training releases are provided. `atomic/` reproduces the
3x-full Function/GUI SFT; `complex/` trains the latest Slides01 action-window
Function/GUI SFT. Each directory can be copied and run by itself. Neither
requires Docker or imports code from the other directory.

The datasets are public in `Furunhao/cua` at
`dataset/cua-training-data/{atomic,complex}`. Model weights and Megatron
checkpoints are not included in this GitHub repository.

## New Linux server

The host needs eight compatible NVIDIA GPUs (the recorded runs used 80 GB
A100/A800 cards), a working NVIDIA driver, a CUDA development toolkit with
`nvcc`, Git, a C++ compiler, `make`, Python 3 with `venv` and pip, and enough free disk.
The scripts install their own Python 3.12 environment and training libraries.
They do not use a container. CUDA 12.8 is recommended to match PyTorch 2.10's
CUDA 12.8 build; other toolkit versions need local validation.

```bash
git clone https://github.com/Tomorrow1sanotherday/cua_train.git
cd cua_train
export INIT_CKPT=/path/to/base-megatron-checkpoint
export WORK_DIR=/path/to/output
export DATA_ROOT=/path/to/atomic-data  # optional; omit to download from HF
bash atomic/run.sh func check
bash atomic/run.sh func smoke
bash atomic/run.sh func train
```

For complex, set `INIT_CKPT` to the **matching atomic lane's** final
`iter_0000422`, optionally set `DATA_ROOT` to the local complex release, and
run `bash complex/run.sh func check|smoke|train`. Substitute `gui` for the
independent GUI route.

`setup.sh`, called by `run.sh`, creates `.runtime/`, downloads the pinned
Megatron-Bridge commit and its Megatron-LM submodule, applies the training
patch, installs PyTorch/Transformers/Energon, and builds Transformer Engine
and the needed Apex fused weight-gradient extension against that PyTorch.
This first compilation can take several minutes. A successful setup writes
`.runtime/.ready`; later runs reuse it.

`check` validates imports, data hashes, sample image decoding, model
processor, checkpoint directory, and GPU count without updating parameters.
`smoke` does one real optimizer update without saving a checkpoint. `train`
uses the recorded update count and save schedule. The launchers download
`Qwen/Qwen3.5-4B` when `MODEL_DIR` is absent; this provides config, tokenizer,
processor, and model files, but does **not** replace `INIT_CKPT`. For a
repeatable download, pin `DATA_REVISION` and `MODEL_REVISION` to HF commits.

The original dataset JSONL and images were verified on HF. The native-host
package still requires a completed smoke on a machine with a suitable CUDA
development toolkit before its end-to-end behavior can be called verified.
