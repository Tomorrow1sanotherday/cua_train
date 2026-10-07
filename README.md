# CUA Training

Two Qwen3.5-VL-4B SFT recipes are preserved here:

| Version | Data | Supervision | Starting weights |
| --- | --- | --- | --- |
| [`atomic/`](atomic/README.md) | 3x-full, 53,917 Function or 53,918 GUI trajectories | Original full assistant turns | Qwen3.5-VL-4B base |
| [`complex/`](complex/README.md) | Slides01 v4, 47,163 Function or 206,406 GUI actions per epoch | Final action and EOS only | Matching atomic `iter_0000422` |

Data lives in the public Hugging Face dataset `Furunhao/cua` under
`dataset/cua-training-data/{atomic,complex}`. Each launcher downloads its own
data release, verifies the training JSONL SHA256, and points the image loader
at the downloaded image tree. The original image paths inside JSONL are kept.

## Environment

- Eight NVIDIA A100 80 GB or equivalent GPUs for the recorded configuration.
- NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:25.04-py3` or a compatible
  environment with PyTorch, CUDA, Megatron-Bridge, Megatron-LM, Energon,
  Transformers, Pillow, and `huggingface_hub`.
- The recorded Bridge revision was `5cb3444c43f7499cf3872b2d46870cf8bc2e00ce`.
  The run also used local CUA adapter code included in this repository.
- Local Qwen3.5-VL-4B model files and a Megatron-format initialization
  checkpoint. See each version's README for the required starting point.
- Enough local disk for the downloaded data, model, and checkpoints. Dataset
  files are downloaded before training; workers do not stream images from HF.

Install `huggingface_hub>=1.24,<3` in the training environment, then set
`BRIDGE_DIR`, `MODEL_DIR`, `INIT_CKPT`, and `WORK_DIR`. The launchers use the
same Function/GUI configuration values as the recorded runs. They do not
automatically upload checkpoints.

## Data download

```bash
python shared/download_data.py atomic --dest /data/cua-hf --revision main
python shared/download_data.py complex --dest /data/cua-hf --revision main
```

The script resolves `main` to an immutable commit before downloading and
prints the resulting local data root. For a repeatable run, pass that commit
as `--revision` or set `DATA_REVISION` when launching training.

## Run

```bash
export BRIDGE_DIR=/path/to/Megatron-Bridge
export MODEL_DIR=/path/to/Qwen3.5-4B
export WORK_DIR=/path/to/cua-output
export INIT_CKPT=/path/to/base-megatron-checkpoint
bash atomic/train.sh func

export INIT_CKPT=/path/to/atomic-func-iter_0000422
bash complex/train.sh func
```

Run GUI separately with its own initialization checkpoint. The two lanes are
independent; GUI must not continue from the Function checkpoint. Both scripts
run a single eight-GPU node with `torch.distributed.run` and write to distinct
output directories.

The commands reproduce the data and model configuration of the recorded runs.
They require the stated hardware and dependencies; no GPU training was run as
part of preparing this release.
