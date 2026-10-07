#!/usr/bin/env python3
"""Fail before torchrun when the runtime or user inputs are incomplete."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lane", choices=("func", "gui"), required=True)
    args = parser.parse_args()
    importlib.import_module("run_qwen35_cua_1ep")
    importlib.import_module("transformer_engine.pytorch")
    from transformers import AutoTokenizer, Qwen3VLProcessor
    if torch.cuda.device_count() != 8:
        raise RuntimeError("the recorded run requires exactly eight visible GPUs")
    data = args.data_root / f"train.3x.{args.lane}.jsonl"
    if not data.is_file():
        raise FileNotFoundError("atomic training JSONL is missing")
    with data.open(encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    image = Path(record["images"][0])
    image = image if image.is_absolute() else args.image_root / image
    with Image.open(image) as picture:
        picture.verify()
    if not args.checkpoint.is_dir() or not any(args.checkpoint.iterdir()):
        raise FileNotFoundError("Megatron initialization checkpoint is missing or empty")
    AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    Qwen3VLProcessor.from_pretrained(args.model_dir, local_files_only=True)
    print(f"RUNTIME_OK lane={args.lane} torch={torch.__version__} gpus=8")


if __name__ == "__main__":
    main()
