#!/usr/bin/env python3
"""Run the one-epoch Qwen3.5 CUA recipe with BF16 gradient reduction."""

from __future__ import annotations

import argparse
import os

if os.environ.get("CUA_DISABLE_JIT_FUSER_EARLY") == "1":
    from megatron.core.jit import disable_jit_fuser
    disable_jit_fuser()

from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step
from megatron.bridge.recipes.qwen_vl.qwen3_vl import _make_energon_dataset
from megatron.bridge.recipes.qwen_vl.qwen35_vl import qwen35_vl_4b_sft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

from cua_energon_adapter import CUAQwenVLTaskEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    args, overrides = parser.parse_known_args()

    config = qwen35_vl_4b_sft_config(hf_path=args.hf_path)
    config.dataset = _make_energon_dataset(
        hf_path=args.hf_path,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
    )
    native_encoder = config.dataset.task_encoder
    config.dataset.task_encoder = CUAQwenVLTaskEncoder(
        tokenizer=native_encoder.hf_tokenizer,
        image_processor=native_encoder.image_processor,
        max_padding_length=args.seq_length,
        min_pixels=native_encoder.min_pixels,
        max_pixels=native_encoder.max_pixels,
        max_num_images=native_encoder.max_num_images,
        max_num_frames=native_encoder.max_num_frames,
        max_visual_tokens=native_encoder.max_visual_tokens,
        image_root=os.environ.get(
            "CUA_IMAGE_ROOT",
            "/lumos-vePFS/data_pipeline/songweishuai/cua_train_and_benchmark/train/images",
        ),
    )
    config = process_config_with_overrides(config, cli_overrides=overrides or None)

    # The stock Qwen BF16 profile forces FP32 gradient reduction during finalize.
    # Materialize the native profile here so the requested BF16 reduction survives.
    config.mixed_precision = get_mixed_precision_config(config.mixed_precision)
    config.mixed_precision.grad_reduce_in_fp32 = False
    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
