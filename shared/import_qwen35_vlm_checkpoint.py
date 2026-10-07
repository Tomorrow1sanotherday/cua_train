#!/usr/bin/env python3
"""Import the local Qwen3.5 VLM while disabling the unavailable Apex fusion."""

from __future__ import annotations

import argparse

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.utils.common_utils import print_rank_0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--megatron-path", required=True)
    return parser.parse_args()


@torchrun_main
def import_checkpoint(hf_model: str, megatron_path: str) -> None:
    print_rank_0(f"Importing Qwen3.5 VLM: {hf_model} -> {megatron_path}")
    bridge = AutoBridge.from_hf_pretrained(hf_model, torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.pipeline_dtype = torch.bfloat16
    provider.params_dtype = torch.bfloat16
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    provider.initialize_model_parallel(seed=1234)

    model = provider.provide_distributed_model(wrap_with_ddp=False)
    tokenizer_kwargs = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs() or {}
    bridge.save_megatron_model(
        model,
        megatron_path,
        hf_tokenizer_path=hf_model,
        hf_tokenizer_kwargs=tokenizer_kwargs,
    )
    print_rank_0("QWEN35_VLM_CHECKPOINT_IMPORT_SUCCESS")


def main() -> None:
    args = parse_args()
    import_checkpoint(args.hf_model, args.megatron_path)


if __name__ == "__main__":
    main()
