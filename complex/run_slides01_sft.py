#!/usr/bin/env python3
"""Independent Slides01 entry for legacy trajectories or action windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

if os.environ.get("CUA_DISABLE_JIT_FUSER_EARLY") == "1":
    from megatron.core.jit import disable_jit_fuser

    disable_jit_fuser()

from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step
from megatron.bridge.recipes.qwen_vl.qwen3_vl import _make_energon_dataset
from megatron.bridge.recipes.qwen_vl.qwen35_vl import qwen35_vl_4b_sft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

from action_window_encoder import ActionWindowEncoder
from action_window_provider import ActionWindowDatasetProvider
from cua_energon_adapter import CUAQwenVLTaskEncoder


def verify_action_data(path: str, image_root: str, smoke_data: bool = False,
                       max_num_images: int = 10) -> dict:
    data = Path(path).resolve()
    expected_names = {"smoke.gui.jsonl", "smoke.func.jsonl"} if smoke_data else {
        "train.action_window.gui.jsonl", "train.action_window.func.jsonl",
    }
    if data.name not in expected_names:
        raise ValueError(f"wrong action_window data path: {data}")
    manifest_path = data.parent / ("smoke_manifest.json" if smoke_data else "conversion_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if smoke_data:
        if max_num_images == 10:
            valid = manifest.get("max_images", 10) == 10
        elif max_num_images == 5:
            valid = (manifest.get("format") == "slides01_action_window_smoke_v2" and
                     manifest.get("max_images") == 5 and
                     manifest.get("omission_style") == "source_filename")
        elif max_num_images == 6:
            valid = (manifest.get("format") == "slides01_action_window_smoke_v3" and
                     manifest.get("max_images") == 6 and
                     manifest.get("max_screenshots") == 5 and
                     manifest.get("include_initial_observation") is True and
                     manifest.get("gt_counts_toward_screenshot_limit") is False)
        else:
            valid = (manifest.get("format") == "slides01_action_window_smoke_v4" and
                     manifest.get("max_images") == 7 and
                     manifest.get("max_screenshots") == 5 and
                     manifest.get("include_initial_observation") is True and
                     manifest.get("initial_counts_toward_screenshot_limit") is False and
                     manifest.get("gt_counts_toward_screenshot_limit") is False)
    else:
        policy = (manifest.get("format"), manifest.get("max_images"),
                  manifest.get("omission_style", "generic"),
                  manifest.get("max_screenshots", manifest.get("max_images")),
                  manifest.get("include_initial_observation", False),
                  manifest.get("gt_counts_toward_screenshot_limit", True),
                  manifest.get("initial_counts_toward_screenshot_limit", True))
        expected = {
            10: ("slides01_action_window_padded_v1", 10, "generic", 10, False, True, True),
            5: ("slides01_action_window_padded_v2", 5, "source_filename", 5, False, True, True),
            6: ("slides01_action_window_padded_v3", 6, "source_filename", 5, True, False, True),
            7: ("slides01_action_window_padded_v4", 7, "source_filename", 5, True, False, False),
        }
        valid = policy == expected.get(max_num_images)
    if not valid:
        raise ValueError(f"wrong conversion manifest or image limit: {manifest_path}")
    lane = data.name.split(".")[1 if smoke_data else 2]
    details = manifest[lane] if smoke_data else manifest["lanes"][lane]
    if not smoke_data and data.stat().st_size != details["output_bytes"]:
        raise ValueError(f"data size mismatch: {data}")
    if not data.with_suffix(".jsonl.idx").exists():
        raise ValueError(f"data size or Energon index mismatch: {data}")
    if not Path(image_root).is_dir():
        raise ValueError(f"image root missing: {image_root}")
    if int(os.environ.get("RANK", "0")) == 0:
        sha = hashlib.sha256()
        with data.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != details["sha256" if smoke_data else "output_sha256"]:
            raise ValueError(f"data SHA256 mismatch: {data}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-mode", choices=("trajectory", "action_window"), required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--smoke-data", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.smoke_data and args.sample_mode != "action_window":
        raise ValueError("--smoke-data only applies to action_window mode")
    config = qwen35_vl_4b_sft_config(hf_path=args.hf_path)
    config.dataset = _make_energon_dataset(
        hf_path=args.hf_path, seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size, global_batch_size=args.global_batch_size,
    )
    native = config.dataset.task_encoder
    image_root = os.environ.get(
        "CUA_IMAGE_ROOT", "/lumos-vePFS/data_pipeline/songweishuai/cua_train_and_benchmark/train/images",
    )
    if args.sample_mode == "trajectory":
        # Match the original entry's construction order for legacy runs.
        config.dataset.task_encoder = CUAQwenVLTaskEncoder(
            tokenizer=native.hf_tokenizer, image_processor=native.image_processor,
            max_padding_length=args.seq_length, min_pixels=native.min_pixels,
            max_pixels=native.max_pixels, max_num_images=native.max_num_images,
            max_num_frames=native.max_num_frames,
            max_visual_tokens=native.max_visual_tokens, image_root=image_root,
        )
        config = process_config_with_overrides(config, cli_overrides=overrides or None)
    else:
        config = process_config_with_overrides(config, cli_overrides=overrides or None)
        if config.dataset.seq_length != config.model.seq_length or config.model.seq_length != args.seq_length:
            raise ValueError("model, dataset and CLI sequence lengths must match")
        if config.dataset.max_num_images not in (5, 6, 7, 10):
            raise ValueError("action_window requires dataset.max_num_images=5, 6, 7 or 10")
        manifest = verify_action_data(config.dataset.path, image_root, args.smoke_data,
                                      config.dataset.max_num_images)
        if not args.smoke_data:
            lane = Path(config.dataset.path).name.split(".")[2]
            expected_gbs = manifest["global_batch_sizes"][lane]
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            parallel_size = (config.model.tensor_model_parallel_size *
                             config.model.pipeline_model_parallel_size * config.model.context_parallel_size)
            if world_size % parallel_size or world_size // parallel_size != manifest["dp"]:
                raise ValueError("runtime DP topology differs from padded data manifest")
            if config.train.global_batch_size != expected_gbs or config.dataset.global_batch_size != expected_gbs:
                raise ValueError("global batch differs from padded data manifest")
        config.dataset = ActionWindowDatasetProvider.from_existing(config.dataset)
        config.dataset.task_encoder = ActionWindowEncoder(
            tokenizer=native.hf_tokenizer, image_processor=native.image_processor,
            max_padding_length=config.dataset.seq_length,
            min_pixels=config.dataset.min_pixels, max_pixels=config.dataset.max_pixels,
            max_num_images=config.dataset.max_num_images,
            max_num_frames=config.dataset.max_num_frames,
            max_visual_tokens=config.dataset.max_visual_tokens, image_root=image_root,
        )
    config.mixed_precision = get_mixed_precision_config(config.mixed_precision)
    config.mixed_precision.grad_reduce_in_fp32 = False
    finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
