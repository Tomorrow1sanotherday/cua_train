#!/usr/bin/env python3
"""Storage adapter from the source CUA JSONL record to Qwen's native task encoder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from megatron.bridge.recipes.qwen_vl.data.energon.task_encoder import QwenVLTaskEncoder
from megatron.energon.task_encoder.base import stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys


@stateless
def cook_cua_jsonl(sample: dict[str, Any]) -> dict[str, Any]:
    """Expose an indexed JSONL record to the task encoder without changing its content."""
    return {
        **basic_sample_keys(sample),
        "json": sample["json"],
    }


@dataclass
class _ChatSample:
    __key__: str
    __subflavors__: dict[str, Any] | None
    conversation: str
    imgs: list[Image.Image]
    videos: None = None


class CUAQwenVLTaskEncoder(QwenVLTaskEncoder):
    """Load source image paths while leaving Qwen encoding and skip guards native."""

    cookers = [Cooker(cook_cua_jsonl)]

    def __init__(self, *args, image_root: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_root = Path(image_root)

    def encode_sample(self, sample):
        record = sample["json"]
        conversation: list[dict[str, str]] = []
        system = record.get("system")
        if system:
            conversation.append({"role": "system", "content": system})

        role_map = {
            "human": "user",
            "user": "user",
            "observation": "tool",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
        }
        for turn in record["conversations"]:
            role = role_map[turn["from"]]
            content = turn["value"]
            if role == "tool":
                content = content.replace(
                    "<image>",
                    "<|vision_start|><|image_pad|><|vision_end|>",
                )
            if role == "assistant" and "</think>" in content:
                reasoning = content.split("</think>", 1)[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                response = content.split("</think>", 1)[1].lstrip("\n")
                content = f"<think>\n{reasoning}\n</think>\n\n{response}"
            conversation.append({"role": role, "content": content})

        image_paths = [Path(path) for path in record["images"]]
        image_paths = [path if path.is_absolute() else self.image_root / path for path in image_paths]
        placeholders = sum(
            turn["content"].count("<image>") + turn["content"].count("<|image_pad|>")
            for turn in conversation
        )
        if placeholders != len(image_paths):
            raise ValueError(
                f"{sample.get('__key__')}: {placeholders} image placeholders but {len(image_paths)} paths"
            )

        images: list[Image.Image] = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        adapted = _ChatSample(
            __key__=str(sample.get("__key__", "unknown")),
            __subflavors__=sample.get("__subflavors__"),
            conversation=json.dumps(conversation, ensure_ascii=False, separators=(",", ":")),
            imgs=images,
        )
        return super().encode_sample(adapted)
