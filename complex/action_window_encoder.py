"""Qwen encoder that computes loss on the final assistant turn only."""

from __future__ import annotations

import hashlib

import torch

from cua_energon_adapter import CUAQwenVLTaskEncoder


def target_span(text: torch.Tensor, tokenizer, expected_assistants: int) -> tuple[int, int]:
    ids = text.tolist()
    marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not marker or end_id is None:
        raise ValueError("tokenizer lacks assistant ChatML markers")
    spans: list[tuple[int, int]] = []
    for start in range(len(ids) - len(marker) + 1):
        if ids[start:start + len(marker)] != marker:
            continue
        begin = start + len(marker)
        try:
            end = ids.index(end_id, begin)
        except ValueError as exc:
            raise ValueError("assistant turn has no end marker") from exc
        if begin == 0 or end <= begin:
            raise ValueError("empty or invalid assistant span")
        spans.append((begin - 1, end))
    if len(spans) != expected_assistants or not spans:
        raise ValueError(f"assistant span count {len(spans)} != {expected_assistants}")
    return spans[-1]


def mask_to_target(text: torch.Tensor, target: torch.Tensor, tokenizer, expected_assistants: int) -> int:
    if text.shape != target.shape:
        raise ValueError("text and target shapes differ")
    begin, end = target_span(text, tokenizer, expected_assistants)
    pad = tokenizer.pad_token_id
    if pad is None or end >= len(text):
        raise ValueError("missing pad token or target EOS")
    expected = torch.full_like(target, pad)
    expected[begin:end] = text[begin + 1:end + 1]
    if not torch.equal(target[begin:end], expected[begin:end]):
        raise ValueError("native target differs from final assistant tokens")
    target.copy_(expected)
    return end - begin


class ActionWindowEncoder(CUAQwenVLTaskEncoder):
    def encode_sample(self, sample):
        record = sample["json"]
        key = record.get("sample_key", sample.get("__key__", "unknown"))
        if record.get("sample_mode") != "action_window":
            raise ValueError(f"{key}: wrong sample mode")
        turns = record["conversations"]
        assistants = sum(turn["from"] in ("gpt", "assistant") for turn in turns)
        if not turns or turns[-1]["from"] not in ("gpt", "assistant"):
            raise ValueError(f"{key}: final turn is not assistant")
        if not 1 <= len(record["images"]) <= self.max_num_images:
            raise ValueError(f"{key}: image count outside 1..{self.max_num_images}")
        digest = hashlib.sha256(turns[-1]["value"].encode("utf-8")).hexdigest()
        if digest != record.get("target_action_sha256"):
            raise ValueError(f"{key}: target action hash mismatch")
        if sum(turn["value"].count("<image>") for turn in turns) != len(record["images"]):
            raise ValueError(f"{key}: placeholder/image count mismatch")
        encoded = super().encode_sample(sample)
        try:
            mask_to_target(encoded.text, encoded.target, self.hf_tokenizer, assistants)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
        if record.get("is_padding"):
            if not str(key).startswith("padding:") or record.get("train_probe"):
                raise ValueError(f"{key}: invalid zero-loss padding metadata")
            encoded.target.fill_(self.hf_tokenizer.pad_token_id)
        return encoded
