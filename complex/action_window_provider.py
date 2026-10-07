"""Qwen Energon provider with deterministic unshuffled action rows."""

from __future__ import annotations

from dataclasses import fields

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.recipes.qwen_vl.qwen3_vl import QwenVLEnergonProvider


class ActionWindowDatasetProvider(QwenVLEnergonProvider):
    @classmethod
    def from_existing(cls, provider):
        return cls(**{field.name: getattr(provider, field.name) for field in fields(cls) if field.init})

    def build_datasets(self, context):
        if not self.path or self.task_encoder is None:
            raise ValueError("action-window path and encoder are required")
        self.task_encoder.seq_len = self.seq_length
        self.task_encoder.seq_length = self.seq_length
        self.task_encoder.min_pixels = self.min_pixels
        self.task_encoder.max_pixels = self.max_pixels
        self.task_encoder.max_num_images = self.max_num_images
        self.task_encoder.max_num_frames = self.max_num_frames
        self.task_encoder.max_visual_tokens = self.max_visual_tokens
        dataset = EnergonMultiModalDataModule(
            path=self.path,
            tokenizer=context.tokenizer if context.tokenizer is not None else self.tokenizer,
            image_processor=self.image_processor,
            seq_length=self.seq_length,
            task_encoder=self.task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            pg_collection=context.pg_collection,
            shuffle_buffer_size=0,
        )
        return iter(dataset.train_dataloader()), iter(dataset.val_dataloader()), iter(dataset.val_dataloader())
