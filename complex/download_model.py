#!/usr/bin/env python3
"""Download the public Qwen3.5-4B model used for config and processor files."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "Qwen/Qwen3.5-4B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    commit = HfApi().model_info(MODEL_ID, revision=args.revision).sha
    snapshot_download(repo_id=MODEL_ID, revision=commit, local_dir=args.dest)
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        if not (args.dest / name).is_file():
            raise FileNotFoundError(args.dest / name)
    print(f"model_revision={commit}")
    print(args.dest.resolve())


if __name__ == "__main__":
    main()
