#!/usr/bin/env python3
"""Download one CUA data release and verify its training JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "Furunhao/cua"
PREFIX = "dataset/cua-training-data"
ATOMIC_SHA256 = {
    "func": "ec5edfc63f8e058282f5b213a5b74454f758273b2bebec595eddc9b3a4e66a92",
    "gui": "4ab6fb3ccdf6ae17678d912f079073d49f16a8fa83d5c12dcb961a41400472e3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(variant: str, root: Path) -> None:
    if variant == "atomic":
        expected = ATOMIC_SHA256
        paths = {lane: root / f"train.3x.{lane}.jsonl" for lane in expected}
    else:
        data = root / "action_window_current"
        manifest = json.loads((data / "conversion_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "slides01_action_window_padded_v4":
            raise ValueError("unexpected complex conversion manifest")
        expected = {lane: manifest["lanes"][lane]["output_sha256"] for lane in ("func", "gui")}
        paths = {lane: data / f"train.action_window.{lane}.jsonl" for lane in expected}
    for lane, path in paths.items():
        if sha256(path) != expected[lane]:
            raise ValueError(f"SHA256 mismatch: {path}")
        if not path.with_suffix(".jsonl.idx").is_file():
            raise FileNotFoundError(f"missing Energon index: {path}.idx")
    if variant == "complex" and not (root / "sft_gui_sharegpt/images_cursor_crosspage").is_dir():
        raise FileNotFoundError("complex image tree is missing")
    if variant == "atomic" and not (root / "images_cursor_regen_20260901").is_dir():
        raise FileNotFoundError("atomic image tree is missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("atomic", "complex"))
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    commit = HfApi().repo_info(REPO_ID, repo_type="dataset", revision=args.revision).sha
    target = f"{PREFIX}/{args.variant}"
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=commit,
        local_dir=args.dest,
        allow_patterns=[f"{target}/*", f"{target}/**"],
    )
    root = args.dest / target
    verify(args.variant, root)
    print(f"revision={commit}")
    print(root.resolve())


if __name__ == "__main__":
    main()
