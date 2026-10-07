from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import download_data


class ComplexDataVerificationTest(unittest.TestCase):
    def test_manifest_hashes_detect_changed_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "action_window_current"
            data.mkdir()
            image_root = root / "separate-images"
            (image_root / "images_cursor_crosspage").mkdir(parents=True)
            lanes = {}
            for lane in ("func", "gui"):
                payload = f'{{"lane":"{lane}"}}\n'.encode()
                path = data / f"train.action_window.{lane}.jsonl"
                path.write_bytes(payload)
                path.with_suffix(".jsonl.idx").write_bytes(b"0\n")
                lanes[lane] = {"output_sha256": hashlib.sha256(payload).hexdigest()}
            (data / "conversion_manifest.json").write_text(
                json.dumps({"format": "slides01_action_window_padded_v4", "lanes": lanes}),
                encoding="utf-8",
            )
            download_data.verify("complex", root, image_root)
            (data / "train.action_window.func.jsonl").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                download_data.verify("complex", root, image_root)


if __name__ == "__main__":
    unittest.main()
