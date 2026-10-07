from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import download_data


class AtomicDataVerificationTest(unittest.TestCase):
    def test_both_lanes_require_matching_jsonl_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "separate-images"
            (image_root / "images_cursor_regen_20260901").mkdir(parents=True)
            expected = {}
            for lane in ("func", "gui"):
                payload = f'{{"lane":"{lane}"}}\n'.encode()
                path = root / f"train.3x.{lane}.jsonl"
                path.write_bytes(payload)
                path.with_suffix(".jsonl.idx").write_bytes(b"0\n")
                expected[lane] = hashlib.sha256(payload).hexdigest()
            previous = download_data.ATOMIC_SHA256
            download_data.ATOMIC_SHA256 = expected
            try:
                download_data.verify("atomic", root, image_root)
                (root / "train.3x.gui.jsonl.idx").unlink()
                with self.assertRaises(FileNotFoundError):
                    download_data.verify("atomic", root, image_root)
            finally:
                download_data.ATOMIC_SHA256 = previous


if __name__ == "__main__":
    unittest.main()
