from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "shared/download_data.py"
SPEC = importlib.util.spec_from_file_location("download_data", MODULE_PATH)
assert SPEC and SPEC.loader
download_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_data)


class DataVerificationTest(unittest.TestCase):
    def test_atomic_verifies_both_lanes_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images_cursor_regen_20260901").mkdir()
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
                download_data.verify("atomic", root)
                (root / "train.3x.gui.jsonl.idx").unlink()
                with self.assertRaises(FileNotFoundError):
                    download_data.verify("atomic", root)
            finally:
                download_data.ATOMIC_SHA256 = previous

    def test_complex_rejects_changed_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "action_window_current"
            data.mkdir()
            (root / "sft_gui_sharegpt/images_cursor_crosspage").mkdir(parents=True)
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
            download_data.verify("complex", root)
            (data / "train.action_window.func.jsonl").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                download_data.verify("complex", root)


if __name__ == "__main__":
    unittest.main()
