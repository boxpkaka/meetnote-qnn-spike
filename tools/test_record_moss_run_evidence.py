#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_moss_run_evidence as evidence


class RecordMossRunEvidenceTest(unittest.TestCase):
    def test_extracts_last_result_and_hashes_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            output = root / "evidence"
            release.mkdir()
            output.mkdir()
            (release / "artifact-manifest.json").write_text("{}", encoding="utf-8")
            (release / "moss_qnn_runner").write_bytes(b"runner")
            wav = root / "input.wav"
            wav.write_bytes(b"wav")
            log = output / "run.log"
            log.write_text(
                "diagnostic\n"
                + json.dumps(
                    {
                        "format": "meetnote.moss_qnn_result.v1",
                        "status": "ok",
                        "prompt_token_ids": [1, 2],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output / "prompt-alignment.json").write_text("{}", encoding="utf-8")

            record = evidence.record(
                release,
                wav,
                log,
                output,
                exit_code=0,
                device_model="V2505A",
                soc_model="SM8850",
                android_version="16",
            )

            self.assertTrue(record["result_present"])
            self.assertEqual("ok", record["result_status"])
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual([1, 2], result["prompt_token_ids"])
            self.assertIn("sha256", record["files"]["run_log"])
            self.assertIn("prompt_alignment", record["files"])

    def test_preserves_crash_evidence_without_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            output = root / "evidence"
            release.mkdir()
            output.mkdir()
            (release / "artifact-manifest.json").write_text("{}", encoding="utf-8")
            (release / "moss_qnn_runner").write_bytes(b"runner")
            wav = root / "input.wav"
            wav.write_bytes(b"wav")
            log = output / "run.log"
            log.write_text("native crash\n", encoding="utf-8")

            record = evidence.record(
                release,
                wav,
                log,
                output,
                exit_code=139,
                device_model="V2505A",
                soc_model="SM8850",
                android_version="16",
            )

            self.assertFalse(record["result_present"])
            self.assertEqual(139, record["process_exit_code"])
            self.assertFalse((output / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
