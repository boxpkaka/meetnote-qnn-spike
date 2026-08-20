#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_moss_audio_token_contract as contract


class MossAudioTokenContractTest(unittest.TestCase):
    def make_model(self, root: Path) -> None:
        (root / "processor_config.json").write_text(
            json.dumps(
                {
                    "audio_tokens_per_second": 12.5,
                    "audio_merge_size": 4,
                    "time_marker_every_seconds": 5,
                    "enable_time_marker": True,
                }
            ),
            encoding="utf-8",
        )
        (root / "config.json").write_text('{"audio_token_id":151671}', encoding="utf-8")
        added = [
            {"id": 151669, "content": "<|audio_start|>"},
            {"id": 151670, "content": "<|audio_end|>"},
        ]
        vocab = {str(value): 15 + value for value in range(10)}
        (root / "tokenizer.json").write_text(
            json.dumps({"added_tokens": added, "model": {"vocab": vocab}}), encoding="utf-8"
        )

    def test_detects_marker_interval_and_boundary_token_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            llm_config = root / "llm_config.json"
            llm_config.write_text(
                json.dumps(
                    {
                        "audio_pad": 151671,
                        "audio_merge_size": 4,
                        "audio_tokens_per_second": 12.5,
                        "time_marker_every_seconds": 2,
                    }
                ),
                encoding="utf-8",
            )

            result = contract.verify(root, llm_config, [60])

            self.assertEqual("invalid", result["status"])
            self.assertEqual(2, result["mismatches"]["time_marker_every_seconds"]["runtime"])
            self.assertIn("audio_start", result["mismatches"])
            self.assertIn("audio_end", result["mismatches"])

    def test_valid_contract_reports_exact_embedding_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            llm_config = root / "llm_config.json"
            llm_config.write_text(
                json.dumps(
                    {
                        "audio_pad": 151671,
                        "audio_start": 151669,
                        "audio_end": 151670,
                        "audio_merge_size": 4,
                        "audio_tokens_per_second": 12.5,
                        "time_marker_every_seconds": 5,
                    }
                ),
                encoding="utf-8",
            )

            result = contract.verify(root, llm_config, [30, 120])

            self.assertEqual("valid", result["status"])
            self.assertEqual([375, 1500], [case["audio_embedding_count"] for case in result["cases"]])
            self.assertEqual(15, result["cases"][1]["last_marker_ids"][-1])

    def test_boundary_markers_cover_58_60_62_and_three_digit_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            llm_config = root / "llm_config.json"
            llm_config.write_text(
                json.dumps(
                    {
                        "audio_pad": 151671,
                        "audio_start": 151669,
                        "audio_end": 151670,
                        "audio_merge_size": 4,
                        "audio_tokens_per_second": 12.5,
                        "time_marker_every_seconds": 5,
                    }
                ),
                encoding="utf-8",
            )

            result = contract.verify(root, llm_config, [58, 60, 62, 100])

            self.assertEqual([55, 60, 60, 100], [case["time_markers"][-1]["second"] for case in result["cases"]])
            marker_100 = result["cases"][-1]["time_markers"][-1]
            self.assertEqual([16, 15, 15], marker_100["token_ids"])
            self.assertEqual(1240, marker_100["audio_embedding_position"])

    def test_rejects_invalid_full_prompt_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            llm_config = root / "llm_config.json"
            llm_config.write_text(
                json.dumps(
                    {
                        "audio_pad": 151671,
                        "audio_start": 151669,
                        "audio_end": 151670,
                        "audio_merge_size": 4,
                        "audio_tokens_per_second": 12.5,
                        "time_marker_every_seconds": 5,
                    }
                ),
                encoding="utf-8",
            )
            report = root / "prompt.json"
            report.write_text(
                json.dumps(
                    {
                        "format": "meetnote.moss_full_prompt_contract.v1",
                        "status": "invalid",
                        "sample_count": 480000,
                        "official_input_ids": [1],
                        "runtime_input_ids": [2],
                    }
                ),
                encoding="utf-8",
            )

            result = contract.verify(root, llm_config, [30], [report])

            self.assertEqual("invalid", result["status"])
            self.assertRegex(result["full_prompt_errors"][0], "token mismatch")

    def test_rejects_circular_full_prompt_match_that_differs_from_hf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_model(root)
            llm_config = root / "llm_config.json"
            llm_config.write_text(
                json.dumps(
                    {
                        "audio_pad": 151671,
                        "audio_start": 151669,
                        "audio_end": 151670,
                        "audio_merge_size": 4,
                        "audio_tokens_per_second": 12.5,
                        "time_marker_every_seconds": 5,
                    }
                ),
                encoding="utf-8",
            )
            report = root / "prompt.json"
            report.write_text(
                json.dumps(
                    {
                        "format": "meetnote.moss_full_prompt_contract.v1",
                        "status": "valid",
                        "sample_count": 480000,
                        "official_input_ids": [1, 2, 3],
                        "runtime_input_ids": [1, 2, 3],
                    }
                ),
                encoding="utf-8",
            )

            result = contract.verify(root, llm_config, [30], [report])

            self.assertEqual("invalid", result["status"])
            self.assertRegex(result["full_prompt_errors"][0], "pinned HF reference")


if __name__ == "__main__":
    unittest.main()
