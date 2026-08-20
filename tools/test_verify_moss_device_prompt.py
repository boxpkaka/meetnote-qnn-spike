#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_moss_device_prompt as prompt


class VerifyMossDevicePromptTest(unittest.TestCase):
    def test_accepts_complete_exact_prompt(self):
        report = prompt.verify(
            {
                "format": "meetnote.moss_qnn_result.v1",
                "prompt_tokens": 3,
                "prompt_token_ids_complete": True,
                "prompt_token_ids": [1, 2, 3],
            },
            {
                "format": "meetnote.moss_full_prompt_contract.v1",
                "sample_count": 480000,
                "runtime_input_ids": [1, 2, 3],
            },
        )
        self.assertEqual("valid", report["status"])
        self.assertEqual(3, report["first_mismatch"])

    def test_reports_first_mismatch(self):
        report = prompt.verify(
            {
                "format": "meetnote.moss_qnn_result.v1",
                "prompt_tokens": 3,
                "prompt_token_ids_complete": True,
                "prompt_token_ids": [1, 9, 3],
            },
            {
                "format": "meetnote.moss_full_prompt_contract.v1",
                "sample_count": 480000,
                "runtime_input_ids": [1, 2, 3],
            },
        )
        self.assertEqual("invalid", report["status"])
        self.assertEqual(1, report["first_mismatch"])


if __name__ == "__main__":
    unittest.main()
