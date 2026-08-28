#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from summarize_moss_speaker_only_eval import (
    AmbiguousCaseError,
    MissingCaseError,
    find_case,
)


class SummarizeMossSpeakerOnlyTest(unittest.TestCase):
    def test_find_case_fails_when_result_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MissingCaseError):
                find_case(Path(directory), "case-1")

    def test_find_case_fails_when_results_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shard in ("a", "b"):
                path = root / shard / "case-1.json"
                path.parent.mkdir()
                path.write_text(json.dumps({"case": "case-1"}), encoding="utf-8")

            with self.assertRaises(AmbiguousCaseError):
                find_case(root, "case-1")


if __name__ == "__main__":
    unittest.main()
