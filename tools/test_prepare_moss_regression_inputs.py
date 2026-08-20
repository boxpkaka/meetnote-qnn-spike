#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_moss_regression_inputs as regression


class PrepareMossRegressionInputsTest(unittest.TestCase):
    def test_writes_exact_prefixes_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as target:
                target.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                target.writeframes(b"\x01\x00" * (3 * 16_000))

            manifest = regression.prepare(source, root / "output", [1, 2, 3])

            self.assertEqual([16_000, 32_000, 48_000], [row["sample_count"] for row in manifest["cases"]])
            saved = json.loads((root / "output/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest, saved)

    def test_rejects_unsorted_or_too_long_durations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as target:
                target.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                target.writeframes(b"\0\0" * 16_000)
            with self.assertRaisesRegex(ValueError, "unique and increasing"):
                regression.prepare(source, root / "bad-order", [1, 1])
            with self.assertRaisesRegex(ValueError, "shorter"):
                regression.prepare(source, root / "too-long", [2])


if __name__ == "__main__":
    unittest.main()
