#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "prepare_moss_calibration", Path(__file__).with_name("prepare_moss_calibration.py")
)
CALIBRATION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CALIBRATION)


class PrepareMossCalibrationTest(unittest.TestCase):
    def test_candidate_phases_include_decode(self):
        starts = CALIBRATION.candidate_starts(400, 800)
        self.assertEqual([336], starts["audio_end"])
        self.assertTrue(starts["decode"])
        self.assertTrue(all(value >= 400 for value in starts["decode"]))

    def test_balanced_selection(self):
        candidates = [
            {"case_id": f"c{index:03d}", "phase": phase, "start": index * 64}
            for phase in CALIBRATION.PHASES
            for index in range(40)
        ]
        selected = CALIBRATION.select_balanced(candidates)
        self.assertEqual(128, len(selected))
        self.assertEqual(
            {phase: 32 for phase in CALIBRATION.PHASES},
            {phase: sum(row["phase"] == phase for row in selected) for phase in CALIBRATION.PHASES},
        )


if __name__ == "__main__":
    unittest.main()
