#!/usr/bin/env python3

import unittest

from stitch_moss_transcribe_diarize_windows import map_window_speakers


class StitchMossWindowsTest(unittest.TestCase):
    def test_maps_many_speakers_by_maximum_overlap(self):
        local = []
        global_timeline = []
        for index in range(10):
            start_ms = index * 100
            local.append(
                {
                    "localSpeakerId": f"L{index}",
                    "startMs": start_ms,
                    "endMs": start_ms + 100,
                }
            )
            global_timeline.append(
                {
                    "speakerId": f"G{index}",
                    "startMs": start_ms,
                    "endMs": start_ms + 100,
                }
            )

        mapping = map_window_speakers(local, global_timeline)

        self.assertEqual(mapping, {f"L{index}": f"G{index}" for index in range(10)})


if __name__ == "__main__":
    unittest.main()
