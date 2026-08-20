#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moss_runtime as moss


class MossRuntimeTest(unittest.TestCase):
    def test_chunk_plan_and_last_chunk_crop(self):
        chunks = moss.plan_audio_chunks(30 * moss.SAMPLE_RATE + 1)
        self.assertEqual([375, 1], [chunk.audio_tokens for chunk in chunks])
        self.assertEqual([moss.CHUNK_SAMPLES, 1], [chunk.valid_samples for chunk in chunks])

    def test_duration_boundary_is_explicit(self):
        self.assertEqual(10, len(moss.plan_audio_chunks(moss.MAX_AUDIO_SAMPLES)))
        with self.assertRaisesRegex(moss.MossInputError, "exceeds 300 seconds"):
            moss.plan_audio_chunks(moss.MAX_AUDIO_SAMPLES + 1)
        with self.assertRaisesRegex(moss.MossInputError, "empty"):
            moss.plan_audio_chunks(0)

    def test_time_markers_do_not_consume_audio_embeddings(self):
        digits = {str(value): 100 + value for value in range(10)}
        ids = moss.build_audio_span_ids(63, 7, digits)
        self.assertEqual(63, ids.count(7))
        self.assertEqual([105], [token for token in ids if token != 7])
        self.assertEqual(list(range(63)), list(range(len(moss.audio_embedding_positions(ids, 7)))))
        self.assertEqual(63, len(moss.validate_embedding_count(ids, 7, 63)))

    def test_time_markers_match_pinned_processor_interval(self):
        digits = {str(value): 100 + value for value in range(10)}
        ids = moss.build_audio_span_ids(120 * 25 // 2, 7, digits)
        marker_ids = [token for token in ids if token != 7]
        expected = [
            token
            for second in range(5, 121, 5)
            for token in (digits[digit] for digit in str(second))
        ]
        self.assertEqual(expected, marker_ids)

    def test_placeholder_expansion_encodes_sides_separately(self):
        digits = {str(value): 100 + value for value in range(10)}
        ids = moss.expand_audio_placeholder(
            "a<A>b",
            audio_token="<A>",
            audio_token_count=63,
            audio_token_id=7,
            digit_token_ids=digits,
            encode=lambda value: [ord(char) for char in value],
        )
        self.assertEqual(ord("a"), ids[0])
        self.assertEqual(ord("b"), ids[-1])
        self.assertEqual(63, ids.count(7))
        self.assertEqual([105], [token for token in ids[1:-1] if token != 7])

    def test_sequence_limit_is_not_silently_truncated(self):
        digits = {str(value): value for value in range(10)}
        with self.assertRaisesRegex(moss.MossInputError, "exceeds 5 tokens"):
            moss.expand_audio_placeholder(
                "<A>",
                audio_token="<A>",
                audio_token_count=6,
                audio_token_id=9,
                digit_token_ids=digits,
                encode=lambda value: [],
                max_sequence_tokens=5,
            )

    def test_logits_range_policy(self):
        self.assertEqual(64.0, moss.logits_symmetric_range(1.0))
        self.assertEqual(128.0, moss.logits_symmetric_range(100.0))
        self.assertEqual(256.0, moss.logits_symmetric_range(117.0))

    def test_transcript_parser_preserves_status(self):
        segments, error = moss.parse_transcript("[0.00][S01]你好[1.25]\n[1.25][S02]再见[2.0]")
        self.assertIsNone(error)
        self.assertEqual(["S01", "S02"], [segment.speaker for segment in segments])
        self.assertEqual("你好", segments[0].text)
        result = moss.result_document(
            raw_text="unparseable",
            timings_ms={"total": 1000.0},
            duration_seconds=2.0,
            peak_pss_bytes=None,
            generated_tokens=moss.MAX_NEW_TOKENS,
        )
        self.assertEqual("error", result["status"])
        self.assertEqual(
            ["transcript_parse_failed", "token_budget_exhausted"], result["errors"]
        )
        self.assertEqual("unparseable", result["raw_text"])


if __name__ == "__main__":
    unittest.main()
