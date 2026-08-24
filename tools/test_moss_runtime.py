#!/usr/bin/env python3

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moss_runtime as moss


class MossRuntimeTest(unittest.TestCase):
    def test_qnn_partition_keeps_position_ids_on_cpu_before_rope(self):
        root = Path(__file__).resolve().parents[1]
        rebuild = (root / "tools/rebuild_moss_transcribe_diarize_qnn.sh").read_text(
            encoding="utf-8"
        )
        cast = rebuild.index("--cpu-op /rotary/Cast_output_0")
        reshape = rebuild.index("--cpu-op /rotary/Reshape_output_0")
        multiply = rebuild.index("--cpu-op /rotary/Mul_output_0")
        self.assertLess(cast, reshape)
        self.assertLess(reshape, multiply)

    def test_qnn_cpu_partition_generator_forwards_debug_outputs(self):
        root = Path(__file__).resolve().parents[1]
        generator = (
            root / "experiments/ablation/generate_qnn_with_cpu_ops.py"
        ).read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--debug-output"', generator)
        self.assertIn('config["debug_outputs"] = debug_outputs', generator)
        self.assertIn("wrapper_args.cpu_op, wrapper_args.debug_output", generator)
        self.assertIn("args, args.chunk_size, hidden_size, mask_type", generator)

    def test_late_teacher_probe_window_is_forwarded_to_native_runner(self):
        root = Path(__file__).resolve().parents[1]
        native = (root / "tools/mnn_teacher_forced_logits.cpp").read_text(
            encoding="utf-8"
        )
        runner = (root / "tools/run_moss_teacher_forced_sm8850.sh").read_text(
            encoding="utf-8"
        )
        comparator = (root / "tools/compare_teacher_forced_logits.mjs").read_text(
            encoding="utf-8"
        )
        self.assertIn('std::getenv("MOSS_TEACHER_START_STEP")', native)
        self.assertIn('std::getenv("MOSS_TEACHER_SAMPLE_INTERVAL")', native)
        self.assertIn('\\"start_step\\"', native)
        self.assertIn("export MOSS_TEACHER_START_STEP='$START_STEP'", runner)
        self.assertIn("export MOSS_TEACHER_SAMPLE_INTERVAL='$SAMPLE_INTERVAL'", runner)
        self.assertIn("header.start_step ?? 0", comparator)
        self.assertIn('{"deq/graph28.bin", 27}', native)
        self.assertIn('write("q_proj", candidate.second, name, outputs.back())', native)
        self.assertIn('"attn_k", candidate.second, name, outputs[2]', native)
        self.assertIn('write("attn_out", effectiveLayer, name, outputs[0])', native)
        self.assertIn('endsWith(name, "deq/graph1.bin")', native)
        self.assertIn('for (int graph = 2; graph <= 28; ++graph)', native)
        self.assertIn('write("attn_q", qnnLayer, name, outputs[1])', native)
        self.assertIn('MOSS_TEACHER_TRACE_PREFILL_LAYER0', native)
        self.assertIn('BoundaryTraceWriter(prefix, prefillLayer0Trace)', native)
        self.assertIn('occurrences_[key]++', native)
        self.assertIn('[[ ! -s "$destination" ]]', runner)
        self.assertIn('RUN_STATUS=1', runner)

    def test_device_runner_always_verifies_prompt_contract(self):
        root = Path(__file__).resolve().parents[1]
        runner = (root / "tools/run_moss_sm8850.sh").read_text(encoding="utf-8")
        self.assertNotIn('if [[ -z "$EVIDENCE_DIR" ]]; then\n  run_device', runner)
        self.assertIn('TEMP_EVIDENCE_ROOT="$(mktemp -d)"', runner)
        self.assertIn('runner succeeded without a result JSON', runner)
        self.assertIn('verify_moss_device_prompt.py', runner)

    def test_native_prompt_matches_reference_fixture_exactly(self):
        root = Path(__file__).resolve().parents[1]
        expected = (root / "fixtures/moss/default-transcription-prompt.txt").read_text(
            encoding="utf-8"
        )
        self.assertTrue(expected.endswith("\n"))

        sources = [
            root / "tools/moss_tokenizer_probe.cpp",
            root / "third_party/patches/mnn/0007-add-moss-native-runner.patch",
        ]
        for path in sources:
            source = path.read_text(encoding="utf-8")
            if path.suffix == ".patch":
                source = "\n".join(
                    line[1:] if line.startswith("+") else line
                    for line in source.splitlines()
                )
            match = re.search(
                r"const char\* kDefaultPrompt\s*=\s*((?:\s*\"(?:\\.|[^\"\\])*\"\s*)+);",
                source,
            )
            self.assertIsNotNone(match, path)
            literals = re.findall(r'\"(?:\\.|[^\"\\])*\"', match.group(1))
            actual = "".join(ast.literal_eval(literal) for literal in literals)
            self.assertEqual(expected, actual, path)

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
