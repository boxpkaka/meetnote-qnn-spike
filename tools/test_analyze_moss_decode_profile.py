#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_moss_decode_profile as profile


class AnalyzeMossDecodeProfileTest(unittest.TestCase):
    def test_computes_amdahl_gate_from_measured_components(self):
        result = {
            "format": "meetnote.moss_qnn_result.v1",
            "timings_ms": {"decode": 418_550.0, "total": 490_511.0},
            "decode_profile": {
                "history_buckets": [
                    {
                        "cpu_attention_ms": 302_880.87,
                        "cpu_attention_compute_ms": 256_732.3,
                        "cpu_attention_kv_update_ms": 46_148.57,
                        "qnn_input_copy_ms": 296.50394,
                        "qnn_execute_sync_ms": 100_243.87,
                        "qnn_output_copy_ms": 425.0342,
                        "qnn_state_update_ms": 6.72712,
                    }
                ]
            },
        }

        report = profile.analyze(result, audio_duration_ms=300_000.0)

        self.assertAlmostEqual(
            302_880.87 / 418_550.0,
            report["decode_breakdown"]["cpu_attention_share"],
        )
        self.assertAlmostEqual(228_039.0, report["timing"]["decode_budget_for_rtf_1_ms"])
        self.assertAlmostEqual(
            2.695,
            report["cross_runtime_overhead_scenarios"][0]["required_attention_speedup"],
            places=3,
        )
        self.assertTrue(report["attention_speedup_projections"][0]["meets_rtf_1"])

    def test_scales_profile_shares_to_a_separate_budget_run(self):
        result = {
            "format": "meetnote.moss_qnn_result.v1",
            "timings_ms": {"decode": 400.0, "total": 500.0},
            "decode_profile": {
                "history_buckets": [
                    {"cpu_attention_ms": 300.0, "qnn_execute_sync_ms": 80.0}
                ]
            },
        }
        budget = {
            "format": "meetnote.moss_qnn_result.v1",
            "timings_ms": {"decode": 600.0, "total": 750.0},
        }

        report = profile.analyze(
            result,
            audio_duration_ms=1_000.0,
            budget_result=budget,
        )

        self.assertEqual(1.5, report["profile_source"]["component_timing_scale"])
        self.assertEqual(450.0, report["decode_breakdown"]["cpu_attention_ms"])
        self.assertEqual(450.0, report["attention_speedup_projections"][0]["projected_total_ms"])

    def test_rejects_profile_that_exceeds_decode_time(self):
        result = {
            "format": "meetnote.moss_qnn_result.v1",
            "timings_ms": {"decode": 10.0, "total": 20.0},
            "decode_profile": {"history_buckets": [{"cpu_attention_ms": 11.0}]},
        }

        with self.assertRaisesRegex(ValueError, "exceeds decode time"):
            profile.analyze(result, audio_duration_ms=30.0)


if __name__ == "__main__":
    unittest.main()
