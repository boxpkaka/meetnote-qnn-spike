import unittest

from project_moss_static_decoder_profile import project


class ProjectStaticDecoderProfileTest(unittest.TestCase):
    def test_projects_layer_replacement_and_remaining_qnn_calls(self):
        profile = {
            "timings_ms": {"decode": 100.0, "total": 130.0},
            "decode_profile": {
                "unaccounted_ms": 5.0,
                "history_buckets": [
                    {
                        "cpu_attention_calls": 28,
                        "qnn_graph_calls": 31,
                        "qnn_execute_sync_ms": 62.0,
                        "qnn_input_copy_ms": 1.0,
                        "qnn_output_copy_ms": 2.0,
                        "qnn_state_update_ms": 3.0,
                    }
                ],
            },
        }
        plan = {
            "name": "test",
            "layer_count": 28,
            "audio_duration_ms": 100.0,
            "segments": [{"layer_calls": 28, "latency_ms_per_layer": 2.0}],
        }

        result = project(
            profile,
            plan,
            {"total_ms": 260.0, "decode_ms": 200.0},
        )

        self.assertEqual(result["remaining_qnn_graph_calls"], 3)
        self.assertAlmostEqual(
            result["components_ms"]["remaining_qnn_execute_sync"], 6.0
        )
        self.assertAlmostEqual(
            result["measured_baseline"]["projected_decode_ms"], 73.0
        )
        self.assertAlmostEqual(result["measured_baseline"]["projected_rtf"], 1.03)
        self.assertAlmostEqual(
            result["conservative_baseline"]["projected_rtf"], 2.06
        )

    def test_rejects_incomplete_plan(self):
        profile = {
            "timings_ms": {"decode": 1.0, "total": 1.0},
            "decode_profile": {
                "unaccounted_ms": 0.0,
                "history_buckets": [
                    {
                        "cpu_attention_calls": 28,
                        "qnn_graph_calls": 28,
                        "qnn_execute_sync_ms": 0.0,
                        "qnn_input_copy_ms": 0.0,
                        "qnn_output_copy_ms": 0.0,
                        "qnn_state_update_ms": 0.0,
                    }
                ],
            },
        }
        plan = {
            "name": "bad",
            "layer_count": 28,
            "audio_duration_ms": 1.0,
            "segments": [{"layer_calls": 27, "latency_ms_per_layer": 1.0}],
        }
        with self.assertRaisesRegex(ValueError, "plan covers 27"):
            project(profile, plan)


if __name__ == "__main__":
    unittest.main()
