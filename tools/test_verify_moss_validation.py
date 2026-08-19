#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "verify_moss_validation", Path(__file__).with_name("verify_moss_validation.py")
)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


def valid_record() -> dict:
    return {
        "format": "meetnote.moss_qnn_validation.v1",
        "release_id": "moss-transcribe-diarize-0.9b-sm8850-v81-poc-v1",
        "component_alignment": {
            "log_mel": {
                "shape_exact": True,
                "all_finite": True,
                "cosine": 0.9999,
                "mean_abs_error": 1e-3,
                "max_abs_error": 0.04,
            },
            "processor_tokens": {
                "single_chunk_exact": True,
                "multi_chunk_exact": True,
                "prompt_exact": True,
            },
            "audio": {
                "cpu_frontend_hf_bf16_cosine": 0.995,
                "hf_bf16_qnn_cosine": 0.995,
            },
            "decoder": {
                "qnn_mnn_top1_agreement": 0.99,
                "qnn_mnn_logits_cosine": 0.98,
                "logits_saturation": False,
                "comparison_layers": ["hf_bf16", "mnn_cpu", "qnn_htp"],
            },
        },
        "quality": {
            "case_count": 10,
            "all_duration_seconds": 300,
            "all_outputs_parseable": True,
            "hf_bf16": {"cer": 0.2, "cpcer": 0.2, "timestamp_boundary_mae_seconds": 0.1},
            "qnn": {"cer": 0.21, "cpcer": 0.21, "timestamp_boundary_mae_seconds": 0.2},
            "cases": [
                {"speaker_count": 2, "missed_speakers": [], "merged_speakers": []}
                for _ in range(10)
            ],
        },
        "device": {
            "soc_id": 87,
            "dsp_arch": "v81",
            "qairt_version": "2.48.40.260702",
            "p95_rtf": 1.0,
            "peak_pss_bytes": 4 * 1024**3,
            "htp_ssr": False,
            "oom": False,
            "crash": False,
            "non_finite_output": False,
            "shared_weight_mapping_failure": False,
            "timings_ms": {
                name: 1
                for name in (
                    "audio_encoder_front",
                    "audio_encoder_back",
                    "prefill",
                    "decode",
                    "cpu_preprocess",
                )
            },
        },
    }


class VerifyMossValidationTest(unittest.TestCase):
    def test_accepts_boundaries(self):
        self.assertEqual("valid", VERIFY.validate(valid_record())["status"])

    def test_rejects_regression(self):
        record = copy.deepcopy(valid_record())
        record["quality"]["qnn"]["cer"] = 0.211
        with self.assertRaisesRegex(ValueError, "CER regression"):
            VERIFY.validate(record)

    def test_rejects_device_failure(self):
        record = copy.deepcopy(valid_record())
        record["device"]["htp_ssr"] = True
        with self.assertRaisesRegex(ValueError, "htp_ssr"):
            VERIFY.validate(record)

    def test_log_mel_max_error_is_recorded_but_not_a_gate(self):
        record = valid_record()
        record["component_alignment"]["log_mel"]["max_abs_error"] = 0.04
        self.assertEqual("valid", VERIFY.validate(record)["status"])

    def test_rejects_log_mel_mean_error(self):
        record = valid_record()
        record["component_alignment"]["log_mel"]["mean_abs_error"] = 0.001001
        with self.assertRaisesRegex(ValueError, "log-mel MAE"):
            VERIFY.validate(record)

    def test_rejects_cpu_frontend_embedding_regression(self):
        record = valid_record()
        record["component_alignment"]["audio"]["cpu_frontend_hf_bf16_cosine"] = 0.9949
        with self.assertRaisesRegex(ValueError, "CPU frontend audio embedding"):
            VERIFY.validate(record)


if __name__ == "__main__":
    unittest.main()
