#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = load_module("verify_source_manifest", "tools/verify_source_manifest.py")
RELEASE = load_module("verify_release_manifest", "tools/verify_release_manifest.py")
sys.modules["verify_release_manifest"] = RELEASE
RUNTIME_LOG = load_module("analyze_qnn_runtime_log", "tools/analyze_qnn_runtime_log.py")
CPU_GATE = load_module("verify_cpu_release_candidate", "tools/verify_cpu_release_candidate.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SourceManifestTest(unittest.TestCase):
    def test_strict_payload_and_cache_handling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            payload = b"model"
            (model / "config.json").write_bytes(payload)
            (model / ".cache").mkdir()
            (model / ".cache/download.lock").write_text("ignored", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "meetnote.model_source_manifest.v1",
                        "model": {"id": "example/model", "revision": "abc"},
                        "files": {"config.json": sha256(payload)},
                    }
                ),
                encoding="utf-8",
            )
            result = SOURCE.verify(model, manifest)
            self.assertEqual("valid", result["status"])
            (model / "unexpected.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                SOURCE.verify(model, manifest)
            self.assertEqual(
                ["unexpected.bin"], SOURCE.verify(model, manifest, True)["extra_files"]
            )

    def test_manifest_cannot_escape_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "meetnote.model_source_manifest.v1",
                        "model": {"id": "example/model", "revision": "abc"},
                        "files": {"../secret": "0" * 64},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsafe"):
                SOURCE.verify(model, manifest)


class ReleaseManifestTest(unittest.TestCase):
    def test_rejects_unmanifested_qnn_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            qnn = Path(directory)
            wrapper = b"wrapper"
            graph = b"graph"
            (qnn / "llm.mnn").write_bytes(wrapper)
            (qnn / "graph0.bin").write_bytes(graph)
            manifest = qnn / "assembly-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "format": "meetnote.qnn_cpu_rope_assembly.v1",
                        "output_graph_count": 1,
                        "wrapper_sha256": sha256(wrapper),
                        "contexts": [{"output": 0, "sha256": sha256(graph)}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual("valid", RELEASE.verify(qnn, manifest)["status"])
            (qnn / "graph99.bin").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "unmanifested"):
                RELEASE.verify(qnn, manifest)


class RuntimeLogGateTest(unittest.TestCase):
    def test_mapping_warning_and_fatal_signals_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "device.log"
            log.write_text(
                "Information: startup complete\n"
                "map result 8003 err 1002\n"
                "Fatal signal 11 SIGSEGV\n",
                encoding="utf-8",
            )
            result = RUNTIME_LOG.analyze([log])
            self.assertEqual("invalid", result["status"])
            self.assertEqual(1, result["counts"]["shared_weight_mapping_failure"])
            self.assertEqual(1, result["counts"]["process_crash"])
            self.assertEqual(0, result["counts"]["non_finite_output"])
            self.assertEqual(sha256(log.read_bytes()), result["files"][0]["sha256"])


class CpuReleaseGateTest(unittest.TestCase):
    def test_repeated_cpu_hashes_and_token_ids_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_hash = sha256(b"llm.mnn.weight")
            candidate = {
                "export": {"llm_weight_sha256": expected_hash},
                "policy": {"determinism_min_runs": 2},
            }
            export_paths = []
            export_dirs = []
            for index in range(2):
                path = root / f"export-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "format": "meetnote.export_verification.v1",
                            "device": "cpu",
                            "llm_weight_sha256": expected_hash,
                        }
                    ),
                    encoding="utf-8",
                )
                export_paths.append(path)
                export_dir = root / f"model-{index}"
                export_dir.mkdir()
                for name in (
                    "config.json",
                    "llm_config.json",
                    "llm.mnn.weight",
                    "tokenizer.mtok",
                    "embeddings_bf16.bin",
                    "llm.mnn",
                ):
                    (export_dir / name).write_bytes(name.encode())
                export_dirs.append(export_dir)
            with mock.patch.object(CPU_GATE, "canonical_mnn_sha256", return_value="b" * 64):
                result = CPU_GATE.validate_export_runs(
                    candidate, export_paths, export_dirs, Path("/unused/MNNConvert")
                )
            self.assertEqual(1, result["unique_hashes"])

            wrong_candidate = {
                "export": {"llm_weight_sha256": "a" * 64},
                "policy": {"determinism_min_runs": 2},
            }
            for path in export_paths:
                record = json.loads(path.read_text(encoding="utf-8"))
                record["llm_weight_sha256"] = "a" * 64
                path.write_text(json.dumps(record), encoding="utf-8")
            with (
                mock.patch.object(CPU_GATE, "canonical_mnn_sha256", return_value="b" * 64),
                self.assertRaisesRegex(ValueError, "payload weight differs"),
            ):
                CPU_GATE.validate_export_runs(
                    wrong_candidate, export_paths, export_dirs, Path("/unused/MNNConvert")
                )

            token_record = {
                "format": "meetnote.token_ids.v1",
                "prompt_ids": [1, 2],
                "reference_ids": [3, 4],
            }
            token_paths = []
            for index in range(2):
                path = root / f"tokens-{index}.json"
                path.write_text(json.dumps(token_record), encoding="utf-8")
                token_paths.append(path)
            self.assertTrue(CPU_GATE.validate_token_ids(token_paths)["exact_match"])

            token_record["reference_ids"] = [3, 5]
            token_paths[1].write_text(json.dumps(token_record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not identical"):
                CPU_GATE.validate_token_ids(token_paths)

    def test_device_record_is_bound_to_candidate_and_qnn_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qnn = root / "qnn"
            qnn.mkdir()
            manifest = qnn / "assembly-manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            candidate = {
                "release_id": "cpu-v1",
                "export": {"llm_weight_sha256": "a" * 64},
            }
            validation = root / "validation.json"
            validation.write_text(
                json.dumps(
                    {
                        "format": "meetnote.qnn_release_validation.v1",
                        "release_id": "cpu-v1",
                        "llm_weight_sha256": "a" * 64,
                        "assembly_manifest_sha256": CPU_GATE.sha256(manifest),
                        "clean_rebuild": {
                            "soc": "SM8850",
                            "dsp_arch": "v81",
                            "all_context_sizes_match": True,
                            "process_crash": False,
                            "dsp_ssr": False,
                            "non_finite_logits": False,
                            "shared_weight_mapping_failure": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = CPU_GATE.validate_device(validation, candidate, qnn)
            self.assertEqual("cpu-v1", result["record"]["release_id"])
            document = json.loads(validation.read_text(encoding="utf-8"))
            document["llm_weight_sha256"] = "b" * 64
            validation.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different exported weight"):
                CPU_GATE.validate_device(validation, candidate, qnn)

    def test_runtime_gate_binds_clean_log_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "device.log"
            log.write_text("QNN startup complete\n", encoding="utf-8")
            gate = root / "runtime-gate.json"
            gate.write_text(
                json.dumps(RUNTIME_LOG.analyze([log])),
                encoding="utf-8",
            )
            self.assertEqual("valid", CPU_GATE.validate_runtime_gate(gate)["record"]["status"])
            log.write_text("Fatal signal 11 SIGSEGV\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from gate evidence"):
                CPU_GATE.validate_runtime_gate(gate)

    def test_probe_metadata_is_bound_to_token_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_ids = [10, 11]
            reference_ids = [20, 21]
            for name, values in (("cpu", (1.0, 0.0)), ("cuda", (1.0, 0.0)), ("hf", (1.0, 0.0))):
                prefix = root / name
                records = [
                    {
                        "type": "header",
                        "format": "meetnote.teacher_logits.v1",
                        "prompt_tokens": 2,
                        "reference_tokens": 2,
                        "prompt_ids": prompt_ids,
                        "reference_ids": reference_ids,
                        "steps": 1,
                        "vocab_size": 2,
                    },
                    {
                        "type": "step",
                        "step": 0,
                        "input_token": 20,
                        "target_token": 21,
                        "input_piece": "a",
                        "target_piece": "b",
                        "top": [{"token": 0}],
                    },
                    {"type": "footer", "steps_written": 1},
                ]
                Path(f"{prefix}.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                Path(f"{prefix}.f32").write_bytes(struct.pack("<2f", *values))

            candidate = {
                "policy": {
                    "quality": {
                        "minimum_teacher_steps": 1,
                        "minimum_cpu_cuda_top1": 1.0,
                        "maximum_hf_top1_regression": 0.0,
                        "maximum_hf_cosine_regression": 0.0,
                    }
                }
            }
            args = argparse.Namespace(
                comparison_tool=ROOT / "tools/compare_teacher_forced_logits.mjs",
                cpu_probe_prefix=root / "cpu",
                cuda_probe_prefix=root / "cuda",
                hf_probe_prefix=root / "hf",
            )
            contract = {"prompt_ids": prompt_ids, "reference_ids": reference_ids}
            result = CPU_GATE.validate_quality(candidate, args, contract)
            self.assertEqual(1.0, result["cpu_vs_cuda"]["summary"]["top1_agreement"])

            metadata = Path(f"{root / 'hf'}.jsonl")
            records = [json.loads(line) for line in metadata.read_text().splitlines()]
            records[0]["prompt_ids"] = [99]
            metadata.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "prompt IDs differ"):
                CPU_GATE.validate_quality(candidate, args, contract)


class RebuildSafetyTest(unittest.TestCase):
    def test_cpu_export_cannot_enter_qnn_generation(self):
        environment = os.environ.copy()
        environment.update(
            {
                "EXPORT_DEVICE": "cpu",
                "EXPORT_ONLY": "false",
                "HF_MODEL_DIR": "/unused/model",
                "CALIBRATION_DATA": "/unused/calibration.jsonl",
                "QNN_SDK_ROOT": "/unused/qairt",
                "WORK_DIR": "/unused/work",
            }
        )
        result = subprocess.run(
            [str(ROOT / "tools/rebuild_qwen3_4b_release.sh")],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("EXPORT_DEVICE=cpu is experimental", result.stdout)


if __name__ == "__main__":
    unittest.main()
