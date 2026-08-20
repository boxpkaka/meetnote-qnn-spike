#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_module(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ASSEMBLE = load_module("assemble_moss_runtime_payload")
VERIFY = load_module("verify_moss_runtime_payload")
WRITE_MANIFEST = load_module("write_moss_artifact_manifest")


class MossRuntimePayloadTest(unittest.TestCase):
    def test_manifest_records_toolchain_and_audio_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.bin").write_bytes(b"payload")

            manifest = WRITE_MANIFEST.write_manifest(root, 64.0, "27.2.12479018")

            self.assertEqual("27.2.12479018", manifest["android_ndk_revision"])
            self.assertEqual(ASSEMBLE.MOSS_AUDIO_CONTRACT, manifest["audio_token_contract"])
            self.assertEqual(["payload.bin"], list(manifest["files"]))

    def test_assembly_rewrites_context_paths_and_keeps_required_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export"
            components = root / "components"
            output = root / "release"
            export.mkdir()
            (export / "config.json").write_text("{}", encoding="utf-8")
            (export / "llm_config.json").write_text(
                json.dumps(ASSEMBLE.MOSS_AUDIO_CONTRACT), encoding="utf-8"
            )
            original_config = (export / "config.json").read_bytes()
            original_llm_config = (export / "llm_config.json").read_bytes()
            (export / "llm.mnn.weight").write_bytes(b"weight")
            for component, wrapper, _ in ASSEMBLE.COMPONENTS:
                qnn = components / component / "qnn"
                qnn.mkdir(parents=True)
                (qnn / wrapper).write_bytes(b"header:qnn/graph0.bin:tail")
                (qnn / "graph0.bin").write_bytes(component.encode("ascii"))

            result = ASSEMBLE.assemble(export, components, output)

            assembled_config = json.loads((output / "config.json").read_text())
            self.assertEqual([64, 1], assembled_config["chunk_limits"])
            self.assertEqual("normal", assembled_config["precision"])
            self.assertEqual(b"weight", (output / "llm.mnn.weight").read_bytes())
            self.assertIn(b"deq/graph0.bin", (output / "decoder/llm.mnn").read_bytes())
            self.assertIn(b"afq/graph0.bin", (output / "audio_encoder_front/audio_encoder_front.mnn").read_bytes())
            self.assertIn(b"abq/graph0.bin", (output / "audio_encoder_back/audio_encoder_back.mnn").read_bytes())
            self.assertEqual(1, result["context_counts"]["decoder"])
            self.assertEqual(original_config, (export / "config.json").read_bytes())
            self.assertEqual(original_llm_config, (export / "llm_config.json").read_bytes())

    def test_assembly_rejects_processor_contract_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export"
            components = root / "components"
            output = root / "release"
            export.mkdir()
            (export / "config.json").write_text("{}", encoding="utf-8")
            bad_contract = dict(ASSEMBLE.MOSS_AUDIO_CONTRACT)
            bad_contract["time_marker_every_seconds"] = 2
            (export / "llm_config.json").write_text(
                json.dumps(bad_contract), encoding="utf-8"
            )
            (export / "llm.mnn.weight").write_bytes(b"weight")
            for component, wrapper, _ in ASSEMBLE.COMPONENTS:
                qnn = components / component / "qnn"
                qnn.mkdir(parents=True)
                (qnn / wrapper).write_bytes(b"header:qnn/graph0.bin:tail")
                (qnn / "graph0.bin").write_bytes(component.encode("ascii"))

            with self.assertRaisesRegex(ValueError, "audio contract differs"):
                ASSEMBLE.assemble(export, components, output)

    def test_assembly_records_custom_token_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export"
            components = root / "components"
            output = root / "release"
            export.mkdir()
            (export / "config.json").write_text("{}", encoding="utf-8")
            (export / "llm_config.json").write_text(
                json.dumps(ASSEMBLE.MOSS_AUDIO_CONTRACT), encoding="utf-8"
            )
            (export / "llm.mnn.weight").write_bytes(b"weight")
            for component, wrapper, _ in ASSEMBLE.COMPONENTS:
                qnn = components / component / "qnn"
                qnn.mkdir(parents=True)
                (qnn / wrapper).write_bytes(b"header:qnn/graph0.bin:tail")
                (qnn / "graph0.bin").write_bytes(component.encode("ascii"))

            ASSEMBLE.assemble(export, components, output, 10240, 6144)

            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(10240, config["max_all_tokens"])
            self.assertEqual(6144, config["max_new_tokens"])

    def test_verifier_rejects_missing_weight_even_with_matching_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"chunk_limits":[64,1]}', encoding="utf-8")
            payload = (root / "config.json").read_bytes()
            manifest = {
                "format": "meetnote.moss_qnn_artifacts.v1",
                "files": {
                    "config.json": {
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                },
            }
            (root / "artifact-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required runtime files missing"):
                VERIFY.verify(root)

    def test_verifier_rejects_low_decoder_cpu_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export"
            components = root / "components"
            output = root / "release"
            export.mkdir()
            (export / "config.json").write_text('{"precision":"low"}', encoding="utf-8")
            (export / "llm_config.json").write_text(
                json.dumps(ASSEMBLE.MOSS_AUDIO_CONTRACT), encoding="utf-8"
            )
            (export / "llm.mnn.weight").write_bytes(b"weight")
            for component, wrapper, _ in ASSEMBLE.COMPONENTS:
                qnn = components / component / "qnn"
                qnn.mkdir(parents=True)
                (qnn / wrapper).write_bytes(b"header:qnn/graph0.bin:tail")
                (qnn / "graph0.bin").write_bytes(component.encode("ascii"))

            ASSEMBLE.assemble(export, components, output)
            config_path = output / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["precision"] = "low"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            (output / "dsp/cdsp").mkdir(parents=True)
            (output / "dsp/cdsp/libQnnHtpV81Skel.so").write_bytes(b"skel")
            (output / "moss_qnn_runner").write_bytes(b"runner")
            prompt_contract = output / "prompt-contract"
            prompt_contract.mkdir()
            for samples in (480000, 960000, 1440000, 1920000):
                (prompt_contract / f"{samples}.json").write_text("{}", encoding="utf-8")
            manifest = WRITE_MANIFEST.write_manifest(output, 64.0, "27.2.12479018")
            self.assertEqual("low", config["precision"])
            self.assertIn("config.json", manifest["files"])

            with self.assertRaisesRegex(ValueError, "low precision corrupts long-sequence"):
                VERIFY.verify(output)


if __name__ == "__main__":
    unittest.main()
