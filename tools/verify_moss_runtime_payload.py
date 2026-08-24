#!/usr/bin/env python3
"""Verify a MOSS Android runtime payload before device installation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from assemble_moss_runtime_payload import DECODER_CPU_PRECISION
from moss_runtime import MOSS_AUDIO_CONTRACT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, object]:
    manifest_path = root / "artifact-manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("format") != "meetnote.moss_qnn_artifacts.v1":
        raise ValueError("unsupported MOSS artifact manifest")
    expected = document.get("files", {})
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(expected) != actual:
        raise ValueError(
            f"payload file set differs: missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    for relative, record in expected.items():
        path = root / relative
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"payload digest differs: {relative}")

    required = {
        "config.json",
        "llm_config.json",
        "llm.mnn.weight",
        "decoder/llm.mnn",
        "audio_encoder_front/audio_encoder_front.mnn",
        "audio_encoder_back/audio_encoder_back.mnn",
        "afq/graph0.bin",
        "abq/graph0.bin",
        "dsp/cdsp/libQnnHtpV81Skel.so",
        "moss_qnn_runner",
        "prompt-contract/480000.json",
        "prompt-contract/960000.json",
        "prompt-contract/1440000.json",
        "prompt-contract/1920000.json",
        "prompt-contract/4800000.json",
    }
    missing = required - actual
    if missing:
        raise ValueError(f"required runtime files missing: {sorted(missing)}")
    if any(path.startswith("qnn/") or "/qnn/" in path for path in actual):
        raise ValueError("ambiguous qnn/ runtime path remains in payload")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("chunk_limits") != [64, 1]:
        raise ValueError("decoder chunk_limits must be [64, 1]")
    if config.get("precision") != DECODER_CPU_PRECISION:
        raise ValueError(
            "decoder CPU precision must be "
            f"{DECODER_CPU_PRECISION!r}; low precision corrupts long-sequence output on SM8850"
        )
    llm_config = json.loads((root / "llm_config.json").read_text(encoding="utf-8"))
    for key, expected in MOSS_AUDIO_CONTRACT.items():
        if llm_config.get(key) != expected:
            raise ValueError(
                f"MOSS audio contract differs for {key}: "
                f"expected={expected!r} actual={llm_config.get(key)!r}"
            )
    return {"status": "valid", "file_count": len(actual)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.payload), indent=2))


if __name__ == "__main__":
    main()
