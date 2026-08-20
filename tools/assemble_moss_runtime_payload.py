#!/usr/bin/env python3
"""Assemble a collision-free MOSS Android QNN runtime payload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from moss_runtime import MOSS_AUDIO_CONTRACT

RELEASE_ID = "moss-transcribe-diarize-0.9b-sm8850-v81-poc-v1"
DECODER_CPU_PRECISION = "normal"
COMPONENTS = (
    ("decoder", "llm.mnn", "deq"),
    ("audio_encoder_front", "audio_encoder_front.mnn", "afq"),
    ("audio_encoder_back", "audio_encoder_back.mnn", "abq"),
)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def rewrite_context_prefix(source: Path, destination: Path, prefix: str) -> int:
    data = source.read_bytes()
    marker = b"qnn/graph"
    replacement = prefix.encode("ascii") + b"/graph"
    if len(marker) != len(replacement):
        raise ValueError("runtime context prefix must remain three ASCII characters")
    count = data.count(marker)
    if count == 0:
        raise ValueError(f"wrapper has no external QNN context paths: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data.replace(marker, replacement))
    return count


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def assemble(
    export_dir: Path,
    component_root: Path,
    output_dir: Path,
    max_all_tokens: int = 8192,
    max_new_tokens: int = 4096,
) -> dict[str, object]:
    if max_all_tokens < 1 or max_new_tokens < 1 or max_new_tokens >= max_all_tokens:
        raise ValueError("token limits must satisfy 0 < max_new_tokens < max_all_tokens")
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper_references: dict[str, int] = {}
    context_counts: dict[str, int] = {}

    for component, wrapper_name, prefix in COMPONENTS:
        source_dir = component_root / component / "qnn"
        wrapper = require_file(source_dir / wrapper_name)
        contexts = sorted(source_dir.glob("graph*.bin"))
        if not contexts or any(path.stat().st_size == 0 for path in contexts):
            raise ValueError(f"missing context binaries for {component}")
        wrapper_references[component] = rewrite_context_prefix(
            wrapper, output_dir / component / wrapper_name, prefix
        )
        for context in contexts:
            link_or_copy(context, output_dir / prefix / context.name)
        context_counts[component] = len(contexts)

    for name in ("config.json", "llm_config.json"):
        source = require_file(export_dir / name)
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    link_or_copy(require_file(export_dir / "llm.mnn.weight"), output_dir / "llm.mnn.weight")
    for name in ("tokenizer.mtok", "tokenizer.txt", "embeddings_bf16.bin"):
        source = export_dir / name
        if source.is_file():
            link_or_copy(source, output_dir / name)

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "llm_model": "decoder/llm.mnn",
            "llm_weight": "llm.mnn.weight",
            "audio_front_model": "audio_encoder_front/audio_encoder_front.mnn",
            "audio_back_model": "audio_encoder_back/audio_encoder_back.mnn",
            "chunk_limits": [64, 1],
            "precision": DECODER_CPU_PRECISION,
            "sampler_type": "greedy",
            "max_all_tokens": max_all_tokens,
            "max_new_tokens": max_new_tokens,
            "async": False,
        }
    )
    config.pop("penalty", None)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    llm_config_path = output_dir / "llm_config.json"
    llm_config = json.loads(llm_config_path.read_text(encoding="utf-8"))
    mismatched = {
        key: {"expected": expected, "actual": llm_config.get(key)}
        for key, expected in MOSS_AUDIO_CONTRACT.items()
        if llm_config.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"exported MOSS audio contract differs: {mismatched}")
    llm_config.update(
        {
            "audio_front_model": "audio_encoder_front/audio_encoder_front.mnn",
            "audio_back_model": "audio_encoder_back/audio_encoder_back.mnn",
        }
    )
    llm_config_path.write_text(json.dumps(llm_config, indent=2) + "\n", encoding="utf-8")

    return {
        "release_id": RELEASE_ID,
        "wrapper_references": wrapper_references,
        "context_counts": context_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-all-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    print(
        json.dumps(
            assemble(
                args.export_dir,
                args.component_root,
                args.output_dir,
                args.max_all_tokens,
                args.max_new_tokens,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
