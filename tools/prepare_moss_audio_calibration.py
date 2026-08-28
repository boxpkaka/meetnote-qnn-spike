#!/usr/bin/env python3
"""Prepare representative MOSS audio encoder calibration tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunks-per-wav", type=int, default=2)
    parser.add_argument("wav", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunks_per_wav <= 0:
        raise ValueError("--chunks-per-wav must be positive")

    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    output = args.output_dir.resolve()
    front_dir = output / "front"
    back_dir = output / "back"
    reference_dir = output / "reference"
    for directory in (front_dir, back_dir, reference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    encoder = model.model.whisper_encoder

    manifest: list[dict[str, object]] = []
    front_lines: list[str] = []
    back_lines: list[str] = []
    sample_index = 0
    chunk_samples = 30 * 16_000
    for wav_path in args.wav:
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        if sample_rate != 16_000 or audio.shape[1] != 1:
            raise ValueError(f"expected 16 kHz mono WAV: {wav_path}")
        chunk_count = max(1, (audio.shape[0] + chunk_samples - 1) // chunk_samples)
        if args.chunks_per_wav == 1:
            selected = [0]
        else:
            selected = np.linspace(
                0, chunk_count - 1, min(args.chunks_per_wav, chunk_count), dtype=int
            ).tolist()
        for chunk_index in selected:
            start = chunk_index * chunk_samples
            chunk = audio[start : start + chunk_samples, 0]
            if chunk.shape[0] < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - chunk.shape[0]))
            features = processor.feature_extractor(
                chunk,
                sampling_rate=sample_rate,
                padding="max_length",
                return_tensors="pt",
            )["input_features"].float()
            with torch.inference_mode():
                hidden = torch.nn.functional.gelu(encoder.conv1(features))
                hidden = torch.nn.functional.gelu(encoder.conv2(hidden)).permute(0, 2, 1)
                hidden = hidden + encoder.embed_positions.weight
                for layer in encoder.layers[:12]:
                    hidden = layer(hidden, None)
                front_output = hidden
                for layer in encoder.layers[12:]:
                    hidden = layer(hidden, None)
                hidden = encoder.layer_norm(hidden)
                batch, frames, width = hidden.shape
                hidden = hidden[:, : frames // 4 * 4].reshape(
                    batch, frames // 4, width * 4
                )
                back_output = model.model.vq_adaptor(hidden)

            stem = f"sample-{sample_index:03d}"
            front_input_path = (front_dir / f"{stem}.f32").resolve()
            back_input_path = (back_dir / f"{stem}.f32").resolve()
            front_reference_path = reference_dir / f"{stem}-front.f32"
            back_reference_path = reference_dir / f"{stem}-back.f32"
            np.ascontiguousarray(features.numpy(), dtype=np.float32).tofile(front_input_path)
            np.ascontiguousarray(front_output.numpy(), dtype=np.float32).tofile(back_input_path)
            np.ascontiguousarray(front_output.numpy(), dtype=np.float32).tofile(
                front_reference_path
            )
            np.ascontiguousarray(back_output.numpy(), dtype=np.float32).tofile(
                back_reference_path
            )
            front_lines.append(f"input_features:={front_input_path}")
            back_lines.append(f"hidden_states:={back_input_path}")
            manifest.append(
                {
                    "sample": stem,
                    "wav": str(wav_path.resolve()),
                    "chunk_index": int(chunk_index),
                    "front_shape": list(features.shape),
                    "hidden_shape": list(front_output.shape),
                    "back_shape": list(back_output.shape),
                    "feature_min": float(features.min()),
                    "feature_max": float(features.max()),
                    "hidden_min": float(front_output.min()),
                    "hidden_max": float(front_output.max()),
                }
            )
            sample_index += 1

    (front_dir / "input-list.txt").write_text("\n".join(front_lines) + "\n")
    (back_dir / "input-list.txt").write_text("\n".join(back_lines) + "\n")
    (output / "manifest.json").write_text(
        json.dumps({"samples": manifest}, ensure_ascii=True, indent=2) + "\n"
    )
    print(json.dumps({"sample_count": sample_index, "output_dir": str(output)}))


if __name__ == "__main__":
    main()
