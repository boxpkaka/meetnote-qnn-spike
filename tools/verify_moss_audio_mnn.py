#!/usr/bin/env python3
"""Compare exported MOSS audio graphs with HF using identical input features."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.reshape(-1).astype(np.float64)
    right64 = right.reshape(-1).astype(np.float64)
    return float(
        np.dot(left64, right64) / (np.linalg.norm(left64) * np.linalg.norm(right64))
    )


def statistics(reference: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: HF={reference.shape} MNN={actual.shape}")
    difference = np.abs(reference - actual)
    return {
        "shape": list(reference.shape),
        "cosine": cosine(reference, actual),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--mnn-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--mnn-fbank-probe", type=Path)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import MNN.nn as mnn_nn
    import MNN.numpy as mnn_np
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    if sample_rate != 16_000 or audio.shape[1] != 1:
        raise ValueError("alignment WAV must be 16 kHz mono")
    chunk = audio[: 30 * sample_rate, 0]
    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    input_features = processor.feature_extractor(
        [chunk], sampling_rate=sample_rate, padding="max_length", return_tensors="pt"
    )["input_features"].float()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    encoder = model.model.whisper_encoder
    with torch.inference_mode():
        hidden = torch.nn.functional.gelu(encoder.conv1(input_features))
        hidden = torch.nn.functional.gelu(encoder.conv2(hidden)).permute(0, 2, 1)
        hidden = hidden + encoder.embed_positions.weight
        for layer in encoder.layers[:12]:
            hidden = layer(hidden, None)
        hf_front = hidden.cpu().numpy()
        for layer in encoder.layers[12:]:
            hidden = layer(hidden, None)
        hidden = encoder.layer_norm(hidden)
        batch, frames, width = hidden.shape
        hidden = hidden[:, : frames // 4 * 4].reshape(batch, frames // 4, width * 4)
        hf_back = model.model.vq_adaptor(hidden).cpu().numpy()

    front = mnn_nn.load_module_from_file(
        str(args.mnn_dir / "audio_encoder_front.mnn"),
        ["input_features"],
        ["hidden_states"],
        dynamic=False,
        shape_mutable=False,
    )
    back = mnn_nn.load_module_from_file(
        str(args.mnn_dir / "audio_encoder_back.mnn"),
        ["hidden_states"],
        ["audio_embeds"],
        dynamic=False,
        shape_mutable=False,
    )
    mnn_front = np.asarray(front.forward(mnn_np.array(input_features.numpy())).read()).copy()
    mnn_back_isolated = np.asarray(back.forward(mnn_np.array(hf_front)).read()).copy()
    mnn_back_chain = np.asarray(back.forward(mnn_np.array(mnn_front)).read()).copy()

    result = {
        "format": "meetnote.moss_audio_mnn_alignment.v1",
        "scope": "HF FP32 to MNN CPU using identical official input_features",
        "front": statistics(hf_front, mnn_front),
        "back_isolated": statistics(hf_back, mnn_back_isolated),
        "end_to_end": statistics(hf_back, mnn_back_chain),
    }
    cosines = [result[name]["cosine"] for name in ("front", "back_isolated", "end_to_end")]
    if args.mnn_fbank_probe is not None:
        with tempfile.TemporaryDirectory(prefix="moss-fbank-") as temp_dir:
            feature_path = Path(temp_dir) / "features.bin"
            subprocess.run(
                [str(args.mnn_fbank_probe), str(args.wav), str(feature_path)], check=True
            )
            cpu_features = np.fromfile(feature_path, dtype=np.float32)
        expected_shape = tuple(input_features.shape)
        if cpu_features.size != int(np.prod(expected_shape)):
            raise ValueError(
                f"MNN fbank size mismatch: expected {np.prod(expected_shape)}, got {cpu_features.size}"
            )
        cpu_features = cpu_features.reshape(expected_shape)
        log_mel = statistics(input_features.numpy(), cpu_features)
        log_mel.update(
            {
                "shape_exact": True,
                "all_finite": bool(np.isfinite(cpu_features).all()),
            }
        )
        cpu_front = np.asarray(front.forward(mnn_np.array(cpu_features)).read()).copy()
        cpu_embedding = np.asarray(back.forward(mnn_np.array(cpu_front)).read()).copy()
        result["scope"] = "HF FP32 versus MNN CPU graph, including the actual MNN whisper_fbank frontend"
        result["log_mel"] = log_mel
        result["cpu_frontend_end_to_end"] = statistics(hf_back, cpu_embedding)
        cosines.append(result["cpu_frontend_end_to_end"]["cosine"])

    log_mel_valid = "log_mel" not in result or (
        result["log_mel"]["all_finite"]
        and result["log_mel"]["cosine"] >= 0.9999
        and result["log_mel"]["mean_abs_error"] <= 1e-3
    )
    result["status"] = "valid" if min(cosines) >= args.min_cosine and log_mel_valid else "invalid"
    print(json.dumps(result, indent=2))
    if result["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
