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
    parser.add_argument("--mnn-graph-probe", type=Path)
    parser.add_argument("--mnn-threads", type=int, default=8)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    return parser.parse_args()


def run_graph_probe(
    executable: Path,
    model: Path,
    input_name: str,
    output_name: str,
    input_value: np.ndarray,
    output_shape: tuple[int, ...],
    threads: int,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="moss-mnn-graph-") as temp_dir:
        input_path = Path(temp_dir) / "input.bin"
        output_path = Path(temp_dir) / "output.bin"
        contiguous = np.ascontiguousarray(input_value, dtype=np.float32)
        contiguous.tofile(input_path)
        subprocess.run(
            [
                str(executable),
                str(model),
                input_name,
                output_name,
                ",".join(str(value) for value in contiguous.shape),
                str(input_path),
                str(output_path),
                str(threads),
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        output = np.fromfile(output_path, dtype=np.float32)
    expected = int(np.prod(output_shape))
    if output.size != expected:
        raise ValueError(f"MNN graph output size mismatch: expected {expected}, got {output.size}")
    return output.reshape(output_shape)


def main() -> None:
    args = parse_args()
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if args.mnn_threads <= 0:
        raise ValueError("--mnn-threads must be positive")
    if args.mnn_graph_probe is None:
        import MNN.nn as mnn_nn
        import MNN.numpy as mnn_np

    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    if sample_rate != 16_000 or audio.shape[1] != 1:
        raise ValueError("alignment WAV must be 16 kHz mono")
    chunk_samples = 30 * sample_rate
    chunks = []
    chunk_ranges = []
    for start in range(0, audio.shape[0], chunk_samples):
        chunk = audio[start : start + chunk_samples, 0]
        valid_samples = int(chunk.shape[0])
        if valid_samples < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - valid_samples))
        chunks.append(chunk)
        chunk_ranges.append((start, start + valid_samples))
    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    input_features = processor.feature_extractor(
        chunks, sampling_rate=sample_rate, padding="max_length", return_tensors="pt"
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

    front_model = args.mnn_dir / "audio_encoder_front.mnn"
    back_model = args.mnn_dir / "audio_encoder_back.mnn"
    if args.mnn_graph_probe is None:
        front = mnn_nn.load_module_from_file(
            str(front_model), ["input_features"], ["hidden_states"], dynamic=False, shape_mutable=False
        )
        back = mnn_nn.load_module_from_file(
            str(back_model), ["hidden_states"], ["audio_embeds"], dynamic=False, shape_mutable=False
        )

    mnn_front_chunks = []
    mnn_back_isolated_chunks = []
    mnn_back_chain_chunks = []
    cpu_feature_chunks = []
    cpu_embedding_chunks = []
    cases = []
    for index, (start, end) in enumerate(chunk_ranges):
        feature = input_features[index : index + 1].numpy()
        hf_front_chunk = hf_front[index : index + 1]
        hf_back_chunk = hf_back[index : index + 1]
        if args.mnn_graph_probe is not None:
            mnn_front_chunk = run_graph_probe(
                args.mnn_graph_probe,
                front_model,
                "input_features",
                "hidden_states",
                feature,
                tuple(hf_front_chunk.shape),
                args.mnn_threads,
            )
            mnn_back_isolated_chunk = run_graph_probe(
                args.mnn_graph_probe,
                back_model,
                "hidden_states",
                "audio_embeds",
                hf_front_chunk,
                tuple(hf_back_chunk.shape),
                args.mnn_threads,
            )
            mnn_back_chain_chunk = run_graph_probe(
                args.mnn_graph_probe,
                back_model,
                "hidden_states",
                "audio_embeds",
                mnn_front_chunk,
                tuple(hf_back_chunk.shape),
                args.mnn_threads,
            )
        else:
            mnn_front_chunk = np.asarray(front.forward(mnn_np.array(feature)).read()).copy()
            mnn_back_isolated_chunk = np.asarray(back.forward(mnn_np.array(hf_front_chunk)).read()).copy()
            mnn_back_chain_chunk = np.asarray(back.forward(mnn_np.array(mnn_front_chunk)).read()).copy()
        mnn_front_chunks.append(mnn_front_chunk)
        mnn_back_isolated_chunks.append(mnn_back_isolated_chunk)
        mnn_back_chain_chunks.append(mnn_back_chain_chunk)
        case = {
            "chunk_index": index,
            "sample_start": start,
            "sample_end": end,
            "front": statistics(hf_front_chunk, mnn_front_chunk),
            "back_isolated": statistics(hf_back_chunk, mnn_back_isolated_chunk),
            "end_to_end": statistics(hf_back_chunk, mnn_back_chain_chunk),
        }
        if args.mnn_fbank_probe is not None:
            with tempfile.TemporaryDirectory(prefix="moss-fbank-") as temp_dir:
                feature_path = Path(temp_dir) / "features.bin"
                subprocess.run(
                    [str(args.mnn_fbank_probe), str(args.wav), str(feature_path), str(start)],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                cpu_features = np.fromfile(feature_path, dtype=np.float32).reshape(feature.shape)
            cpu_feature_chunks.append(cpu_features)
            cpu_front = (
                run_graph_probe(
                    args.mnn_graph_probe,
                    front_model,
                    "input_features",
                    "hidden_states",
                    cpu_features,
                    tuple(hf_front_chunk.shape),
                    args.mnn_threads,
                )
                if args.mnn_graph_probe is not None
                else np.asarray(front.forward(mnn_np.array(cpu_features)).read()).copy()
            )
            cpu_embedding = (
                run_graph_probe(
                    args.mnn_graph_probe,
                    back_model,
                    "hidden_states",
                    "audio_embeds",
                    cpu_front,
                    tuple(hf_back_chunk.shape),
                    args.mnn_threads,
                )
                if args.mnn_graph_probe is not None
                else np.asarray(back.forward(mnn_np.array(cpu_front)).read()).copy()
            )
            cpu_embedding_chunks.append(cpu_embedding)
            case["log_mel"] = statistics(feature, cpu_features) | {
                "shape_exact": True,
                "all_finite": bool(np.isfinite(cpu_features).all()),
            }
            case["cpu_frontend_end_to_end"] = statistics(hf_back_chunk, cpu_embedding)
        cases.append(case)

    mnn_front = np.concatenate(mnn_front_chunks)
    mnn_back_isolated = np.concatenate(mnn_back_isolated_chunks)
    mnn_back_chain = np.concatenate(mnn_back_chain_chunks)

    result = {
        "format": "meetnote.moss_audio_mnn_alignment.v2",
        "scope": "HF FP32 to MNN CPU using identical official input_features",
        "chunk_count": len(cases),
        "cases": cases,
        "front": statistics(hf_front, mnn_front),
        "back_isolated": statistics(hf_back, mnn_back_isolated),
        "end_to_end": statistics(hf_back, mnn_back_chain),
    }
    cosines = [result[name]["cosine"] for name in ("front", "back_isolated", "end_to_end")]
    if args.mnn_fbank_probe is not None:
        cpu_features = np.concatenate(cpu_feature_chunks)
        log_mel = statistics(input_features.numpy(), cpu_features)
        log_mel.update(
            {
                "shape_exact": True,
                "all_finite": bool(np.isfinite(cpu_features).all()),
            }
        )
        result["scope"] = "HF FP32 versus MNN CPU graph, including the actual MNN whisper_fbank frontend"
        result["log_mel"] = log_mel
        result["cpu_frontend_end_to_end"] = statistics(
            hf_back, np.concatenate(cpu_embedding_chunks)
        )
        cosines.append(result["cpu_frontend_end_to_end"]["cosine"])

    log_mel_valid = "log_mel" not in result or (
        result["log_mel"]["all_finite"]
        and result["log_mel"]["cosine"] >= 0.9999
        and result["log_mel"]["mean_abs_error"] <= 1e-3
    )
    case_cosines = [
        case[name]["cosine"]
        for case in cases
        for name in ("front", "back_isolated", "end_to_end")
    ]
    if args.mnn_fbank_probe is not None:
        case_cosines.extend(case["cpu_frontend_end_to_end"]["cosine"] for case in cases)
        log_mel_valid = log_mel_valid and all(
            case["log_mel"]["all_finite"]
            and case["log_mel"]["cosine"] >= 0.9999
            and case["log_mel"]["mean_abs_error"] <= 1e-3
            for case in cases
        )
    result["status"] = "valid" if min(cosines + case_cosines) >= args.min_cosine and log_mel_valid else "invalid"
    print(json.dumps(result, indent=2))
    if result["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
