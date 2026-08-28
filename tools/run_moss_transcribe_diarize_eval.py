#!/usr/bin/env python3
"""Evaluate official MOSS-Transcribe-Diarize output on a MeetNote manifest.

This runner keeps the model's raw generated transcript, parses its native
``[start][Sxx]text[end]`` segments, and reports both text and speaker-aware
metrics.  It is intended for the server-side quality upper bound, not as an
Android runtime implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import time
import traceback
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    Levenshtein = None


DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
DEFAULT_REVISION = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"
SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Remap manifest WAV/expected paths to <root>/<case>/<basename>.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--decoder-gguf",
        type=Path,
        help="Replace the official text decoder with weights dequantized from this GGUF.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-length", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--prompt",
        help="Override the official timestamped-diarization prompt.",
    )
    parser.add_argument("--case-exact", action="append", default=[])
    parser.add_argument("--gain-db", type=float, default=30.0)
    parser.add_argument("--limiter-peak", type=float, default=0.98)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed case JSON only when its inputs and inference settings match.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def replace_text_decoder_from_gguf(
    model, gguf_path: Path, gguf_sha256: str, dtype
) -> dict[str, Any]:
    """Load a Qwen3 GGUF decoder while keeping MOSS audio modules unchanged."""
    from transformers import Qwen3ForCausalLM
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if not gguf_path.is_file():
        raise FileNotFoundError(gguf_path)

    # llama.cpp's bundled editable gguf package reports version "N/A". It is
    # importable and provides the reader/dequantizer, so skip only the broken
    # package-metadata check rather than changing the evaluation environment.
    gguf_utils.is_gguf_available = lambda: True

    text_config = model.config.text_config
    original_language_model = model.model.language_model
    original_lm_head = model.lm_head
    model.model.language_model = None
    model.lm_head = None
    del original_language_model, original_lm_head
    gc.collect()

    decoder = Qwen3ForCausalLM(text_config).to(dtype=dtype)
    checkpoint = gguf_utils.load_gguf_checkpoint(
        str(gguf_path),
        return_tensors=True,
        model_to_load=decoder,
        torch_dtype=dtype,
    )
    incompatible = decoder.load_state_dict(checkpoint["tensors"], strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    allowed_missing = {"lm_head.weight"} if text_config.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"incomplete GGUF decoder load: missing={missing}, unexpected={unexpected}"
        )
    decoder.tie_weights()
    model.model.language_model = decoder.model
    model.lm_head = decoder.lm_head
    del checkpoint, decoder
    gc.collect()
    return {
        "path": str(gguf_path.resolve()),
        "sha256": gguf_sha256,
        "missingKeys": missing,
        "unexpectedKeys": unexpected,
    }


def normalize_text(text: str) -> str:
    return "".join(
        char.lower()
        for char in unicodedata.normalize("NFKC", text)
        if char.isalnum()
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    if Levenshtein is not None:
        return int(Levenshtein.distance(reference, hypothesis))
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, 1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> dict[str, Any]:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    errors = edit_distance(normalized_reference, normalized_hypothesis)
    return {
        "errors": errors,
        "referenceChars": len(normalized_reference),
        "hypothesisChars": len(normalized_hypothesis),
        "cer": errors / len(normalized_reference) if normalized_reference else None,
    }


def minimum_assignment(costs: list[list[int]]) -> tuple[int, list[int]]:
    """Return minimum square assignment cost and row-to-column assignment."""
    size = len(costs)
    if size == 0:
        return 0, []
    if any(len(row) != size for row in costs):
        raise ValueError("assignment matrix must be square")

    row_potential = [0] * (size + 1)
    column_potential = [0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    infinity = sum(max(row, default=0) for row in costs) + 1

    for row_index in range(1, size + 1):
        matched_row[0] = row_index
        current_column = 0
        minimum = [infinity] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = infinity
            next_column = 0
            for column_index in range(1, size + 1):
                if used[column_index]:
                    continue
                reduced = (
                    costs[current_row - 1][column_index - 1]
                    - row_potential[current_row]
                    - column_potential[column_index]
                )
                if reduced < minimum[column_index]:
                    minimum[column_index] = reduced
                    predecessor[column_index] = current_column
                if minimum[column_index] < delta:
                    delta = minimum[column_index]
                    next_column = column_index
            for column_index in range(size + 1):
                if used[column_index]:
                    row_potential[matched_row[column_index]] += delta
                    column_potential[column_index] -= delta
                else:
                    minimum[column_index] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            previous_column = predecessor[current_column]
            matched_row[current_column] = matched_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    assignment = [-1] * size
    for column_index in range(1, size + 1):
        assignment[matched_row[column_index] - 1] = column_index - 1
    return sum(costs[row][column] for row, column in enumerate(assignment)), assignment


def concatenated_permutation_cer(
    reference: list[dict[str, Any]], hypothesis: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_by_speaker: dict[str, list[str]] = defaultdict(list)
    hypothesis_by_speaker: dict[str, list[str]] = defaultdict(list)
    for segment in sorted(reference, key=lambda item: (item["startMs"], item["endMs"])):
        reference_by_speaker[str(segment["speakerRef"])].append(segment.get("text", ""))
    for segment in sorted(hypothesis, key=lambda item: (item["startMs"], item["endMs"])):
        hypothesis_by_speaker[str(segment["speakerId"])].append(segment.get("text", ""))

    reference_labels = sorted(reference_by_speaker)
    hypothesis_labels = sorted(hypothesis_by_speaker)
    size = max(len(reference_labels), len(hypothesis_labels))
    padded_reference = reference_labels + [None] * (size - len(reference_labels))
    padded_hypothesis = hypothesis_labels + [None] * (size - len(hypothesis_labels))
    reference_texts = [
        normalize_text("".join(reference_by_speaker[label])) if label is not None else ""
        for label in padded_reference
    ]
    hypothesis_texts = [
        normalize_text("".join(hypothesis_by_speaker[label])) if label is not None else ""
        for label in padded_hypothesis
    ]
    costs = [
        [edit_distance(reference_text, hypothesis_text) for hypothesis_text in hypothesis_texts]
        for reference_text in reference_texts
    ]
    errors, assignment = minimum_assignment(costs)
    mapping = {
        str(padded_hypothesis[column]): str(padded_reference[row])
        for row, column in enumerate(assignment)
        if padded_reference[row] is not None and padded_hypothesis[column] is not None
    }
    reference_chars = sum(len(text) for text in reference_texts)
    return {
        "errors": errors,
        "referenceChars": reference_chars,
        "cpCer": errors / reference_chars if reference_chars else None,
        "mappingHypToRef": mapping,
        "referenceSpeakerCount": len(reference_labels),
        "hypothesisSpeakerCount": len(hypothesis_labels),
    }


def speaker_attributed_cer(
    reference: list[dict[str, Any]],
    hypothesis: list[dict[str, Any]],
    mapping: dict[str, str],
) -> dict[str, Any]:
    reference_by_speaker: dict[str, list[str]] = defaultdict(list)
    hypothesis_by_speaker: dict[str, list[str]] = defaultdict(list)
    for segment in sorted(reference, key=lambda item: (item["startMs"], item["endMs"])):
        reference_by_speaker[str(segment["speakerRef"])].append(segment.get("text", ""))
    for segment in sorted(hypothesis, key=lambda item: (item["startMs"], item["endMs"])):
        mapped = mapping.get(str(segment["speakerId"]), "__unattributed__")
        hypothesis_by_speaker[mapped].append(segment.get("text", ""))
    speakers = sorted(set(reference_by_speaker) | set(hypothesis_by_speaker))
    per_speaker = {
        speaker: cer(
            "".join(reference_by_speaker[speaker]),
            "".join(hypothesis_by_speaker[speaker]),
        )
        for speaker in speakers
    }
    errors = sum(int(metric["errors"]) for metric in per_speaker.values())
    reference_chars = sum(int(metric["referenceChars"]) for metric in per_speaker.values())
    return {
        "errors": errors,
        "referenceChars": reference_chars,
        "saCer": errors / reference_chars if reference_chars else None,
        "perSpeaker": per_speaker,
    }


def resolve_case_path(case: dict[str, Any], field: str, data_root: Path | None) -> Path:
    source = Path(case[field])
    if data_root is None:
        return source
    return data_root / str(case["case"]) / source.name


def preprocess_audio(
    source_path: Path,
    output_path: Path,
    gain_db: float,
    limiter_peak: float,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[1] != 1:
        raise RuntimeError(
            f"{source_path}: expected mono {SAMPLE_RATE} Hz, got {audio.shape} at {sample_rate}"
        )
    source = audio[:, 0]
    amplified = source * (10 ** (gain_db / 20.0))
    output = limiter_peak * np.tanh(amplified / limiter_peak)
    output = output.astype(np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, output, SAMPLE_RATE, subtype="PCM_16")
    return {
        "sourcePath": str(source_path.resolve()),
        "sourceSha256": sha256(source_path),
        "preparedPath": str(output_path.resolve()),
        "preparedSha256": sha256(output_path),
        "sampleRate": SAMPLE_RATE,
        "sampleCount": len(output),
        "durationSec": len(output) / SAMPLE_RATE,
        "gainDb": gain_db,
        "limiter": "tanh",
        "limiterPeak": limiter_peak,
        "sourcePeak": float(np.max(np.abs(source))) if source.size else 0.0,
        "outputPeak": float(np.max(np.abs(output))) if output.size else 0.0,
    }


def timing_diagnostics(segments: list[dict[str, Any]], duration_ms: int) -> dict[str, Any]:
    return {
        "segmentCount": len(segments),
        "backwardStartPairs": sum(
            current["startMs"] < previous["startMs"]
            for previous, current in zip(segments, segments[1:])
        ),
        "overlappingAdjacentPairs": sum(
            current["startMs"] < previous["endMs"]
            for previous, current in zip(segments, segments[1:])
        ),
        "outOfBoundsSegments": sum(
            item["startMs"] < 0 or item["endMs"] > duration_ms for item in segments
        ),
        "nonPositiveDurationSegments": sum(
            item["endMs"] <= item["startMs"] for item in segments
        ),
    }


def weighted(rows: list[dict[str, Any]], key: str, value: str) -> float | None:
    errors = sum(int(row["metrics"][key]["errors"]) for row in rows)
    reference = sum(int(row["metrics"][key]["referenceChars"]) for row in rows)
    return errors / reference if reference else None


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    device: str,
    dtype: str,
    model_load_sec: float,
    model_allocated_bytes: int | None,
    resumed_case_count: int,
) -> dict[str, Any]:
    return {
        "schemaVersion": "meetnote.moss-transcribe-diarize-summary.v2",
        "manifest": str(args.manifest.resolve()),
        "manifestSha256": sha256(args.manifest),
        "model": args.model,
        "modelRevision": args.revision,
        "decoderGguf": (
            {
                "path": str(args.decoder_gguf.resolve()),
                "sha256": args.decoder_gguf_sha256,
            }
            if args.decoder_gguf is not None
            else None
        ),
        "device": device,
        "dtype": dtype,
        "maxLength": args.max_length,
        "maxNewTokens": args.max_new_tokens,
        "prompt": args.prompt,
        "modelLoadSec": model_load_sec,
        "modelAllocatedBytes": model_allocated_bytes,
        "caseCount": len(rows),
        "resumedCaseCount": resumed_case_count,
        "corpusCer": weighted(rows, "plainCer", "cer"),
        "corpusCpCer": weighted(rows, "cpCer", "cpCer"),
        "corpusTimeMappedSaCer": weighted(rows, "timeMappedSaCer", "saCer"),
        "meanDer": (
            sum(float(row["metrics"]["diarization"]["der"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "meanRtf": (
            sum(float(row["generation"]["rtf"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "perCase": [
            {
                "case": row["case"],
                "cer": row["metrics"]["plainCer"]["cer"],
                "cpCer": row["metrics"]["cpCer"]["cpCer"],
                "timeMappedSaCer": row["metrics"]["timeMappedSaCer"]["saCer"],
                "der": row["metrics"]["diarization"]["der"],
                "predictedSpeakerCount": row["metrics"]["diarization"]["predictedSpeakerCount"],
                "rtf": row["generation"]["rtf"],
                "generatedTokens": row["generation"]["generatedTokens"],
                "hitMaxNewTokens": row["generation"]["hitMaxNewTokens"],
            }
            for row in rows
        ],
    }


def resume_mismatches(
    row: dict[str, Any],
    args: argparse.Namespace,
    case_name: str,
    source_sha256: str,
    expected_sha256: str,
    *,
    device: str,
    dtype: str,
) -> list[str]:
    checks = {
        "case": (row.get("case"), case_name),
        "model": (row.get("model"), args.model),
        "modelRevision": (row.get("modelRevision"), args.revision),
        "decoderGgufSha256": (
            row.get("decoderGguf", {}).get("sha256")
            if row.get("decoderGguf") is not None
            else None,
            args.decoder_gguf_sha256,
        ),
        "device": (row.get("device"), device),
        "dtype": (row.get("dtype"), dtype),
        "maxLength": (row.get("generation", {}).get("maxLength"), args.max_length),
        "maxNewTokens": (
            row.get("generation", {}).get("maxNewTokens"),
            args.max_new_tokens,
        ),
        "prompt": (row.get("generation", {}).get("prompt"), args.prompt),
        "gainDb": (row.get("audio", {}).get("gainDb"), args.gain_db),
        "limiterPeak": (row.get("audio", {}).get("limiterPeak"), args.limiter_peak),
        "sourceSha256": (row.get("audio", {}).get("sourceSha256"), source_sha256),
        "expectedSha256": (row.get("expectedSha256"), expected_sha256),
    }
    return [name for name, (actual, expected) in checks.items() if actual != expected]


def write_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    device: str,
    dtype: str,
    model_load_sec: float,
    model_allocated_bytes: int | None,
    resumed_case_count: int,
) -> dict[str, Any]:
    summary = build_summary(
        args,
        rows,
        device=device,
        dtype=dtype,
        model_load_sec=model_load_sec,
        model_allocated_bytes=model_allocated_bytes,
        resumed_case_count=resumed_case_count,
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    import evaluate_speaker_logs as speaker_eval
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        dtype_from_name,
        generate_transcription,
        resolve_device,
    )

    if args.manifest is None or args.out_dir is None:
        raise SystemExit("--manifest and --out-dir are required")
    if args.max_length <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("--max-length and --max-new-tokens must be positive")
    if not 0 < args.limiter_peak <= 1:
        raise SystemExit("--limiter-peak must be in (0, 1]")
    args.decoder_gguf_sha256 = (
        sha256(args.decoder_gguf) if args.decoder_gguf is not None else None
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = set(args.case_exact)
    cases = [case for case in manifest["cases"] if not selected or case["case"] in selected]
    missing = selected - {str(case["case"]) for case in cases}
    if missing:
        raise SystemExit(f"unknown cases: {sorted(missing)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32
    model_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        dtype="auto",
    ).to(dtype=dtype).to(device).eval()
    decoder_gguf = None
    if args.decoder_gguf is not None:
        if device.type != "cpu":
            raise RuntimeError("--decoder-gguf currently supports CPU evaluation only")
        decoder_gguf = replace_text_decoder_from_gguf(
            model, args.decoder_gguf, args.decoder_gguf_sha256, dtype
        )
        model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    model_load_sec = time.monotonic() - model_started
    model_allocated_bytes = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    rows: list[dict[str, Any]] = []
    resumed_case_count = 0

    for case in cases:
        case_name = str(case["case"])
        source_wav = resolve_case_path(case, "wav", args.data_root)
        expected_path = resolve_case_path(case, "expected", args.data_root)
        source_sha256 = sha256(source_wav)
        expected_sha256 = sha256(expected_path)
        case_path = args.out_dir / f"{case_name}.json"
        if args.resume and case_path.exists():
            resumed = json.loads(case_path.read_text(encoding="utf-8"))
            mismatches = resume_mismatches(
                resumed,
                args,
                case_name,
                source_sha256,
                expected_sha256,
                device=str(device),
                dtype=str(dtype),
            )
            if mismatches:
                raise RuntimeError(
                    f"refusing to resume {case_name}; mismatched fields: {mismatches}"
                )
            rows.append(resumed)
            resumed_case_count += 1
            print(json.dumps({"case": case_name, "status": "resumed"}), flush=True)
            continue

        prepared_wav = args.out_dir / "audio" / f"{case_name}.g{args.gain_db:g}-tanh.wav"
        failure_path = args.out_dir / f"{case_name}.failure.json"
        try:
            audio = preprocess_audio(
                source_wav,
                prepared_wav,
                args.gain_db,
                args.limiter_peak,
            )
            reference = json.loads(expected_path.read_text(encoding="utf-8"))["segments"]
            reference_text = "".join(
                item.get("text", "")
                for item in sorted(reference, key=lambda item: (item["startMs"], item["endMs"]))
            )

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.monotonic()
            messages = (
                build_transcription_messages(prepared_wav, prompt=args.prompt)
                if args.prompt is not None
                else build_transcription_messages(prepared_wav)
            )
            generation = generate_transcription(
                model,
                processor,
                messages,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                device=device,
                dtype=dtype,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_sec = time.monotonic() - started
            parsed = parse_transcript(generation["text"])
            segments = [
                {
                    "id": f"moss-{index}",
                    "startMs": round(item.start * 1000),
                    "endMs": round(item.end * 1000),
                    "speakerId": item.speaker,
                    "text": item.text,
                }
                for index, item in enumerate(parsed)
            ]
            hypothesis_text = "".join(item["text"] for item in segments)
            plain = cer(reference_text, hypothesis_text)
            cp_metric = concatenated_permutation_cer(reference, segments)
            diarization = speaker_eval.evaluate_segments(reference, segments)
            sa_metric = speaker_attributed_cer(
                reference,
                segments,
                diarization["mappingHypToRef"],
            )
            duration_sec = float(case["durationSec"])
            row = {
                "schemaVersion": "meetnote.moss-transcribe-diarize-case.v1",
                "case": case_name,
                "durationSec": duration_sec,
                "model": args.model,
                "modelRevision": args.revision,
                "decoderGguf": decoder_gguf,
                "device": str(device),
                "dtype": str(dtype),
                "expectedSha256": expected_sha256,
                "audio": audio,
                "generation": {
                    "rawText": generation["text"],
                    "promptLen": int(generation["prompt_len"]),
                    "generatedTokens": int(generation["generated_tokens"]),
                    "maxLength": args.max_length,
                    "maxNewTokens": args.max_new_tokens,
                    "prompt": args.prompt,
                    "hitMaxNewTokens": (
                        int(generation["generated_tokens"]) >= args.max_new_tokens
                    ),
                    "inferenceSec": inference_sec,
                    "rtf": inference_sec / duration_sec,
                    "processMaxRssKiB": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "cudaPeakAllocatedBytes": (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else None
                    ),
                    "cudaPeakReservedBytes": (
                        torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else None
                    ),
                },
                "timing": timing_diagnostics(segments, round(duration_sec * 1000)),
                "segments": segments,
                "referenceText": reference_text,
                "hypothesisText": hypothesis_text,
                "metrics": {
                    "plainCer": plain,
                    "cpCer": cp_metric,
                    "deltaCp": (
                        cp_metric["cpCer"] - plain["cer"]
                        if cp_metric["cpCer"] is not None and plain["cer"] is not None
                        else None
                    ),
                    "timeMappedSaCer": sa_metric,
                    "diarization": diarization,
                },
            }
        except Exception as exc:
            failure = {
                "schemaVersion": "meetnote.moss-transcribe-diarize-failure.v1",
                "case": case_name,
                "sourcePath": str(source_wav.resolve()),
                "sourceSha256": source_sha256,
                "expectedPath": str(expected_path.resolve()),
                "expectedSha256": expected_sha256,
                "model": args.model,
                "modelRevision": args.revision,
                "decoderGguf": decoder_gguf,
                "device": str(device),
                "dtype": str(dtype),
                "maxLength": args.max_length,
                "maxNewTokens": args.max_new_tokens,
                "errorType": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failure_path.write_text(
                json.dumps(failure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise
        failure_path.unlink(missing_ok=True)
        rows.append(row)
        case_path.write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_summary(
            args,
            rows,
            device=str(device),
            dtype=str(dtype),
            model_load_sec=model_load_sec,
            model_allocated_bytes=model_allocated_bytes,
            resumed_case_count=resumed_case_count,
        )
        print(
            json.dumps(
                {
                    "case": case_name,
                    "cer": plain["cer"],
                    "cpCer": cp_metric["cpCer"],
                    "saCer": sa_metric["saCer"],
                    "der": diarization["der"],
                    "speakers": diarization["predictedSpeakerCount"],
                    "rtf": inference_sec / duration_sec,
                    "tokens": int(generation["generated_tokens"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = write_summary(
        args,
        rows,
        device=str(device),
        dtype=str(dtype),
        model_load_sec=model_load_sec,
        model_allocated_bytes=model_allocated_bytes,
        resumed_case_count=resumed_case_count,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def self_test() -> None:
    costs = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    cost, assignment = minimum_assignment(costs)
    assert cost == 5, (cost, assignment)

    reference = [
        {"speakerRef": "A", "startMs": 0, "endMs": 1000, "text": "甲乙"},
        {"speakerRef": "B", "startMs": 1000, "endMs": 2000, "text": "丙丁"},
    ]
    swapped = [
        {"speakerId": "X", "startMs": 0, "endMs": 1000, "text": "丙丁"},
        {"speakerId": "Y", "startMs": 1000, "endMs": 2000, "text": "甲乙"},
    ]
    metric = concatenated_permutation_cer(reference, swapped)
    assert metric["cpCer"] == 0.0, metric
    assert metric["mappingHypToRef"] == {"Y": "A", "X": "B"}, metric

    extra = swapped + [
        {"speakerId": "Z", "startMs": 2000, "endMs": 3000, "text": "多余"}
    ]
    metric = concatenated_permutation_cer(reference, extra)
    assert metric["errors"] == 2, metric
    assert math.isclose(metric["cpCer"], 0.5), metric

    resume_args = argparse.Namespace(
        model="model",
        revision="revision",
        decoder_gguf=None,
        decoder_gguf_sha256=None,
        max_length=1024,
        max_new_tokens=512,
        prompt=None,
        gain_db=30.0,
        limiter_peak=0.98,
    )
    resumable = {
        "case": "case",
        "model": "model",
        "modelRevision": "revision",
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "expectedSha256": "expected",
        "audio": {
            "sourceSha256": "source",
            "gainDb": 30.0,
            "limiterPeak": 0.98,
        },
        "generation": {"maxLength": 1024, "maxNewTokens": 512, "prompt": None},
    }
    assert not resume_mismatches(
        resumable,
        resume_args,
        "case",
        "source",
        "expected",
        device="cuda:0",
        dtype="torch.bfloat16",
    )
    assert resume_mismatches(
        resumable,
        resume_args,
        "case",
        "source",
        "expected",
        device="cuda:0",
        dtype="torch.float16",
    ) == ["dtype"]
    print("run_moss_transcribe_diarize_eval.py self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
