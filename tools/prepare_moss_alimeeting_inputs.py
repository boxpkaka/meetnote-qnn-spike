#!/usr/bin/env python3
"""Prepare fixed AliMeeting channel-0 clips for the MOSS SM8850 PoC."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import struct
import wave
from pathlib import Path
from typing import BinaryIO

SAMPLE_RATE = 16_000
CALIBRATION_CASES = 40
CALIBRATION_SECONDS = 120
VALIDATION_CASES = 10
VALIDATION_SECONDS = 300


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def riff_chunks(source: BinaryIO):
    if source.read(12)[:4] != b"RIFF":
        raise ValueError("input is not a RIFF WAV")
    while header := source.read(8):
        chunk_id, size = struct.unpack("<4sI", header)
        offset = source.tell()
        yield chunk_id, offset, size
        source.seek(offset + size + (size & 1))


def wav_layout(path: Path) -> tuple[int, int, int, int, int]:
    fmt = None
    data = None
    with path.open("rb") as source:
        for chunk_id, offset, size in riff_chunks(source):
            if chunk_id == b"fmt ":
                source.seek(offset)
                fmt = source.read(size)
            elif chunk_id == b"data":
                data = (offset, size)
    if fmt is None or data is None or len(fmt) < 16:
        raise ValueError(f"missing WAV fmt/data chunk: {path}")
    format_tag, channels, rate, _, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if format_tag == 0xFFFE and len(fmt) >= 40:
        format_tag = struct.unpack("<H", fmt[24:26])[0]
    if format_tag != 1 or rate != SAMPLE_RATE or bits != 16:
        raise ValueError(f"expected 16 kHz signed PCM16 WAV: {path}")
    if block_align != channels * 2 or channels < 1:
        raise ValueError(f"invalid WAV channel layout: {path}")
    data_offset, data_size = data
    return channels, block_align, data_offset, data_size, data_size // block_align


def write_channel_zero_clip(source_path: Path, output: Path, start: int, frames: int) -> int:
    channels, block_align, data_offset, _, available = wav_layout(source_path)
    frames = min(frames, max(0, available - start))
    if frames <= 0:
        raise ValueError(f"clip starts after WAV end: {source_path}")
    with source_path.open("rb") as source:
        source.seek(data_offset + start * block_align)
        interleaved = array.array("h")
        interleaved.frombytes(source.read(frames * block_align))
    if struct.pack("=H", 1) == struct.pack(">H", 1):
        interleaved.byteswap()
    mono = interleaved[0::channels]
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as target:
        target.setparams((1, 2, SAMPLE_RATE, len(mono), "NONE", "not compressed"))
        target.writeframes(mono.tobytes())
    return len(mono)


def load_records(path: Path, split: str) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("dataset") == "AliMeeting" and record.get("split") == split:
                records.append(record)
    return sorted(records, key=lambda row: row["meeting_id"])


def transcript_for_clip(record: dict, start_ms: int, end_ms: int) -> str:
    segments = []
    for utterance in record["utterances"]:
        start = max(start_ms, int(utterance["start_ms"]))
        end = min(end_ms, int(utterance["end_ms"]))
        if end <= start:
            continue
        speaker = f"S{int(str(utterance['speaker'])[1:]):02d}"
        segments.append(
            f"[{(start - start_ms) / 1000:.2f}]"
            f"[{speaker}]{utterance['text']}"
            f"[{(end - start_ms) / 1000:.2f}]"
        )
    return "\n".join(segments)


def source_wav(record: dict, data_root: Path) -> Path:
    paths = record.get("media", {}).get("far", [])
    if not paths:
        raise ValueError(f"missing far-field WAV: {record['meeting_id']}")
    return data_root / paths[0]


def make_case(
    record: dict, data_root: Path, output_dir: Path, start_seconds: int, duration_seconds: int
) -> dict:
    case_id = f"{record['meeting_id']}_{start_seconds:04d}_{duration_seconds}s"
    output = output_dir / f"{case_id}.wav"
    frames = write_channel_zero_clip(
        source_wav(record, data_root),
        output,
        start_seconds * SAMPLE_RATE,
        duration_seconds * SAMPLE_RATE,
    )
    end_ms = start_seconds * 1000 + round(frames * 1000 / SAMPLE_RATE)
    transcript = transcript_for_clip(record, start_seconds * 1000, end_ms)
    if not transcript:
        raise ValueError(f"clip has no reference transcript: {case_id}")
    return {
        "id": case_id,
        "meeting_id": record["meeting_id"],
        "split": record["split"],
        "source_start_ms": start_seconds * 1000,
        "source_end_ms": end_ms,
        "wav": str(output.resolve()),
        "sha256": sha256(output),
        "transcript": transcript,
    }


def write_manifest(path: Path, format_name: str, cases: list[dict]) -> None:
    path.write_text(
        json.dumps({"format": format_name, "cases": cases}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/data"))
    parser.add_argument(
        "--canonical-dir", type=Path, default=Path("/data/data/MeetNoteExperiments/v1")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    train = load_records(args.canonical_dir / "train.jsonl", "train")
    validation = load_records(args.canonical_dir / "validation.jsonl", "validation")
    if len(train) < CALIBRATION_CASES or len(validation) < 2:
        parser.error("AliMeeting canonical split does not contain enough meetings")

    calibration = [
        make_case(record, args.data_root, args.output_dir / "calibration", 0, CALIBRATION_SECONDS)
        for record in train[:CALIBRATION_CASES]
    ]
    validation_choices = [(record, 0) for record in validation]
    validation_choices.extend((record, VALIDATION_SECONDS) for record in validation[:2])
    acceptance = [
        make_case(record, args.data_root, args.output_dir / "validation", start, VALIDATION_SECONDS)
        for record, start in validation_choices[:VALIDATION_CASES]
    ]
    write_manifest(
        args.output_dir / "calibration-inputs.json",
        "meetnote.moss_calibration_inputs.v1",
        calibration,
    )
    write_manifest(
        args.output_dir / "validation-inputs.json",
        "meetnote.moss_acceptance_inputs.v1",
        acceptance,
    )
    print(json.dumps({"calibration_cases": len(calibration), "validation_cases": len(acceptance)}))


if __name__ == "__main__":
    main()
