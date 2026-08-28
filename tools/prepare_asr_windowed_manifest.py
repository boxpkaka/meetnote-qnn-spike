#!/usr/bin/env python3
"""Split full ASR evaluation meetings into reference-safe parallel windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def crossing(segments: list[dict[str, Any]], point_ms: int) -> list[dict[str, Any]]:
    return [
        segment
        for segment in segments
        if int(segment["startMs"]) < point_ms < int(segment["endMs"])
    ]


def safe_boundary(segments: list[dict[str, Any]], target_ms: int, lower_ms: int) -> int:
    cutoff = target_ms
    while active := crossing(segments, cutoff):
        cutoff = min(int(segment["startMs"]) for segment in active)
    if cutoff > lower_ms:
        return cutoff
    cutoff = target_ms
    while active := crossing(segments, cutoff):
        cutoff = max(int(segment["endMs"]) for segment in active)
    return cutoff


def boundaries(
    segments: list[dict[str, Any]], duration_ms: int, window_ms: int
) -> list[int]:
    result = [0]
    while duration_ms - result[-1] > window_ms:
        point = safe_boundary(segments, result[-1] + window_ms, result[-1])
        if point <= result[-1] or point >= duration_ms:
            break
        result.append(point)
    result.append(duration_ms)
    if len(result) > 2 and result[-1] - result[-2] < 30_000:
        del result[-2]
    for point in result[1:-1]:
        if crossing(segments, point):
            raise RuntimeError(f"unsafe reference boundary: {point}")
    return result


def fixed_boundaries(duration_ms: int, window_ms: int) -> list[int]:
    return list(range(0, duration_ms, window_ms)) + [duration_ms]


def balanced_fixed_boundaries(duration_ms: int, window_ms: int) -> list[int]:
    window_count = max(1, math.ceil(duration_ms / window_ms))
    return [round(index * duration_ms / window_count) for index in range(window_count + 1)]


def crop_wav(source: Path, destination: Path, start_ms: int, end_ms: int) -> float:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with wave.open(str(source), "rb") as inp:
        if inp.getnchannels() != 1 or inp.getsampwidth() != 2:
            raise RuntimeError(f"{source}: expected mono 16-bit PCM")
        sample_rate = inp.getframerate()
        start_frame = round(start_ms * sample_rate / 1000)
        end_frame = min(inp.getnframes(), round(end_ms * sample_rate / 1000))
        inp.setpos(start_frame)
        with wave.open(str(temporary), "wb") as out:
            out.setparams((1, 2, sample_rate, end_frame - start_frame, "NONE", "not compressed"))
            remaining = end_frame - start_frame
            while remaining:
                chunk = inp.readframes(min(remaining, 65_536))
                if not chunk:
                    break
                out.writeframesraw(chunk)
                remaining -= len(chunk) // 2
    temporary.replace(destination)
    return (end_frame - start_frame) / sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--window-sec", type=float, default=180.0)
    parser.add_argument(
        "--boundary-mode",
        choices=("reference-safe", "fixed", "balanced-fixed"),
        default="reference-safe",
    )
    parser.add_argument("--case-exact", action="append", default=[])
    args = parser.parse_args()
    if args.window_sec <= 0:
        raise SystemExit("--window-sec must be positive")

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    selected = set(args.case_exact)
    source_cases = [
        case for case in source["cases"] if not selected or str(case["case"]) in selected
    ]
    missing = selected - {str(case["case"]) for case in source_cases}
    if missing:
        raise SystemExit(f"unknown cases: {sorted(missing)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_cases: list[dict[str, Any]] = []
    for source_case in source_cases:
        expected_payload = json.loads(Path(source_case["expected"]).read_text(encoding="utf-8"))
        reference = expected_payload["segments"]
        duration_ms = round(float(source_case["durationSec"]) * 1000)
        window_ms = round(args.window_sec * 1000)
        points = (
            boundaries(reference, duration_ms, window_ms)
            if args.boundary_mode == "reference-safe"
            else (
                fixed_boundaries(duration_ms, window_ms)
                if args.boundary_mode == "fixed"
                else balanced_fixed_boundaries(duration_ms, window_ms)
            )
        )
        for index, (start_ms, end_ms) in enumerate(zip(points, points[1:])):
            name = f"{source_case['case']}_w{index:03d}"
            case_dir = args.out_dir / name
            case_dir.mkdir(parents=True, exist_ok=True)
            wav = case_dir / f"{name}.wav"
            actual_duration = crop_wav(Path(source_case["wav"]), wav, start_ms, end_ms)
            if args.boundary_mode == "reference-safe":
                selected_segments = [
                    segment
                    for segment in reference
                    if int(segment["startMs"]) >= start_ms
                    and int(segment["endMs"]) <= end_ms
                ]
            else:
                # A crossing utterance is assigned once, by midpoint. Its full text is
                # retained so stitched reference accounting remains exact.
                selected_segments = [
                    segment
                    for segment in reference
                    if start_ms
                    <= (int(segment["startMs"]) + int(segment["endMs"])) // 2
                    < end_ms
                ]
            local_segments = [
                {
                    **segment,
                    "startMs": max(start_ms, int(segment["startMs"])) - start_ms,
                    "endMs": min(end_ms, int(segment["endMs"])) - start_ms,
                }
                for segment in selected_segments
            ]
            expected = case_dir / f"{name}.expected.json"
            expected.write_text(
                json.dumps(
                    {
                        "schemaVersion": "meetnote.asr.reference-window.v1",
                        "case": name,
                        "sourceCase": source_case["case"],
                        "sourceStartMs": start_ms,
                        "durationMs": round(actual_duration * 1000),
                        "segments": local_segments,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            speakers = sorted({segment["speakerRef"] for segment in local_segments})
            output_cases.append(
                {
                    "case": name,
                    "sourceCase": source_case["case"],
                    "sourceStartMs": start_ms,
                    "sourceEndMs": end_ms,
                    "wav": str(wav.resolve()),
                    "expected": str(expected.resolve()),
                    "durationSec": actual_duration,
                    "expectedSpeakerCount": len(speakers),
                    "expectedTurnCount": len(local_segments),
                    "referenceSpeakers": speakers,
                    "referenceCharsRaw": sum(len(segment.get("text", "")) for segment in local_segments),
                    "wavSha256": sha256(wav),
                    "referenceSha256": sha256(expected),
                }
            )

    manifest = {
        "schemaVersion": "meetnote.asr.windowed-manifest.v1",
        "dataset": source.get("dataset", "AliMeeting"),
        "split": f"{source.get('split', 'validation')}-{args.boundary_mode}-windows",
        "selection": {
            "policy": (
                "contiguous windows with boundaries outside every reference utterance"
                if args.boundary_mode == "reference-safe"
                else (
                    "contiguous fixed-duration windows independent of reference boundaries"
                    if args.boundary_mode == "fixed"
                    else "contiguous duration-balanced windows independent of reference boundaries"
                )
            ),
            "boundaryMode": args.boundary_mode,
            "requestedWindowSec": args.window_sec,
            "sourceManifest": str(args.source_manifest.resolve()),
            "sourceManifestSha256": sha256(args.source_manifest),
        },
        "cases": output_cases,
    }
    destination = args.out_dir / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(destination.resolve()),
                "windows": len(output_cases),
                "durationSec": sum(float(case["durationSec"]) for case in output_cases),
                "sourceCases": len(source_cases),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
