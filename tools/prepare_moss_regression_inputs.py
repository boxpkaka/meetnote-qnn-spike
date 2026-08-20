#!/usr/bin/env python3
"""Create deterministic duration prefixes for MOSS boundary regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_moss_alimeeting_inputs import SAMPLE_RATE, sha256, wav_layout, write_channel_zero_clip


DEFAULT_DURATIONS = (30, 60, 90, 120)


def prepare(source: Path, output_dir: Path, durations: list[int]) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _, _, _, _, available_frames = wav_layout(source)
    if not durations or any(seconds <= 0 for seconds in durations):
        raise ValueError("durations must contain positive seconds")
    if durations != sorted(set(durations)):
        raise ValueError("durations must be unique and increasing")
    if durations[-1] * SAMPLE_RATE > available_frames:
        raise ValueError("source WAV is shorter than the longest regression duration")

    output_dir.mkdir(parents=True)
    cases = []
    for seconds in durations:
        output = output_dir / f"{source.stem}_{seconds}s.wav"
        frames = write_channel_zero_clip(source, output, 0, seconds * SAMPLE_RATE)
        cases.append(
            {
                "id": f"{source.stem}_{seconds}s",
                "duration_seconds": frames / SAMPLE_RATE,
                "sample_count": frames,
                "wav": str(output.resolve()),
                "sha256": sha256(output),
            }
        )
    manifest = {
        "format": "meetnote.moss_boundary_regression_inputs.v1",
        "source": {
            "wav": str(source.resolve()),
            "sha256": sha256(source),
            "sample_count": available_frames,
        },
        "cases": cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--durations", type=int, nargs="+", default=list(DEFAULT_DURATIONS))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_wav, args.output_dir, args.durations), indent=2))


if __name__ == "__main__":
    main()
