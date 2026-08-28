#!/usr/bin/env python3
"""Score MOSS generations produced from native audio-encoder tensor dumps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def find_baseline(root: Path, case_name: str) -> Path:
    matches = sorted(root.glob(f"*/{case_name}.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one baseline for {case_name}, found {matches}")
    return matches[0]


def aggregate(rows: list[dict], key: str) -> float:
    errors = sum(row[key]["errors"] for row in rows)
    reference_chars = sum(row[key]["referenceChars"] for row in rows)
    return errors / reference_chars


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import evaluate_speaker_logs as speaker_eval
    from moss_transcribe_diarize import parse_transcript
    from run_moss_transcribe_diarize_eval import (
        cer,
        concatenated_permutation_cer,
        speaker_attributed_cer,
        timing_diagnostics,
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = {case["case"]: case for case in manifest["cases"]}
    metadata_paths = sorted(args.metadata_root.glob("*-q8-cpu/metadata.json"))
    rows = []
    for metadata_path in metadata_paths:
        case_name = metadata_path.parent.name.removesuffix("-q8-cpu")
        case = cases[case_name]
        expected = json.loads(Path(case["expected"]).read_text(encoding="utf-8"))["segments"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parsed = parse_transcript(metadata["generatedText"])
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
        reference_text = "".join(
            item.get("text", "")
            for item in sorted(expected, key=lambda item: (item["startMs"], item["endMs"]))
        )
        hypothesis_text = "".join(item["text"] for item in segments)
        plain = cer(reference_text, hypothesis_text)
        cp = concatenated_permutation_cer(expected, segments)
        diarization = speaker_eval.evaluate_segments(expected, segments)
        speaker_attributed = speaker_attributed_cer(
            expected, segments, diarization["mappingHypToRef"]
        )
        baseline_path = find_baseline(args.baseline_root, case_name)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        output_delta = cer(baseline["hypothesisText"], hypothesis_text)
        rows.append(
            {
                "case": case_name,
                "promptTokens": metadata["promptTokens"],
                "generatedTokens": len(metadata["generatedTokenIds"]),
                "segmentCount": len(segments),
                "plainCer": plain,
                "cpCer": cp,
                "speakerAttributedCer": speaker_attributed,
                "der": diarization["der"],
                "timing": timing_diagnostics(
                    segments, round(float(case["durationSec"]) * 1000)
                ),
                "outputCerVsBaseline": output_delta["cer"],
                "baseline": {
                    "plainCer": baseline["metrics"]["plainCer"]["cer"],
                    "cpCer": baseline["metrics"]["cpCer"]["cpCer"],
                    "der": baseline["metrics"]["diarization"]["der"],
                    "generatedTokens": baseline["generation"]["generatedTokens"],
                    "segmentCount": len(baseline["segments"]),
                },
            }
        )

    summary = {
        "schemaVersion": "meetnote.moss-audio-dump-eval.v1",
        "manifest": str(args.manifest.resolve()),
        "metadataRoot": str(args.metadata_root.resolve()),
        "baselineRoot": str(args.baseline_root.resolve()),
        "caseCount": len(rows),
        "aggregate": {
            "plainCer": aggregate(rows, "plainCer"),
            "cpCer": aggregate(rows, "cpCer"),
            "meanDer": sum(row["der"] for row in rows) / len(rows),
            "meanOutputCerVsBaseline": sum(
                row["outputCerVsBaseline"] for row in rows
            )
            / len(rows),
            "totalOutOfBoundsSegments": sum(
                row["timing"]["outOfBoundsSegments"] for row in rows
            ),
            "totalNonPositiveDurationSegments": sum(
                row["timing"]["nonPositiveDurationSegments"] for row in rows
            ),
        },
        "baselineAggregate": {
            "plainCer": sum(
                row["plainCer"]["referenceChars"] * row["baseline"]["plainCer"]
                for row in rows
            )
            / sum(row["plainCer"]["referenceChars"] for row in rows),
            "cpCer": sum(
                row["cpCer"]["referenceChars"] * row["baseline"]["cpCer"]
                for row in rows
            )
            / sum(row["cpCer"]["referenceChars"] for row in rows),
            "meanDer": sum(row["baseline"]["der"] for row in rows) / len(rows),
        },
        "perCase": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
