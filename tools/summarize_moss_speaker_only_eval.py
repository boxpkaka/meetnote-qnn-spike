#!/usr/bin/env python3
"""Score MOSS speaker-only output without requiring timestamp markers."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_moss_transcribe_diarize_eval import cer, concatenated_permutation_cer
from stitch_moss_transcribe_diarize_windows import (
    map_window_speakers,
    sortformer_paths,
    speaker_attributed_cer,
)

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    Levenshtein = None


SPEAKER_MARKER = re.compile(r"\[(S\d+)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--anchor-root", type=Path)
    parser.add_argument("--sortformer-root", type=Path)
    parser.add_argument("--comparison-summary", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing cases while still rejecting duplicate/ambiguous results.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


class MissingCaseError(RuntimeError):
    pass


class AmbiguousCaseError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_case(root: Path, case: str) -> Path:
    matches = list(root.glob(f"**/{case}.json"))
    if not matches:
        raise MissingCaseError(f"missing result for {case} under {root}")
    if len(matches) > 1:
        raise AmbiguousCaseError(f"ambiguous results for {case}: {sorted(matches)}")
    return matches[0]


def parse_speaker_only(raw_text: str) -> list[dict[str, Any]]:
    parts = SPEAKER_MARKER.split(raw_text)
    segments = []
    for index in range(1, len(parts), 2):
        text = parts[index + 1].strip()
        if not text:
            continue
        order = len(segments)
        segments.append(
            {
                "speakerId": parts[index],
                "text": text,
                "startMs": order,
                "endMs": order + 1,
            }
        )
    return segments


def normalized_chars(text: str) -> str:
    from run_moss_transcribe_diarize_eval import normalize_text

    return normalize_text(text)


def anchor_char_timestamps(anchor: dict[str, Any]) -> tuple[str, list[int]]:
    chars = []
    timestamps = []
    for segment in sorted(anchor["segments"], key=lambda item: item["startMs"]):
        for token, timestamp in zip(
            segment.get("tokens", []), segment.get("absoluteTokenTimestampsMs", [])
        ):
            normalized = normalized_chars(str(token))
            chars.extend(normalized)
            timestamps.extend([int(timestamp)] * len(normalized))
    return "".join(chars), timestamps


def load_anchor_for_case(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    case_name = str(case["case"])
    try:
        return json.loads(find_case(root, case_name).read_text(encoding="utf-8"))
    except MissingCaseError:
        source_case = str(case["sourceCase"])
        full = json.loads(find_case(root, source_case).read_text(encoding="utf-8"))
        start_ms = int(case["sourceStartMs"])
        end_ms = int(case["sourceEndMs"])
        segments = []
        for segment in full["segments"]:
            tokens = []
            timestamps = []
            for token, timestamp in zip(
                segment.get("tokens", []), segment.get("absoluteTokenTimestampsMs", [])
            ):
                if start_ms <= int(timestamp) < end_ms:
                    tokens.append(token)
                    timestamps.append(int(timestamp) - start_ms)
            if tokens:
                segments.append(
                    {
                        "startMs": timestamps[0],
                        "tokens": tokens,
                        "absoluteTokenTimestampsMs": timestamps,
                    }
                )
        return {"segments": segments}


def transfer_timestamps(source: str, target: str, source_times: list[int]) -> list[int]:
    """Align target characters to source and interpolate unmatched target positions."""
    mapped: list[int | None] = [None] * len(target)
    for tag, source_start, source_end, target_start, target_end in alignment_opcodes(
        source, target
    ):
        if tag == "equal":
            for offset in range(target_end - target_start):
                mapped[target_start + offset] = source_times[source_start + offset]
        elif tag == "replace" and source_end > source_start:
            for offset in range(target_end - target_start):
                source_offset = min(
                    source_end - source_start - 1,
                    (offset * (source_end - source_start)) // max(1, target_end - target_start),
                )
                mapped[target_start + offset] = source_times[source_start + source_offset]

    known = [index for index, value in enumerate(mapped) if value is not None]
    if not known:
        return [0] * len(target)
    for index, value in enumerate(mapped):
        if value is not None:
            continue
        left = next((item for item in reversed(known) if item < index), None)
        right = next((item for item in known if item > index), None)
        if left is None:
            mapped[index] = mapped[right] if right is not None else 0
        elif right is None:
            mapped[index] = mapped[left]
        else:
            fraction = (index - left) / (right - left)
            mapped[index] = round(mapped[left] + fraction * (mapped[right] - mapped[left]))
    result = [int(value) for value in mapped]
    for index in range(1, len(result)):
        result[index] = max(result[index], result[index - 1])
    return result


def alignment_opcodes(source: str, target: str) -> list[tuple[str, int, int, int, int]]:
    if Levenshtein is not None:
        return [
            (opcode.tag, opcode.src_start, opcode.src_end, opcode.dest_start, opcode.dest_end)
            for opcode in Levenshtein.opcodes(source, target)
        ]
    return list(difflib.SequenceMatcher(a=source, b=target, autojunk=False).get_opcodes())


def align_segments_to_anchor(
    segments: list[dict[str, Any]], anchor: dict[str, Any], source_start_ms: int
) -> tuple[list[dict[str, Any]], float]:
    target = "".join(normalized_chars(segment["text"]) for segment in segments)
    source, source_times = anchor_char_timestamps(anchor)
    target_times = transfer_timestamps(source, target, source_times)
    matched = sum(
        target_end - target_start
        for tag, _, _, target_start, target_end in alignment_opcodes(source, target)
        if tag == "equal"
    )
    aligned = []
    cursor = 0
    for segment in segments:
        length = len(normalized_chars(segment["text"]))
        times = target_times[cursor : cursor + length]
        cursor += length
        if not times:
            continue
        aligned.append(
            {
                **segment,
                "startMs": source_start_ms + times[0],
                "endMs": source_start_ms + times[-1] + 1,
                "localSpeakerId": segment["speakerId"],
            }
        )
    return aligned, matched / len(target) if target else 0.0


def anchor_mapped_role_metric(
    case: dict[str, Any],
    segments: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    anchor_root: Path,
    sortformer_root: Path,
) -> tuple[dict[str, Any], float, dict[str, str], list[dict[str, Any]]]:
    anchor = load_anchor_for_case(anchor_root, case)
    source_start_ms = int(case["sourceStartMs"])
    aligned, coverage = align_segments_to_anchor(segments, anchor, source_start_ms)
    result_path, metrics_path = sortformer_paths(sortformer_root, str(case["sourceCase"]))
    diar_segments = json.loads(result_path.read_text(encoding="utf-8"))["segments"]
    diar_mapping = json.loads(metrics_path.read_text(encoding="utf-8"))["mappingHypToRef"]
    local_to_global = map_window_speakers(aligned, diar_segments)
    for segment in aligned:
        global_speaker = local_to_global.get(segment["localSpeakerId"])
        segment["mappedReferenceSpeaker"] = diar_mapping.get(global_speaker)
    return (
        speaker_attributed_cer(expected, aligned, "mappedReferenceSpeaker"),
        coverage,
        local_to_global,
        aligned,
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    missing_cases = []
    meeting_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "reference": [],
            "hypothesis": [],
            "aligned": [],
            "matchedChars": 0.0,
            "targetChars": 0,
            "windowCount": 0,
        }
    )
    expected_windows_by_source: dict[str, int] = defaultdict(int)
    for case in manifest["cases"]:
        expected_windows_by_source[str(case["sourceCase"])] += 1
    for case in manifest["cases"]:
        case_name = str(case["case"])
        try:
            result_path = find_case(args.input_root, case_name)
        except MissingCaseError:
            if not args.allow_partial:
                raise
            missing_cases.append(case_name)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected = json.loads(Path(case["expected"]).read_text(encoding="utf-8"))["segments"]
        segments = parse_speaker_only(result["generation"]["rawText"])
        hypothesis = "".join(segment["text"] for segment in segments)
        reference = "".join(segment.get("text", "") for segment in expected)
        plain = cer(reference, hypothesis)
        cp = concatenated_permutation_cer(expected, segments)
        anchor_role = None
        alignment_coverage = None
        local_to_global = None
        aligned = []
        if args.anchor_root is not None and args.sortformer_root is not None:
            anchor_role, alignment_coverage, local_to_global, aligned = anchor_mapped_role_metric(
                case,
                segments,
                expected,
                args.anchor_root,
                args.sortformer_root,
            )
        baseline = None
        if args.baseline_root is not None:
            baseline_result = json.loads(
                find_case(args.baseline_root, case_name).read_text(encoding="utf-8")
            )
            baseline = {
                "generatedTokens": baseline_result["generation"]["generatedTokens"],
                "plainCer": baseline_result["metrics"]["plainCer"]["cer"],
                "cpCer": baseline_result["metrics"]["cpCer"]["cpCer"],
                "outputCer": cer(baseline_result["hypothesisText"], hypothesis)["cer"],
            }
        rows.append(
            {
                "case": case_name,
                "resultPath": str(result_path.resolve()),
                "resultSha256": sha256(result_path),
                "generatedTokens": result["generation"]["generatedTokens"],
                "hitMaxNewTokens": result["generation"]["hitMaxNewTokens"],
                "segmentCount": len(segments),
                "speakerCount": len({segment["speakerId"] for segment in segments}),
                "plainCer": plain,
                "cpCer": cp,
                "anchorMappedRoleCer": anchor_role,
                "alignmentCoverage": alignment_coverage,
                "localToGlobal": local_to_global,
                "baseline": baseline,
            }
        )

        source_case = str(case["sourceCase"])
        source_start_ms = int(case["sourceStartMs"])
        meeting = meeting_data[source_case]
        meeting["hypothesis"].append((source_start_ms, hypothesis))
        meeting["reference"].extend(
            {
                **segment,
                "startMs": int(segment["startMs"]) + source_start_ms,
                "endMs": int(segment["endMs"]) + source_start_ms,
            }
            for segment in expected
        )
        meeting["aligned"].extend(aligned)
        target_chars = len(normalized_chars(hypothesis))
        meeting["targetChars"] += target_chars
        meeting["matchedChars"] += (alignment_coverage or 0.0) * target_chars
        meeting["windowCount"] += 1

    comparison = None
    comparison_by_case: dict[str, dict[str, Any]] = {}
    if args.comparison_summary is not None:
        comparison = json.loads(args.comparison_summary.read_text(encoding="utf-8"))
        comparison_by_case = {
            str(row["case"]): row for row in comparison.get("perCase", [])
        }

    per_source = []
    for source_case, meeting in sorted(meeting_data.items()):
        reference = sorted(
            meeting["reference"], key=lambda item: (item["startMs"], item["endMs"])
        )
        reference_text = "".join(segment.get("text", "") for segment in reference)
        hypothesis_text = "".join(
            text for _, text in sorted(meeting["hypothesis"], key=lambda item: item[0])
        )
        plain = cer(reference_text, hypothesis_text)
        role = (
            speaker_attributed_cer(
                reference,
                sorted(meeting["aligned"], key=lambda item: (item["startMs"], item["endMs"])),
                "mappedReferenceSpeaker",
            )
            if meeting["aligned"]
            else None
        )
        complete_coverage = meeting["windowCount"] == expected_windows_by_source[source_case]
        baseline = comparison_by_case.get(source_case) if complete_coverage else None
        per_source.append(
            {
                "sourceCase": source_case,
                "windowCount": meeting["windowCount"],
                "expectedWindowCount": expected_windows_by_source[source_case],
                "completeCoverage": complete_coverage,
                "plainCer": plain,
                "anchorMappedRoleCer": role,
                "alignmentCoverage": (
                    meeting["matchedChars"] / meeting["targetChars"]
                    if meeting["targetChars"]
                    else None
                ),
                "comparison": baseline,
                "beatsComparisonPlainCer": (
                    plain["cer"] < baseline["plainCer"] if baseline is not None else None
                ),
                "beatsComparisonRoleCer": (
                    role["saCer"] < baseline["tokenSortformerSaCer"]
                    if role is not None and baseline is not None
                    else None
                ),
            }
        )

    reference_chars = sum(row["plainCer"]["referenceChars"] for row in rows)
    whole_reference_chars = sum(row["plainCer"]["referenceChars"] for row in per_source)
    output = {
        "schemaVersion": "meetnote.moss-speaker-only-eval.v2",
        "manifest": str(args.manifest.resolve()),
        "manifestSha256": sha256(args.manifest),
        "inputRoot": str(args.input_root.resolve()),
        "inputResultsSha256": hashlib.sha256(
            "".join(
                f"{row['case']}:{row['resultSha256']}\n"
                for row in sorted(rows, key=lambda item: item["case"])
            ).encode("utf-8")
        ).hexdigest(),
        "expectedCaseCount": len(manifest["cases"]),
        "caseCount": len(rows),
        "complete": not missing_cases and len(rows) == len(manifest["cases"]),
        "missingCases": missing_cases,
        "aggregate": {
            "plainCer": (
                sum(row["plainCer"]["errors"] for row in rows) / reference_chars
                if reference_chars
                else None
            ),
            "cpCer": (
                sum(row["cpCer"]["errors"] for row in rows) / reference_chars
                if reference_chars
                else None
            ),
            "generatedTokens": sum(row["generatedTokens"] for row in rows),
            "baselineGeneratedTokens": sum(
                row["baseline"]["generatedTokens"]
                for row in rows
                if row["baseline"] is not None
            ),
            "hitMaxNewTokens": sum(row["hitMaxNewTokens"] for row in rows),
            "anchorMappedRoleCer": (
                sum(row["anchorMappedRoleCer"]["errors"] for row in per_source)
                / whole_reference_chars
                if whole_reference_chars
                and all(row["anchorMappedRoleCer"] is not None for row in per_source)
                else None
            ),
            "meanAlignmentCoverage": (
                sum(row["alignmentCoverage"] for row in rows) / len(rows)
                if rows and all(row["alignmentCoverage"] is not None for row in rows)
                else None
            ),
            "wholeMeetingPlainCer": (
                sum(row["plainCer"]["errors"] for row in per_source)
                / whole_reference_chars
                if whole_reference_chars
                else None
            ),
            "meetingsBeatingComparisonPlainCer": sum(
                row["beatsComparisonPlainCer"] is True for row in per_source
            ),
            "meetingsBeatingComparisonRoleCer": sum(
                row["beatsComparisonRoleCer"] is True for row in per_source
            ),
        },
        "comparisonSummary": comparison,
        "perSource": per_source,
        "perCase": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
