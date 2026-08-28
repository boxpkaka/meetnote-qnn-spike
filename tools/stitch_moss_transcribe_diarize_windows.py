#!/usr/bin/env python3
"""Stitch MOSS window outputs and evaluate them with a global Sortformer timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import run_moss_transcribe_diarize_eval as moss

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    Levenshtein = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--window-manifest", required=True, type=Path)
    parser.add_argument("--window-results-dir", required=True, type=Path)
    parser.add_argument("--sortformer-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--case-exact", action="append", default=[])
    parser.add_argument("--context-limit", type=int, default=4096)
    return parser.parse_args()


def cer(reference: str, hypothesis: str) -> dict[str, Any]:
    normalized_reference = moss.normalize_text(reference)
    normalized_hypothesis = moss.normalize_text(hypothesis)
    if Levenshtein is None:
        return moss.cer(reference, hypothesis)
    errors = int(Levenshtein.distance(normalized_reference, normalized_hypothesis))
    return {
        "errors": errors,
        "referenceChars": len(normalized_reference),
        "hypothesisChars": len(normalized_hypothesis),
        "cer": errors / len(normalized_reference) if normalized_reference else None,
    }


def overlap_ms(a: dict[str, Any], b: dict[str, Any]) -> int:
    return max(0, min(int(a["endMs"]), int(b["endMs"])) - max(int(a["startMs"]), int(b["startMs"])))


def sortformer_paths(root: Path, case: str) -> tuple[Path, Path]:
    for directory in (root, root / case):
        result = directory / f"{case}.result.json"
        metrics = directory / f"{case}.metrics.json"
        if result.is_file() and metrics.is_file():
            return result, metrics
    raise FileNotFoundError(f"missing Sortformer result/metrics for {case} under {root}")


def choose_timeline_speaker(
    timestamp_ms: int,
    segments: list[dict[str, Any]],
    previous: str | None,
) -> str | None:
    active = sorted(
        {
            str(segment["speakerId"])
            for segment in segments
            if int(segment["startMs"]) <= timestamp_ms < int(segment["endMs"])
        }
    )
    if previous in active:
        return previous
    if len(active) == 1:
        return active[0]
    if not active:
        return None
    window = {"startMs": timestamp_ms - 100, "endMs": timestamp_ms + 100}
    support = {
        speaker: sum(
            overlap_ms(window, item)
            for item in segments
            if str(item["speakerId"]) == speaker
        )
        for speaker in active
    }
    return min(active, key=lambda speaker: (-support[speaker], speaker))


def label_segments(
    hypothesis: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
    diar_mapping: dict[str, str],
) -> None:
    for segment in hypothesis:
        support: dict[str, int] = defaultdict(int)
        for turn in diar_segments:
            amount = overlap_ms(segment, turn)
            if amount:
                support[str(turn["speakerId"])] += amount
        speaker = min(support, key=lambda item: (-support[item], item)) if support else None
        segment["sortformerSpeakerId"] = speaker
        segment["mappedReferenceSpeaker"] = diar_mapping.get(speaker) if speaker else None


def map_window_speakers(
    segments: list[dict[str, Any]], diar_segments: list[dict[str, Any]]
) -> dict[str, str]:
    local_speakers = sorted({str(segment["localSpeakerId"]) for segment in segments})
    global_speakers = sorted({str(segment["speakerId"]) for segment in diar_segments})
    size = max(len(local_speakers), len(global_speakers))
    if size == 0:
        return {}
    overlap_by_pair = {
        (local, global_speaker): sum(
            overlap_ms(local_segment, global_segment)
            for local_segment in segments
            if str(local_segment["localSpeakerId"]) == local
            for global_segment in diar_segments
            if str(global_segment["speakerId"]) == global_speaker
        )
        for local in local_speakers
        for global_speaker in global_speakers
    }
    max_overlap = max(overlap_by_pair.values(), default=0)
    costs = [
        [
            max_overlap
            - (
                overlap_by_pair[(local_speakers[row], global_speakers[column])]
                if row < len(local_speakers) and column < len(global_speakers)
                else 0
            )
            for column in range(size)
        ]
        for row in range(size)
    ]
    _, assignment = moss.minimum_assignment(costs)
    return {
        local_speakers[row]: global_speakers[column]
        for row, column in enumerate(assignment[: len(local_speakers)])
        if column < len(global_speakers)
    }


def speaker_attributed_cer(
    reference: list[dict[str, Any]],
    hypothesis: list[dict[str, Any]],
    speaker_field: str,
) -> dict[str, Any]:
    reference_by_speaker: dict[str, list[str]] = defaultdict(list)
    hypothesis_by_speaker: dict[str, list[str]] = defaultdict(list)
    for segment in sorted(reference, key=lambda item: (item["startMs"], item["endMs"])):
        reference_by_speaker[str(segment["speakerRef"])].append(segment.get("text", ""))
    for segment in sorted(hypothesis, key=lambda item: (item["startMs"], item["endMs"])):
        speaker = str(segment.get(speaker_field) or "__unattributed__")
        hypothesis_by_speaker[speaker].append(segment.get("text", ""))
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


def weighted_window_metric(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    errors = sum(int(row["metrics"][key]["errors"]) for row in rows)
    reference_chars = sum(int(row["metrics"][key]["referenceChars"]) for row in rows)
    return {
        "errors": errors,
        "referenceChars": reference_chars,
        value: errors / reference_chars if reference_chars else None,
    }


def character_timeline_units(
    segments: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
    diar_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """Approximate per-character timestamps uniformly inside each MOSS segment."""
    units: list[dict[str, Any]] = []
    previous_speaker: str | None = None
    for segment in segments:
        text = str(segment.get("text", ""))
        if not text:
            continue
        start_ms = int(segment["startMs"])
        duration_ms = max(1, int(segment["endMs"]) - start_ms)
        for index, char in enumerate(text):
            timestamp_ms = start_ms + round((index + 0.5) * duration_ms / len(text))
            speaker = choose_timeline_speaker(timestamp_ms, diar_segments, previous_speaker)
            if speaker is not None:
                previous_speaker = speaker
            units.append(
                {
                    "id": f"char-{len(units)}",
                    "startMs": timestamp_ms,
                    "endMs": timestamp_ms + 1,
                    "text": char,
                    "mappedReferenceSpeaker": diar_mapping.get(speaker) if speaker else None,
                }
            )
    return units


def shift_segments(
    result: dict[str, Any], window: dict[str, Any], window_index: int
) -> list[dict[str, Any]]:
    offset_ms = int(window["sourceStartMs"])
    window_end_ms = int(window["sourceEndMs"])
    shifted = []
    for segment_index, segment in enumerate(result["segments"]):
        shifted.append(
            {
                **segment,
                "id": f"w{window_index:03d}-utt-{segment_index}",
                "localSpeakerId": str(segment["speakerId"]),
                "windowCase": str(window["case"]),
                "localStartMs": int(segment["startMs"]),
                "localEndMs": int(segment["endMs"]),
                "startMs": int(segment["startMs"]) + offset_ms,
                "endMs": min(int(segment["endMs"]) + offset_ms, window_end_ms),
            }
        )
    return shifted


def corpus_metric(rows: list[dict[str, Any]], path: tuple[str, ...], value: str) -> dict[str, Any]:
    errors = 0
    reference_chars = 0
    for row in rows:
        metric: Any = row
        for key in path:
            metric = metric[key]
        errors += int(metric["errors"])
        reference_chars += int(metric["referenceChars"])
    return {
        "errors": errors,
        "referenceChars": reference_chars,
        value: errors / reference_chars if reference_chars else None,
    }


def optional_corpus_metric(
    rows: list[dict[str, Any]], path: tuple[str, ...], value: str
) -> float | None:
    if any(row["metrics"].get(path[-1]) is None for row in rows):
        return None
    return corpus_metric(rows, path, value)[value]


def main() -> None:
    args = parse_args()
    if args.context_limit <= 0:
        raise SystemExit("--context-limit must be positive")
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    window_manifest = json.loads(args.window_manifest.read_text(encoding="utf-8"))
    boundary_mode = window_manifest.get("selection", {}).get("boundaryMode", "reference-safe")
    selected = set(args.case_exact)
    source_cases = [
        case for case in source_manifest["cases"] if not selected or str(case["case"]) in selected
    ]
    missing = selected - {str(case["case"]) for case in source_cases}
    if missing:
        raise SystemExit(f"unknown source cases: {sorted(missing)}")

    windows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in window_manifest["cases"]:
        windows_by_source[str(window["sourceCase"])].append(window)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source_case in source_cases:
        source_name = str(source_case["case"])
        windows = sorted(
            windows_by_source[source_name], key=lambda item: int(item["sourceStartMs"])
        )
        if not windows or int(windows[0]["sourceStartMs"]) != 0:
            raise RuntimeError(f"incomplete leading window coverage for {source_name}")

        previous_end = 0
        window_rows = []
        segments: list[dict[str, Any]] = []
        inference_sec = 0.0
        for window_index, window in enumerate(windows):
            start_ms = int(window["sourceStartMs"])
            if start_ms != previous_end:
                raise RuntimeError(
                    f"non-contiguous windows for {source_name}: {previous_end} -> {start_ms}"
                )
            previous_end = int(window["sourceEndMs"])
            result_path = args.window_results_dir / f"{window['case']}.json"
            if not result_path.is_file():
                raise FileNotFoundError(f"missing MOSS window result: {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            window_rows.append(result)
            inference_sec += float(result["generation"]["inferenceSec"])
            segments.extend(shift_segments(result, window, window_index))

        source_duration_ms = round(float(source_case["durationSec"]) * 1000)
        if previous_end != source_duration_ms:
            raise RuntimeError(
                f"incomplete trailing window coverage for {source_name}: "
                f"{previous_end} != {source_duration_ms}"
            )
        reference = json.loads(Path(source_case["expected"]).read_text(encoding="utf-8"))["segments"]
        reference_text = "".join(
            item.get("text", "")
            for item in sorted(reference, key=lambda item: (item["startMs"], item["endMs"]))
        )
        hypothesis_text = "".join(
            item.get("text", "")
            for item in sorted(segments, key=lambda item: (item["startMs"], item["endMs"]))
        )

        sortformer_result_path, sortformer_metrics_path = sortformer_paths(args.sortformer_dir, source_name)
        sortformer_result = json.loads(sortformer_result_path.read_text(encoding="utf-8"))
        sortformer_metrics = json.loads(sortformer_metrics_path.read_text(encoding="utf-8"))
        diar_segments = sortformer_result["segments"]
        diar_mapping = sortformer_metrics["mappingHypToRef"]
        label_segments(segments, diar_segments, diar_mapping)
        window_speaker_mappings = {}
        for window in windows:
            window_case = str(window["case"])
            window_segments = [
                segment for segment in segments if segment["windowCase"] == window_case
            ]
            local_to_global = map_window_speakers(window_segments, diar_segments)
            window_speaker_mappings[window_case] = local_to_global
            for segment in window_segments:
                global_speaker = local_to_global.get(str(segment["localSpeakerId"]))
                segment["mappedWindowSpeakerReference"] = (
                    diar_mapping.get(global_speaker) if global_speaker else None
                )
        character_units = character_timeline_units(segments, diar_segments, diar_mapping)

        plain_cer = cer(reference_text, hypothesis_text)
        segment_sortformer = speaker_attributed_cer(reference, segments, "mappedReferenceSpeaker")
        character_sortformer = speaker_attributed_cer(
            reference, character_units, "mappedReferenceSpeaker"
        )
        window_mapped_sortformer = speaker_attributed_cer(
            reference, segments, "mappedWindowSpeakerReference"
        )
        local_window_cp = (
            weighted_window_metric(window_rows, "cpCer", "cpCer")
            if boundary_mode == "reference-safe"
            else None
        )
        local_window_time_mapped = (
            weighted_window_metric(window_rows, "timeMappedSaCer", "saCer")
            if boundary_mode == "reference-safe"
            else None
        )
        context_lengths = [
            int(window_row["generation"]["promptLen"])
            + int(window_row["generation"]["generatedTokens"])
            for window_row in window_rows
        ]
        row = {
            "schemaVersion": "meetnote.moss-window-stitch.v2",
            "case": source_name,
            "durationSec": float(source_case["durationSec"]),
            "windowCount": len(windows),
            "windowCases": [str(window["case"]) for window in windows],
            "referenceText": reference_text,
            "hypothesisText": hypothesis_text,
            "inferenceSec": inference_sec,
            "rtf": inference_sec / float(source_case["durationSec"]),
            "context": {
                "limit": args.context_limit,
                "maxPromptTokens": max(
                    int(window_row["generation"]["promptLen"])
                    for window_row in window_rows
                ),
                "maxGeneratedTokens": max(
                    int(window_row["generation"]["generatedTokens"])
                    for window_row in window_rows
                ),
                "maxTotalTokens": max(context_lengths),
                "windowsOverLimit": sum(
                    length > args.context_limit for length in context_lengths
                ),
                "perWindow": [
                    {
                        "case": str(window_row["case"]),
                        "promptTokens": int(window_row["generation"]["promptLen"]),
                        "generatedTokens": int(
                            window_row["generation"]["generatedTokens"]
                        ),
                        "totalTokens": total_tokens,
                        "overLimit": total_tokens > args.context_limit,
                    }
                    for window_row, total_tokens in zip(window_rows, context_lengths)
                ],
            },
            "metrics": {
                "plainCer": plain_cer,
                "segmentSortformerSaCer": segment_sortformer,
                "windowSpeakerMappedSortformerSaCer": window_mapped_sortformer,
                "uniformCharacterSortformerSaCer": character_sortformer,
                "localWindowCpCerUpperBound": local_window_cp,
                "localWindowTimeMappedSaCerUpperBound": local_window_time_mapped,
                "sortformerDer": sortformer_metrics["der"],
            },
            "segments": segments,
            "windowSpeakerMappings": window_speaker_mappings,
        }
        rows.append(row)
        (args.out_dir / f"{source_name}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    duration_sec = sum(float(row["durationSec"]) for row in rows)
    summary = {
        "schemaVersion": "meetnote.moss-window-stitch-summary.v2",
        "sourceManifest": str(args.source_manifest.resolve()),
        "sourceManifestSha256": sha256(args.source_manifest),
        "windowManifest": str(args.window_manifest.resolve()),
        "windowManifestSha256": sha256(args.window_manifest),
        "boundaryMode": boundary_mode,
        "windowResultsDir": str(args.window_results_dir.resolve()),
        "sortformerDir": str(args.sortformer_dir.resolve()),
        "caseCount": len(rows),
        "durationSec": duration_sec,
        "contextLimit": args.context_limit,
        "maxPromptTokens": max(row["context"]["maxPromptTokens"] for row in rows),
        "maxGeneratedTokens": max(
            row["context"]["maxGeneratedTokens"] for row in rows
        ),
        "maxTotalTokens": max(row["context"]["maxTotalTokens"] for row in rows),
        "windowsOverContextLimit": sum(
            row["context"]["windowsOverLimit"] for row in rows
        ),
        "corpusPlainCer": corpus_metric(rows, ("metrics", "plainCer"), "cer")["cer"],
        "corpusSegmentSortformerSaCer": corpus_metric(
            rows, ("metrics", "segmentSortformerSaCer"), "saCer"
        )["saCer"],
        "corpusWindowSpeakerMappedSortformerSaCer": corpus_metric(
            rows, ("metrics", "windowSpeakerMappedSortformerSaCer"), "saCer"
        )["saCer"],
        "corpusUniformCharacterSortformerSaCer": corpus_metric(
            rows, ("metrics", "uniformCharacterSortformerSaCer"), "saCer"
        )["saCer"],
        "corpusLocalWindowCpCerUpperBound": optional_corpus_metric(
            rows, ("metrics", "localWindowCpCerUpperBound"), "cpCer"
        ),
        "corpusLocalWindowTimeMappedSaCerUpperBound": optional_corpus_metric(
            rows, ("metrics", "localWindowTimeMappedSaCerUpperBound"), "saCer"
        ),
        "meanSortformerDer": (
            sum(float(row["metrics"]["sortformerDer"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "rtf": sum(float(row["inferenceSec"]) for row in rows) / duration_sec,
        "perCase": [
            {
                "case": row["case"],
                "cer": row["metrics"]["plainCer"]["cer"],
                "segmentSortformerSaCer": row["metrics"]["segmentSortformerSaCer"]["saCer"],
                "windowSpeakerMappedSortformerSaCer": row["metrics"][
                    "windowSpeakerMappedSortformerSaCer"
                ]["saCer"],
                "uniformCharacterSortformerSaCer": row["metrics"][
                    "uniformCharacterSortformerSaCer"
                ]["saCer"],
                "localWindowCpCerUpperBound": (
                    row["metrics"]["localWindowCpCerUpperBound"]["cpCer"]
                    if row["metrics"]["localWindowCpCerUpperBound"] is not None
                    else None
                ),
                "sortformerDer": row["metrics"]["sortformerDer"],
                "rtf": row["rtf"],
                "maxTotalTokens": row["context"]["maxTotalTokens"],
                "windowsOverContextLimit": row["context"]["windowsOverLimit"],
            }
            for row in rows
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
