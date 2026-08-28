#!/usr/bin/env python3
"""Evaluate diarization with speaker-log-oriented diagnostics.

Besides DER, this reports minority-speaker recall, identity concentration,
overlap recall, and gold-utterance attribution.  All identity metrics use one
global optimal hypothesis-to-reference mapping; no lexical heuristics are used.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from synthesize_diarization_eval import best_hyp_to_ref_mapping, compute_der


FRAME_MS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.expected is None or args.result is None or args.out is None:
        parser.error("--expected, --result, and --out are required unless --self-test is used")
    metrics = evaluate(args.expected, args.result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


def evaluate(expected_path: Path, result_path: Path) -> dict[str, Any]:
    expected_payload = json.loads(expected_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = expected_payload["segments"]
    predicted = result.get("segments", [])
    return evaluate_segments(expected, predicted, result, expected_path)


def evaluate_segments(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    result: dict[str, Any] | None = None,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    result = result or {}
    ref_frames, hyp_frames = build_frames(expected, predicted)
    indices = list(range(len(ref_frames)))
    mapping = best_hyp_to_ref_mapping(ref_frames, hyp_frames, indices)
    ref_labels = sorted({speaker for frame in ref_frames for speaker in frame})
    hyp_labels = sorted({speaker for frame in hyp_frames for speaker in frame})
    matrix = overlap_matrix(ref_labels, hyp_labels, ref_frames, hyp_frames)

    per_speaker = speaker_metrics(ref_labels, hyp_labels, matrix, mapping, ref_frames, hyp_frames)
    attribution = utterance_attribution(expected, predicted, mapping)
    overlap_metrics = overlap_recall(ref_frames, hyp_frames, mapping)
    der_250 = compute_der(expected, predicted, frame_ms=FRAME_MS, collar_ms=250, skip_overlap=False)
    der_0 = compute_der(expected, predicted, frame_ms=FRAME_MS, collar_ms=0, skip_overlap=False)
    der_250_no_overlap = compute_der(
        expected, predicted, frame_ms=FRAME_MS, collar_ms=250, skip_overlap=True
    )
    jer = mean(item["jaccardErrorRate"] for item in per_speaker.values())

    return {
        "schemaVersion": "meetnote.speaker-log-eval.v1",
        "case": result.get("case") or (expected_path.stem.split(".")[0] if expected_path else None),
        "variant": result.get("variant"),
        "host": result.get("host"),
        "durationMs": result.get("durationMs"),
        "inferenceMs": result.get("inferenceMs"),
        "rtf": result.get("rtf"),
        "expectedSpeakerCount": len(ref_labels),
        "predictedSpeakerCount": len(hyp_labels),
        "speakerCountError": len(hyp_labels) - len(ref_labels),
        "speakerCountAbsError": abs(len(hyp_labels) - len(ref_labels)),
        "expectedTurnCount": len(expected),
        "predictedTurnCount": len(predicted),
        "mappingHypToRef": mapping,
        "der": der_250["der"],
        "der250ms": der_250,
        "der0ms": der_0,
        "der250msSkipOverlap": der_250_no_overlap,
        "jer": jer,
        "macroSpeakerRecall": mean(item["recall"] for item in per_speaker.values()),
        "minSpeakerRecall": min((item["recall"] for item in per_speaker.values()), default=None),
        "macroDominantHypCoverage": mean(
            item["dominantHypCoverage"] for item in per_speaker.values()
        ),
        "macroEffectiveFragmentation": mean(
            item["effectiveHypotheses"] for item in per_speaker.values()
        ),
        "macroEffectiveMerge": effective_merge(hyp_labels, ref_labels, matrix),
        **overlap_metrics,
        **attribution,
        "perSpeaker": per_speaker,
        "overlapFrames": {
            ref: {hyp: frames * FRAME_MS for hyp, frames in row.items()}
            for ref, row in matrix.items()
        },
    }


def build_frames(
    expected: list[dict[str, Any]], predicted: list[dict[str, Any]]
) -> tuple[list[set[str]], list[set[str]]]:
    max_ms = max(
        [int(item["endMs"]) for item in expected]
        + [int(item["endMs"]) for item in predicted]
        + [0]
    )
    count = math.ceil(max_ms / FRAME_MS)
    refs = [set() for _ in range(count)]
    hyps = [set() for _ in range(count)]
    for item in expected:
        add_frames(refs, int(item["startMs"]), int(item["endMs"]), item["speakerRef"])
    for item in predicted:
        add_frames(hyps, int(item["startMs"]), int(item["endMs"]), hyp_label(item))
    return refs, hyps


def add_frames(frames: list[set[str]], start_ms: int, end_ms: int, speaker: str) -> None:
    for index in range(max(0, start_ms // FRAME_MS), min(len(frames), math.ceil(end_ms / FRAME_MS))):
        frames[index].add(speaker)


def overlap_matrix(
    refs: list[str], hyps: list[str], ref_frames: list[set[str]], hyp_frames: list[set[str]]
) -> dict[str, dict[str, int]]:
    matrix = {ref: {hyp: 0 for hyp in hyps} for ref in refs}
    for ref_set, hyp_set in zip(ref_frames, hyp_frames):
        for ref in ref_set:
            for hyp in hyp_set:
                matrix[ref][hyp] += 1
    return matrix


def speaker_metrics(
    refs: list[str],
    hyps: list[str],
    matrix: dict[str, dict[str, int]],
    mapping: dict[str, str],
    ref_frames: list[set[str]],
    hyp_frames: list[set[str]],
) -> dict[str, dict[str, Any]]:
    metrics = {}
    for ref in refs:
        ref_total = sum(ref in frame for frame in ref_frames)
        mapped_hyp = next((hyp for hyp, target in mapping.items() if target == ref), None)
        correct = matrix[ref].get(mapped_hyp, 0) if mapped_hyp else 0
        hyp_total = sum(mapped_hyp in frame for frame in hyp_frames) if mapped_hyp else 0
        union = ref_total + hyp_total - correct
        shares = [matrix[ref][hyp] for hyp in hyps if matrix[ref][hyp] > 0]
        metrics[ref] = {
            "referenceSpeechMs": ref_total * FRAME_MS,
            "mappedHypothesis": mapped_hyp,
            "recall": correct / ref_total if ref_total else 0.0,
            "jaccardErrorRate": 1.0 - (correct / union if union else 0.0),
            "dominantHypCoverage": max(shares, default=0) / ref_total if ref_total else 0.0,
            "effectiveHypotheses": effective_count(shares),
        }
    return metrics


def effective_merge(
    hyps: list[str], refs: list[str], matrix: dict[str, dict[str, int]]
) -> float | None:
    values = [effective_count([matrix[ref][hyp] for ref in refs if matrix[ref][hyp] > 0]) for hyp in hyps]
    return mean(values)


def effective_count(weights: list[int]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    return 1.0 / sum((weight / total) ** 2 for weight in weights)


def overlap_recall(
    refs: list[set[str]], hyps: list[set[str]], mapping: dict[str, str]
) -> dict[str, Any]:
    reference_instances = 0
    correct_instances = 0
    overlap_frames = 0
    for ref_set, hyp_set in zip(refs, hyps):
        if len(ref_set) <= 1:
            continue
        overlap_frames += 1
        mapped = {mapping[hyp] for hyp in hyp_set if hyp in mapping}
        reference_instances += len(ref_set)
        correct_instances += len(ref_set.intersection(mapped))
    return {
        "referenceOverlapMs": overlap_frames * FRAME_MS,
        "overlapSpeakerRecall": (
            correct_instances / reference_instances if reference_instances else None
        ),
    }


def utterance_attribution(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    mapping: dict[str, str],
) -> dict[str, Any]:
    correct_count = 0
    attributed_count = 0
    correct_duration = 0
    total_duration = 0
    by_ref: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for utterance in expected:
        duration = max(0, int(utterance["endMs"]) - int(utterance["startMs"]))
        total_duration += duration
        overlaps: dict[str, int] = defaultdict(int)
        for hypothesis in predicted:
            amount = overlap_ms(utterance, hypothesis)
            if amount:
                overlaps[hyp_label(hypothesis)] += amount
        predicted_speaker = max(overlaps, key=lambda key: (overlaps[key], key)) if overlaps else None
        mapped = mapping.get(predicted_speaker) if predicted_speaker else None
        correct = mapped == utterance["speakerRef"]
        if predicted_speaker:
            attributed_count += 1
        if correct:
            correct_count += 1
            correct_duration += duration
        by_ref[utterance["speakerRef"]][0] += int(correct)
        by_ref[utterance["speakerRef"]][1] += 1
    count = len(expected)
    per_ref_accuracy = {
        ref: correct / total if total else 0.0 for ref, (correct, total) in by_ref.items()
    }
    return {
        "utteranceAttributionAccuracy": correct_count / count if count else None,
        "durationWeightedUtteranceAttributionAccuracy": (
            correct_duration / total_duration if total_duration else None
        ),
        "macroUtteranceAttributionAccuracy": mean(per_ref_accuracy.values()),
        "unattributedUtteranceRate": (count - attributed_count) / count if count else None,
        "perSpeakerUtteranceAttributionAccuracy": per_ref_accuracy,
    }


def hyp_label(segment: dict[str, Any]) -> str:
    return segment.get("speakerId") or f"S{int(segment['speakerIndex']) + 1}"


def overlap_ms(a: dict[str, Any], b: dict[str, Any]) -> int:
    return max(0, min(int(a["endMs"]), int(b["endMs"])) - max(int(a["startMs"]), int(b["startMs"])))


def mean(values: Any) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def self_test() -> None:
    expected = [
        {"speakerRef": "A", "startMs": 0, "endMs": 1000, "utteranceId": "u1"},
        {"speakerRef": "B", "startMs": 1000, "endMs": 2000, "utteranceId": "u2"},
    ]
    perfect = [
        {"speakerId": "X", "startMs": 0, "endMs": 1000},
        {"speakerId": "Y", "startMs": 1000, "endMs": 2000},
    ]
    metrics = evaluate_segments(expected, perfect)
    assert metrics["der0ms"]["der"] == 0.0, metrics
    assert metrics["macroSpeakerRecall"] == 1.0, metrics
    assert metrics["utteranceAttributionAccuracy"] == 1.0, metrics

    merged = [{"speakerId": "X", "startMs": 0, "endMs": 2000}]
    metrics = evaluate_segments(expected, merged)
    assert metrics["speakerCountAbsError"] == 1, metrics
    assert metrics["macroEffectiveMerge"] == 2.0, metrics
    assert metrics["utteranceAttributionAccuracy"] == 0.5, metrics
    print("evaluate_speaker_logs.py self-test passed")


if __name__ == "__main__":
    main()
