#!/usr/bin/env python3
"""Build and evaluate synthetic multi-speaker diarization fixtures."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import shutil
import subprocess
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BYTES = 2
SILENCE_MS_DEFAULT = 300


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    synth = subparsers.add_parser("synthesize")
    synth.add_argument("--zip", required=True, type=Path)
    synth.add_argument("--out-dir", required=True, type=Path)
    synth.add_argument("--silence-ms", type=int, default=SILENCE_MS_DEFAULT)
    synth.add_argument("--max-clip-ms", type=int, default=0)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--expected", required=True, type=Path)
    evaluate.add_argument("--result", required=True, type=Path)
    evaluate.add_argument("--out", required=True, type=Path)

    subparsers.add_parser("self-test")

    args = parser.parse_args()
    if args.cmd == "synthesize":
        synthesize(args.zip, args.out_dir, args.silence_ms, args.max_clip_ms)
    elif args.cmd == "evaluate":
        evaluate_result(args.expected, args.result, args.out)
    elif args.cmd == "self-test":
        self_test()


def synthesize(zip_path: Path, out_dir: Path, silence_ms: int, max_clip_ms: int) -> None:
    require_tool("afconvert")
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = out_dir / "extracted"
    dataset_root = extract_dataset(zip_path, extract_dir)
    records = load_records(dataset_root)
    supervisions = load_supervisions(dataset_root)
    by_recording_id = {item["recording_id"]: item for item in supervisions}

    cases = {
        "synthetic_2spk": [0, 1, 0, 1],
        "synthetic_3spk": [0, 1, 2, 0, 2, 1],
    }
    selected = select_long_recordings(records, by_recording_id, max_speakers=3)
    decoded_dir = out_dir / "decoded"
    decoded_dir.mkdir(parents=True, exist_ok=True)
    decoded = []
    for record in selected:
        mp3 = dataset_root / record["sources"][0]["source"]
        wav = decoded_dir / f"{record['id']}.wav"
        if not wav.exists():
            subprocess.run(
                [
                    "afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    f"LEI16@{SAMPLE_RATE}",
                    "-c",
                    "1",
                    str(mp3),
                    str(wav),
                ],
                check=True,
            )
        samples = read_wav_i16(wav)
        if max_clip_ms > 0:
            max_bytes = int(SAMPLE_RATE * max_clip_ms / 1000) * SAMPLE_WIDTH_BYTES
            samples = samples[:max_bytes]
        supervision = by_recording_id[record["id"]]
        decoded.append(
            {
                "recording_id": record["id"],
                "speaker": supervision["speaker"],
                "text": supervision.get("text", ""),
                "source": str(mp3.relative_to(dataset_root)),
                "samples": samples,
            }
        )

    silence = b"\x00\x00" * int(SAMPLE_RATE * silence_ms / 1000)
    summary = {}
    for case_name, pattern in cases.items():
        expected = []
        pcm = bytearray()
        for segment_index, speaker_index in enumerate(pattern):
            if pcm:
                pcm.extend(silence)
            start_sample = len(pcm) // SAMPLE_WIDTH_BYTES
            item = decoded[speaker_index]
            pcm.extend(item["samples"])
            end_sample = len(pcm) // SAMPLE_WIDTH_BYTES
            expected.append(
                {
                    "segmentIndex": segment_index,
                    "speakerRef": item["speaker"],
                    "recordingId": item["recording_id"],
                    "source": item["source"],
                    "text": item["text"],
                    "startMs": round(start_sample * 1000 / SAMPLE_RATE),
                    "endMs": round(end_sample * 1000 / SAMPLE_RATE),
                }
            )

        wav_path = out_dir / f"{case_name}.wav"
        expected_path = out_dir / f"{case_name}.expected.json"
        write_wav_i16(wav_path, bytes(pcm))
        expected_path.write_text(json.dumps({"segments": expected}, ensure_ascii=False, indent=2))
        summary[case_name] = {
            "wav": str(wav_path),
            "expected": str(expected_path),
            "durationMs": round(len(pcm) / SAMPLE_WIDTH_BYTES * 1000 / SAMPLE_RATE),
            "speakers": sorted({s["speakerRef"] for s in expected}),
            "numExpectedTurns": len(expected),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_result(expected_path: Path, result_path: Path, out_path: Path) -> None:
    expected = json.loads(expected_path.read_text())["segments"]
    result = json.loads(result_path.read_text())
    predicted = result["segments"]

    expected_speech_ms = sum(max(0, seg["endMs"] - seg["startMs"]) for seg in expected)
    predicted_speech_ms = sum(max(0, seg["endMs"] - seg["startMs"]) for seg in predicted)
    overlap_ms = 0
    best_matches = []
    by_ref = defaultdict(Counter)

    for exp in expected:
        overlaps = [
            (
                overlap(exp["startMs"], exp["endMs"], pred["startMs"], pred["endMs"]),
                pred.get("speakerId") or f"S{pred['speakerIndex'] + 1}",
                pred,
            )
            for pred in predicted
        ]
        overlaps.sort(key=lambda item: item[0], reverse=True)
        best_overlap, best_speaker, _ = overlaps[0] if overlaps else (0, "NONE", None)
        overlap_ms += sum(item[0] for item in overlaps)
        by_ref[exp["speakerRef"]][best_speaker] += 1
        duration = max(1, exp["endMs"] - exp["startMs"])
        best_matches.append(
            {
                "speakerRef": exp["speakerRef"],
                "bestPredictedSpeaker": best_speaker,
                "bestOverlapRatio": best_overlap / duration,
            }
        )

    speaker_consistency = {}
    for speaker_ref, counts in by_ref.items():
        total = sum(counts.values())
        best_speaker, best_count = counts.most_common(1)[0]
        speaker_consistency[speaker_ref] = {
            "bestPredictedSpeaker": best_speaker,
            "consistency": best_count / total if total else 0.0,
            "counts": dict(counts),
        }

    predicted_speakers = sorted({seg.get("speakerId") or f"S{seg['speakerIndex'] + 1}" for seg in predicted})
    expected_speakers = sorted({seg["speakerRef"] for seg in expected})
    speaker_count_error = len(predicted_speakers) - len(expected_speakers)
    der_metrics = compute_der(expected, predicted)
    metrics = {
        "case": expected_path.name.replace(".expected.json", ""),
        "variant": result.get("variant"),
        "host": result.get("host"),
        "deviceModel": result.get("deviceModel"),
        "androidSdk": result.get("androidSdk"),
        "durationMs": result.get("durationMs"),
        "inferenceMs": result.get("inferenceMs"),
        "rtf": result.get("rtf"),
        "vmRssKbBefore": result.get("vmRssKbBefore"),
        "vmRssKbAfter": result.get("vmRssKbAfter"),
        "nativeHeapBytesBefore": result.get("nativeHeapBytesBefore"),
        "nativeHeapBytesAfter": result.get("nativeHeapBytesAfter"),
        "expectedSpeakerCount": len(expected_speakers),
        "predictedSpeakerCount": len(predicted_speakers),
        "speakerCountError": speaker_count_error,
        "speakerCountAbsError": abs(speaker_count_error),
        "speakerCountCorrect": speaker_count_error == 0,
        "expectedTurnCount": len(expected),
        "predictedTurnCount": len(predicted),
        "expectedSpeechMs": expected_speech_ms,
        "predictedSpeechMs": predicted_speech_ms,
        "speechOverlapCoverage": min(overlap_ms, expected_speech_ms) / max(1, expected_speech_ms),
        "meanBestTurnOverlap": (
            sum(item["bestOverlapRatio"] for item in best_matches) / max(1, len(best_matches))
        ),
        "expectedSpeakers": expected_speakers,
        "predictedSpeakers": predicted_speakers,
        **der_metrics,
        "speakerConsistency": speaker_consistency,
        "bestMatches": best_matches,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def compute_der(
    expected: list[dict],
    predicted: list[dict],
    frame_ms: int = 10,
    collar_ms: int = 250,
    skip_overlap: bool = False,
) -> dict:
    max_time = 0
    for seg in expected:
        max_time = max(max_time, int(seg["endMs"]))
    for seg in predicted:
        max_time = max(max_time, int(seg["endMs"]))
    if max_time <= 0:
        return der_result(
            miss_ms=0,
            false_alarm_ms=0,
            confusion_ms=0,
            scored_ref_ms=0,
            frame_ms=frame_ms,
            collar_ms=collar_ms,
            skip_overlap=skip_overlap,
        )

    frame_count = (max_time + frame_ms - 1) // frame_ms
    ref_frames = [set() for _ in range(frame_count)]
    hyp_frames = [set() for _ in range(frame_count)]
    ignored = [False] * frame_count

    for seg in expected:
        add_segment_to_frames(ref_frames, int(seg["startMs"]), int(seg["endMs"]), frame_ms, seg["speakerRef"])
        for boundary_ms in (int(seg["startMs"]), int(seg["endMs"])):
            mark_collar(ignored, boundary_ms, collar_ms, frame_ms)

    for seg in predicted:
        speaker = seg.get("speakerId") or f"S{seg['speakerIndex'] + 1}"
        add_segment_to_frames(hyp_frames, int(seg["startMs"]), int(seg["endMs"]), frame_ms, speaker)

    scored_indices = [
        index
        for index in range(frame_count)
        if not ignored[index] and not (skip_overlap and len(ref_frames[index]) > 1)
    ]
    mapping = best_hyp_to_ref_mapping(ref_frames, hyp_frames, scored_indices)

    miss_ms = 0
    false_alarm_ms = 0
    confusion_ms = 0
    scored_ref_ms = 0
    for index in scored_indices:
        refs = ref_frames[index]
        hyps = hyp_frames[index]
        ref_count = len(refs)
        hyp_count = len(hyps)
        scored_ref_ms += ref_count * frame_ms
        if ref_count == 0:
            false_alarm_ms += hyp_count * frame_ms
            continue
        if hyp_count == 0:
            miss_ms += ref_count * frame_ms
            continue

        mapped_hyps = {mapping[hyp] for hyp in hyps if hyp in mapping}
        correct = len(refs.intersection(mapped_hyps))
        miss_ms += max(0, ref_count - hyp_count) * frame_ms
        false_alarm_ms += max(0, hyp_count - ref_count) * frame_ms
        confusion_ms += max(0, min(ref_count, hyp_count) - correct) * frame_ms

    return der_result(
        miss_ms=miss_ms,
        false_alarm_ms=false_alarm_ms,
        confusion_ms=confusion_ms,
        scored_ref_ms=scored_ref_ms,
        frame_ms=frame_ms,
        collar_ms=collar_ms,
        skip_overlap=skip_overlap,
    )


def der_result(
    miss_ms: int,
    false_alarm_ms: int,
    confusion_ms: int,
    scored_ref_ms: int,
    frame_ms: int,
    collar_ms: int,
    skip_overlap: bool,
) -> dict:
    denom = max(1, scored_ref_ms)
    der = (miss_ms + false_alarm_ms + confusion_ms) / denom
    return {
        "der": der,
        "derMissRate": miss_ms / denom,
        "derFalseAlarmRate": false_alarm_ms / denom,
        "derConfusionRate": confusion_ms / denom,
        "derFrameMs": frame_ms,
        "derCollarMs": collar_ms,
        "derSkipOverlap": skip_overlap,
        "derScoredRefMs": scored_ref_ms,
    }


def add_segment_to_frames(frames: list[set], start_ms: int, end_ms: int, frame_ms: int, speaker: str) -> None:
    if end_ms <= start_ms:
        return
    start = max(0, start_ms // frame_ms)
    end = max(start, (end_ms + frame_ms - 1) // frame_ms)
    for index in range(start, min(end, len(frames))):
        frames[index].add(speaker)


def mark_collar(ignored: list[bool], boundary_ms: int, collar_ms: int, frame_ms: int) -> None:
    start = max(0, (boundary_ms - collar_ms) // frame_ms)
    end = min(len(ignored), (boundary_ms + collar_ms + frame_ms - 1) // frame_ms)
    for index in range(start, end):
        ignored[index] = True


def best_hyp_to_ref_mapping(
    ref_frames: list[set],
    hyp_frames: list[set],
    scored_indices: list[int],
) -> dict[str, str]:
    ref_labels = sorted({speaker for index in scored_indices for speaker in ref_frames[index]})
    hyp_labels = sorted({speaker for index in scored_indices for speaker in hyp_frames[index]})
    if not ref_labels or not hyp_labels:
        return {}

    weights = {
        (ref, hyp): 0
        for ref in ref_labels
        for hyp in hyp_labels
    }
    for index in scored_indices:
        for ref in ref_frames[index]:
            for hyp in hyp_frames[index]:
                weights[(ref, hyp)] += 1

    if max(len(ref_labels), len(hyp_labels)) <= 7:
        return exact_best_mapping(ref_labels, hyp_labels, weights)
    if min(len(ref_labels), len(hyp_labels)) <= 16:
        return dynamic_best_mapping(ref_labels, hyp_labels, weights)
    return greedy_best_mapping(ref_labels, hyp_labels, weights)


def exact_best_mapping(ref_labels: list[str], hyp_labels: list[str], weights: dict[tuple[str, str], int]) -> dict[str, str]:
    best_score = -1
    best_mapping: dict[str, str] = {}
    if len(hyp_labels) <= len(ref_labels):
        for refs in itertools.permutations(ref_labels, len(hyp_labels)):
            mapping = dict(zip(hyp_labels, refs))
            score = sum(weights[(ref, hyp)] for hyp, ref in mapping.items())
            if score > best_score:
                best_score = score
                best_mapping = mapping
    else:
        for hyps in itertools.permutations(hyp_labels, len(ref_labels)):
            mapping = {hyp: ref for hyp, ref in zip(hyps, ref_labels)}
            score = sum(weights[(ref, hyp)] for hyp, ref in mapping.items())
            if score > best_score:
                best_score = score
                best_mapping = mapping
    return best_mapping


def dynamic_best_mapping(
    ref_labels: list[str], hyp_labels: list[str], weights: dict[tuple[str, str], int]
) -> dict[str, str]:
    """Find an exact mapping when one side is small and the other is large.

    Long-meeting auto clustering can produce hundreds of hypothesis identities.
    Greedy matching is not guaranteed to maximize the global overlap in that
    regime.  The DP is exponential only in the smaller side.
    """
    if len(ref_labels) <= len(hyp_labels):
        # mask tracks reference identities already assigned while hypotheses
        # are streamed one by one.
        states: dict[int, tuple[int, dict[str, str]]] = {0: (0, {})}
        for hyp in hyp_labels:
            next_states = dict(states)
            for mask, (score, mapping) in states.items():
                for index, ref in enumerate(ref_labels):
                    bit = 1 << index
                    if mask & bit:
                        continue
                    candidate_score = score + weights[(ref, hyp)]
                    candidate_mask = mask | bit
                    current = next_states.get(candidate_mask)
                    if current is None or candidate_score > current[0]:
                        next_states[candidate_mask] = (
                            candidate_score,
                            {**mapping, hyp: ref},
                        )
            states = next_states
    else:
        # mask tracks hypotheses already assigned while references are
        # streamed.  The stored mapping remains hypothesis -> reference.
        states = {0: (0, {})}
        for ref in ref_labels:
            next_states = dict(states)
            for mask, (score, mapping) in states.items():
                for index, hyp in enumerate(hyp_labels):
                    bit = 1 << index
                    if mask & bit:
                        continue
                    candidate_score = score + weights[(ref, hyp)]
                    candidate_mask = mask | bit
                    current = next_states.get(candidate_mask)
                    if current is None or candidate_score > current[0]:
                        next_states[candidate_mask] = (
                            candidate_score,
                            {**mapping, hyp: ref},
                        )
            states = next_states
    return max(states.values(), key=lambda item: item[0])[1]


def greedy_best_mapping(ref_labels: list[str], hyp_labels: list[str], weights: dict[tuple[str, str], int]) -> dict[str, str]:
    mapping = {}
    used_refs = set()
    used_hyps = set()
    pairs = sorted(
        ((score, ref, hyp) for (ref, hyp), score in weights.items()),
        reverse=True,
    )
    for score, ref, hyp in pairs:
        if score <= 0:
            break
        if ref in used_refs or hyp in used_hyps:
            continue
        mapping[hyp] = ref
        used_refs.add(ref)
        used_hyps.add(hyp)
    return mapping


def self_test() -> None:
    cases = [
        (
            "perfect",
            [{"startMs": 0, "endMs": 1000, "speakerRef": "A"}],
            [{"startMs": 0, "endMs": 1000, "speakerId": "S1"}],
            0.0,
        ),
        (
            "false_alarm",
            [{"startMs": 0, "endMs": 1000, "speakerRef": "A"}],
            [{"startMs": 0, "endMs": 1500, "speakerId": "S1"}],
            0.5,
        ),
        (
            "miss",
            [{"startMs": 0, "endMs": 1000, "speakerRef": "A"}],
            [],
            1.0,
        ),
        (
            "confusion",
            [
                {"startMs": 0, "endMs": 500, "speakerRef": "A"},
                {"startMs": 500, "endMs": 1000, "speakerRef": "B"},
            ],
            [{"startMs": 0, "endMs": 1000, "speakerId": "S1"}],
            0.5,
        ),
        (
            "overlap_miss",
            [
                {"startMs": 0, "endMs": 1000, "speakerRef": "A"},
                {"startMs": 0, "endMs": 1000, "speakerRef": "B"},
            ],
            [{"startMs": 0, "endMs": 1000, "speakerId": "S1"}],
            0.5,
        ),
        (
            "collar",
            [{"startMs": 0, "endMs": 1000, "speakerRef": "A"}],
            [{"startMs": 0, "endMs": 1250, "speakerId": "S1"}],
            0.0,
        ),
    ]

    for item in cases:
        name, expected, predicted, want = item[:4]
        got = compute_der(expected, predicted, collar_ms=0)["der"]
        if name == "collar":
            got = compute_der(expected, predicted, collar_ms=250)["der"]
        if abs(got - want) > 1e-6:
            raise AssertionError(f"{name}: got DER={got}, want {want}")

    # Greedy picks H1->A=10 first and then H2->B=0.  The exact optimum is
    # H1->B=9 plus H2->A=9.  Extra zero-weight hypotheses force the large-side
    # DP path used by long-meeting auto clustering.
    refs = ["A", "B"]
    hyps = [f"H{i}" for i in range(8)]
    weights = {(ref, hyp): 0 for ref in refs for hyp in hyps}
    weights.update({("A", "H0"): 10, ("B", "H0"): 9, ("A", "H1"): 9})
    mapping = dynamic_best_mapping(refs, hyps, weights)
    if mapping.get("H0") != "B" or mapping.get("H1") != "A":
        raise AssertionError(f"dynamic mapping is not globally optimal: {mapping}")
    print("synthesize_diarization_eval.py self-test passed")


def extract_dataset(zip_path: Path, extract_dir: Path) -> Path:
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    roots = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"expected one dataset root under {extract_dir}, got {roots}")
    return roots[0]


def load_records(dataset_root: Path) -> list[dict]:
    path = dataset_root / "aidatatang_test_spk_balanced_500_recordings_packaged.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_supervisions(dataset_root: Path) -> list[dict]:
    path = dataset_root / "aidatatang_test_spk_balanced_500_supervisions_cleaned.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def select_long_recordings(records: list[dict], supervisions: dict, max_speakers: int) -> list[dict]:
    records = [
        record
        for record in records
        if record["id"] in supervisions and record.get("duration", 0.0) >= 2.5
    ]
    records.sort(key=lambda item: item["duration"], reverse=True)
    selected = []
    seen_speakers = set()
    for record in records:
        speaker = supervisions[record["id"]]["speaker"]
        if speaker in seen_speakers:
            continue
        selected.append(record)
        seen_speakers.add(speaker)
        if len(selected) >= max_speakers:
            return selected
    raise RuntimeError(f"only selected {len(selected)} speakers")


def read_wav_i16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != SAMPLE_RATE or wav.getsampwidth() != SAMPLE_WIDTH_BYTES:
            raise RuntimeError(
                f"unexpected wav format for {path}: "
                f"channels={wav.getnchannels()} rate={wav.getframerate()} width={wav.getsampwidth()}"
            )
        return wav.readframes(wav.getnframes())


def write_wav_i16(path: Path, frames: bytes) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(frames)


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"required tool not found: {name}")


if __name__ == "__main__":
    main()
