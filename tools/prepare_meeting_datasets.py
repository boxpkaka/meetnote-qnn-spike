#!/usr/bin/env python3
"""Build a model-independent canonical meeting dataset.

The converter keeps source transcripts intact. AliMeeting utterances come only
from the official far-field TextGrid annotations. MeetingBank has no reliable
utterance timing or speaker boundaries, so its utterance list remains empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = "meetnote.experimental.meeting.v1"
MANIFEST_VERSION = "meetnote.experimental.dataset_manifest.v1"
TOOL_VERSION = 2
ALLOWED_USAGE_SCOPE = "internal-research"

ALI_HOMEPAGE = "https://www.openslr.org/119/"
ALI_LICENSE = "CC BY-SA 4.0"
MEETINGBANK_HOMEPAGE = "https://meetingbank.github.io/"
MEETINGBANK_LICENSE = "CC BY-NC-SA 4.0"

ALI_SPLITS = {
    "train": {
        "far_textgrids": "AliMeeting/data/Train_Ali_far/textgrid_dir",
        "far_audio": "AliMeeting/data/Train_Ali_far/audio_dir",
        "near_audio": "AliMeeting/data/Train_Ali_near/audio_dir",
    },
    "validation": {
        "far_textgrids": "AliMeeting/data/Eval_Ali/Eval_Ali_far/textgrid_dir",
        "far_audio": "AliMeeting/data/Eval_Ali/Eval_Ali_far/audio_dir",
        "near_audio": "AliMeeting/data/Eval_Ali/Eval_Ali_near/audio_dir",
    },
    "test": {
        "far_textgrids": "AliMeeting/data/Test_Ali/Test_Ali_far/textgrid_dir",
        "far_audio": "AliMeeting/data/Test_Ali/Test_Ali_far/audio_dir",
        "near_audio": "AliMeeting/data/Test_Ali/Test_Ali_near/audio_dir",
    },
}

MEETINGBANK_SPLITS = {
    "train": "MeetingBank/text/train.json",
    "validation": "MeetingBank/text/validation.json",
    "test": "MeetingBank/text/test.json",
}

TIER_RE = re.compile(r"^\s*item \[\d+\]:\s*$")
INTERVAL_RE = re.compile(r"^\s*intervals \[(\d+)\]:\s*$")
SPEAKER_RE = re.compile(r"(SPK\d+)$")


@dataclass(frozen=True)
class TextGridInterval:
    source_speaker_id: str
    source_tier: str
    source_interval_index: int
    start_seconds: Decimal
    end_seconds: Decimal
    text: str


def sha256_file(path: Path, buffer_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(buffer_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path, data_root: Path) -> str:
    return path.resolve().relative_to(data_root.resolve()).as_posix()


def parse_praat_string(value: str) -> str:
    value = value.strip()
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        raise ValueError(f"invalid Praat string: {value!r}")
    return value[1:-1].replace('""', '"')


def seconds_to_ms(value: Decimal) -> int:
    return int((value * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def parse_textgrid(path: Path) -> tuple[int, list[TextGridInterval]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or 'Object class = "TextGrid"' not in lines[:4]:
        raise ValueError(f"not a long-form TextGrid: {path}")

    global_xmax: Decimal | None = None
    tier_name: str | None = None
    interval_index: int | None = None
    interval_xmin: Decimal | None = None
    interval_xmax: Decimal | None = None
    intervals: list[TextGridInterval] = []
    inside_tiers = False

    for line in lines:
        stripped = line.strip()
        if stripped == "item []:":
            inside_tiers = True
            continue
        if not inside_tiers:
            if stripped.startswith("xmax = "):
                global_xmax = Decimal(stripped.split("=", 1)[1].strip())
            continue
        if TIER_RE.match(line):
            tier_name = None
            interval_index = None
            continue
        if stripped.startswith("name = "):
            tier_name = parse_praat_string(stripped.split("=", 1)[1])
            continue
        match = INTERVAL_RE.match(line)
        if match:
            if tier_name is None:
                raise ValueError(f"interval before tier name in {path}")
            interval_index = int(match.group(1))
            interval_xmin = None
            interval_xmax = None
            continue
        if interval_index is None:
            continue
        if stripped.startswith("xmin = "):
            interval_xmin = Decimal(stripped.split("=", 1)[1].strip())
        elif stripped.startswith("xmax = "):
            interval_xmax = Decimal(stripped.split("=", 1)[1].strip())
        elif stripped.startswith("text = "):
            if interval_xmin is None or interval_xmax is None or tier_name is None:
                raise ValueError(f"incomplete interval {interval_index} in {path}")
            text = parse_praat_string(stripped.split("=", 1)[1])
            if text.strip():
                speaker_match = SPEAKER_RE.search(tier_name)
                if not speaker_match:
                    raise ValueError(f"tier has no SPK identifier: {tier_name!r} in {path}")
                if interval_xmin < 0 or interval_xmax <= interval_xmin:
                    raise ValueError(
                        f"invalid interval {interval_index} in {path}: "
                        f"{interval_xmin}..{interval_xmax}"
                    )
                intervals.append(
                    TextGridInterval(
                        source_speaker_id=speaker_match.group(1),
                        source_tier=tier_name,
                        source_interval_index=interval_index,
                        start_seconds=interval_xmin,
                        end_seconds=interval_xmax,
                        text=text,
                    )
                )
            interval_index = None

    if global_xmax is None or global_xmax <= 0:
        raise ValueError(f"TextGrid has no positive global xmax: {path}")
    if not intervals:
        raise ValueError(f"TextGrid has no non-empty intervals: {path}")
    return seconds_to_ms(global_xmax), intervals


def canonicalize_intervals(intervals: list[TextGridInterval]) -> list[dict[str, object]]:
    speaker_ids = sorted({interval.source_speaker_id for interval in intervals})
    speaker_map = {
        source_speaker: f"S{index + 1}"
        for index, source_speaker in enumerate(speaker_ids)
    }
    ordered = sorted(
        intervals,
        key=lambda item: (
            item.start_seconds,
            item.end_seconds,
            item.source_speaker_id,
            item.source_interval_index,
        ),
    )
    return [
        {
            "id": f"utt-{index}",
            "speaker": speaker_map[interval.source_speaker_id],
            "source_speaker_id": interval.source_speaker_id,
            "start_ms": seconds_to_ms(interval.start_seconds),
            "end_ms": seconds_to_ms(interval.end_seconds),
            "text": interval.text,
            "source_tier": interval.source_tier,
            "source_interval_index": interval.source_interval_index,
        }
        for index, interval in enumerate(ordered)
    ]


def matching_audio(directory: Path, meeting_id: str) -> list[Path]:
    return sorted(directory.glob(f"{meeting_id}_*.wav"), key=lambda path: path.name)


def iter_alimeeting(data_root: Path, split: str) -> Iterator[dict[str, object]]:
    layout = ALI_SPLITS[split]
    textgrid_dir = data_root / layout["far_textgrids"]
    far_audio_dir = data_root / layout["far_audio"]
    near_audio_dir = data_root / layout["near_audio"]
    for directory in (textgrid_dir, far_audio_dir, near_audio_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    for textgrid in sorted(textgrid_dir.glob("*.TextGrid"), key=lambda path: path.name):
        meeting_id = textgrid.stem
        duration_ms, raw_intervals = parse_textgrid(textgrid)
        utterances = canonicalize_intervals(raw_intervals)
        far_audio = matching_audio(far_audio_dir, meeting_id)
        near_audio = matching_audio(near_audio_dir, meeting_id)
        if not far_audio:
            raise ValueError(f"AliMeeting far audio missing for {meeting_id}")
        if not near_audio:
            raise ValueError(f"AliMeeting near audio missing for {meeting_id}")
        yield {
            "schema_version": SCHEMA_VERSION,
            "record_id": f"alimeeting:{meeting_id}",
            "dataset": "AliMeeting",
            "split": split,
            "meeting_id": meeting_id,
            "language": "zh-CN",
            "duration_ms": duration_ms,
            "transcript": "\n".join(str(item["text"]) for item in utterances),
            "utterances": utterances,
            "reference_summary": None,
            "media": {
                "far": [relative_path(path, data_root) for path in far_audio],
                "near": [relative_path(path, data_root) for path in near_audio],
            },
            "source": {
                "transcript_path": relative_path(textgrid, data_root),
                "transcript_sha256": sha256_file(textgrid),
                "homepage": ALI_HOMEPAGE,
                "license": ALI_LICENSE,
            },
        }


def iter_json_lines(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open(encoding="utf-8") as source:
        for row_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            document = json.loads(line)
            if not isinstance(document, dict):
                raise ValueError(f"expected JSON object at {path}:{row_number}")
            yield row_number, document


def iter_meetingbank(data_root: Path, split: str) -> Iterator[dict[str, object]]:
    source_path = data_root / MEETINGBANK_SPLITS[split]
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_relative = relative_path(source_path, data_root)
    for row_number, row in iter_json_lines(source_path):
        uid = row.get("uid")
        transcript = row.get("transcript")
        summary = row.get("summary")
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError(f"MeetingBank uid missing at {source_path}:{row_number}")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"MeetingBank transcript missing at {source_path}:{row_number}")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"MeetingBank summary missing at {source_path}:{row_number}")
        yield {
            "schema_version": SCHEMA_VERSION,
            "record_id": f"meetingbank:{uid}",
            "dataset": "MeetingBank",
            "split": split,
            "meeting_id": uid,
            "language": "en",
            "duration_ms": None,
            "transcript": transcript,
            "utterances": [],
            "reference_summary": summary,
            "media": {"far": [], "near": []},
            "source": {
                "transcript_path": source_relative,
                "row_number": row_number,
                "source_id": row.get("id"),
                "homepage": MEETINGBANK_HOMEPAGE,
                "license": MEETINGBANK_LICENSE,
            },
        }


def validate_record(record: dict[str, object]) -> None:
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema version: {record['schema_version']}")
    record_id = record.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be non-empty")
    transcript = record.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError(f"empty transcript: {record_id}")
    utterances = record.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError(f"utterances must be a list: {record_id}")
    for index, utterance in enumerate(utterances):
        if not isinstance(utterance, dict):
            raise ValueError(f"invalid utterance at {record_id}:{index}")
        if utterance.get("id") != f"utt-{index}":
            raise ValueError(f"non-contiguous evidence ID at {record_id}:{index}")
        start_ms = utterance.get("start_ms")
        end_ms = utterance.get("end_ms")
        if not isinstance(start_ms, int) or not isinstance(end_ms, int) or end_ms <= start_ms:
            raise ValueError(f"invalid timestamps at {record_id}:{index}")


def update_stats(stats: dict[str, object], record: dict[str, object]) -> None:
    stats["records"] += 1
    stats["transcript_chars"] += len(str(record["transcript"]))
    stats["reference_summary_chars"] += len(record["reference_summary"] or "")
    stats["utterances"] += len(record["utterances"])
    duration = record["duration_ms"]
    if isinstance(duration, int):
        stats["duration_ms"] += duration
    stats["by_dataset"][record["dataset"]] += 1
    stats["by_language"][record["language"]] += 1


def json_ready_stats(stats: dict[str, object]) -> dict[str, object]:
    return {
        "records": stats["records"],
        "transcript_chars": stats["transcript_chars"],
        "reference_summary_chars": stats["reference_summary_chars"],
        "utterances": stats["utterances"],
        "duration_ms": stats["duration_ms"],
        "by_dataset": dict(sorted(stats["by_dataset"].items())),
        "by_language": dict(sorted(stats["by_language"].items())),
    }


def new_stats() -> dict[str, object]:
    return {
        "records": 0,
        "transcript_chars": 0,
        "reference_summary_chars": 0,
        "utterances": 0,
        "duration_ms": 0,
        "by_dataset": Counter(),
        "by_language": Counter(),
    }


def write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_archive_checksums(data_root: Path) -> dict[str, str]:
    checksum_path = data_root / "AliMeeting/checksums/SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        checksum, recorded_path = line.split(maxsplit=1)
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"invalid AliMeeting checksum at {checksum_path}:{line_number}")
        archive = data_root / "AliMeeting/archives" / Path(recorded_path.strip()).name
        if not archive.is_file():
            raise FileNotFoundError(archive)
        relative = relative_path(archive, data_root)
        if relative in checksums:
            raise ValueError(f"duplicate AliMeeting archive checksum: {relative}")
        actual = sha256_file(archive)
        if actual != checksum:
            raise ValueError(
                f"AliMeeting archive checksum mismatch for {relative}: "
                f"expected={checksum} actual={actual}"
            )
        checksums[relative] = actual
    return dict(sorted(checksums.items()))


def validate_usage_scope(usage_scope: str) -> None:
    if usage_scope != ALLOWED_USAGE_SCOPE:
        raise ValueError(
            "MeetingBank is CC BY-NC-SA 4.0; this combined dataset is restricted "
            "to internal research until a separate license review is recorded"
        )


def build_dataset(
    data_root: Path,
    output_dir: Path,
    usage_scope: str = ALLOWED_USAGE_SCOPE,
) -> dict[str, object]:
    validate_usage_scope(usage_scope)
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    all_ids: set[str] = set()
    split_ids: dict[str, set[str]] = defaultdict(set)
    total_stats = new_stats()
    split_stats: dict[str, dict[str, object]] = {}

    try:
        for split in ("train", "validation", "test"):
            stats = new_stats()
            output_path = staging / f"{split}.jsonl"
            records: Iterable[dict[str, object]] = chain(
                iter_alimeeting(data_root, split),
                iter_meetingbank(data_root, split),
            )
            with output_path.open("w", encoding="utf-8", newline="\n") as output:
                for record in records:
                    validate_record(record)
                    record_id = str(record["record_id"])
                    if record_id in all_ids:
                        raise ValueError(f"record appears in multiple splits: {record_id}")
                    all_ids.add(record_id)
                    split_ids[split].add(record_id)
                    update_stats(stats, record)
                    update_stats(total_stats, record)
                    output.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
            split_stats[split] = json_ready_stats(stats)

        leakage = {
            "train_validation": sorted(split_ids["train"] & split_ids["validation"]),
            "train_test": sorted(split_ids["train"] & split_ids["test"]),
            "validation_test": sorted(split_ids["validation"] & split_ids["test"]),
        }
        if any(leakage.values()):
            raise ValueError(f"split leakage detected: {leakage}")

        stats_document = {
            "schema_version": SCHEMA_VERSION,
            "total": json_ready_stats(total_stats),
            "splits": split_stats,
        }
        write_json(staging / "stats.json", stats_document)
        validation_document = {
            "format": "meetnote.experimental.dataset_validation.v1",
            "records": len(all_ids),
            "checks": {
                "record_ids_unique": True,
                "split_leakage_absent": True,
                "source_transcripts_present": True,
                "alimeeting_far_and_near_audio_present": True,
                "alimeeting_timestamps_positive": True,
                "alimeeting_evidence_ids_contiguous": True,
                "meetingbank_source_boundaries_not_inferred": True,
            },
            "split_leakage": leakage,
        }
        write_json(staging / "validation-report.json", validation_document)

        meetingbank_sources = {
            split: {
                "path": relative,
                "size_bytes": (data_root / relative).stat().st_size,
                "sha256": sha256_file(data_root / relative),
            }
            for split, relative in MEETINGBANK_SPLITS.items()
        }
        output_files = {
            path.name: {
                "records": split_stats[path.stem]["records"] if path.suffix == ".jsonl" else None,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.name != "manifest.json"
        }
        manifest = {
            "format": MANIFEST_VERSION,
            "tool": {
                "path": "tools/prepare_meeting_datasets.py",
                "version": TOOL_VERSION,
            },
            "usage_scope": usage_scope,
            "contract": {
                "record_schema": SCHEMA_VERSION,
                "alimeeting_transcript_source": "official far-field TextGrid",
                "meetingbank_utterances": (
                    "empty because the source has no timing/speaker boundaries"
                ),
                "evidence_ids": "AliMeeting utterances use contiguous utt-0..utt-N",
                "text_policy": (
                    "source text is preserved; no semantic cleanup or inferred segmentation"
                ),
            },
            "sources": {
                "AliMeeting": {
                    "homepage": ALI_HOMEPAGE,
                    "license": ALI_LICENSE,
                    "archive_sha256": parse_archive_checksums(data_root),
                },
                "MeetingBank": {
                    "homepage": MEETINGBANK_HOMEPAGE,
                    "license": MEETINGBANK_LICENSE,
                    "files": meetingbank_sources,
                },
            },
            "split_leakage": leakage,
            "outputs": output_files,
        }
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--usage-scope",
        choices=("internal-research", "commercial-product"),
        default=ALLOWED_USAGE_SCOPE,
    )
    args = parser.parse_args()
    try:
        manifest = build_dataset(args.data_root, args.output_dir, args.usage_scope)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"output_dir": str(args.output_dir.resolve()), **manifest["outputs"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
