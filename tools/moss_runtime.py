#!/usr/bin/env python3
"""Host-side contracts shared by the MOSS QNN PoC tools.

The QNN runner owns inference, while this module owns the parts which must stay
bit-for-bit aligned with the official processor: fixed 30 second chunking,
audio placeholder expansion, time markers, input limits, and transcript
parsing.  It intentionally has no import-time dependency on Torch or NumPy.
"""

from __future__ import annotations

import dataclasses
import math
import re
import wave
from pathlib import Path
from typing import Callable, Sequence

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
MAX_AUDIO_SECONDS = 300
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
MEL_BINS = 80
MEL_FRAMES = 3_000
WHISPER_ENCODER_STRIDE = 2
AUDIO_MERGE_SIZE = 4
AUDIO_SAMPLE_STRIDE = 160 * WHISPER_ENCODER_STRIDE * AUDIO_MERGE_SIZE
AUDIO_TOKENS_PER_SECOND = 12.5
TIME_MARKER_SECONDS = 2
PREFILL_CHUNK = 64
MAX_SEQUENCE_TOKENS = 8_192
MAX_NEW_TOKENS = 4_096
EOS_TOKEN_ID = 151_645
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


class MossInputError(ValueError):
    """An input violates the fixed PoC contract."""


@dataclasses.dataclass(frozen=True)
class AudioChunk:
    index: int
    start_sample: int
    valid_samples: int
    audio_tokens: int


@dataclasses.dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str


_SEGMENT = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]"
    r"\[(?P<speaker>S\d+)\]"
    r"(?P<text>.*?)"
    r"\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL,
)


def plan_audio_chunks(num_samples: int) -> list[AudioChunk]:
    """Return the official fixed-shape chunk plan for one mono waveform."""
    if num_samples <= 0:
        raise MossInputError("audio is empty")
    if num_samples > MAX_AUDIO_SAMPLES:
        raise MossInputError(
            f"audio exceeds {MAX_AUDIO_SECONDS} seconds: "
            f"samples={num_samples} sample_rate={SAMPLE_RATE}"
        )
    chunks = []
    for index, start in enumerate(range(0, num_samples, CHUNK_SAMPLES)):
        valid = min(CHUNK_SAMPLES, num_samples - start)
        chunks.append(
            AudioChunk(
                index=index,
                start_sample=start,
                valid_samples=valid,
                audio_tokens=math.ceil(valid / AUDIO_SAMPLE_STRIDE),
            )
        )
    return chunks


def build_audio_span_ids(
    audio_token_count: int,
    audio_token_id: int,
    digit_token_ids: dict[str, int],
    *,
    marker_seconds: int = TIME_MARKER_SECONDS,
) -> list[int]:
    """Match the official processor's marker-separated audio token span."""
    if audio_token_count < 0:
        raise MossInputError("audio token count cannot be negative")
    if set(digit_token_ids) != set("0123456789"):
        raise MossInputError("digit_token_ids must contain exactly 0 through 9")
    if audio_token_count == 0 or marker_seconds <= 0:
        return [audio_token_id] * audio_token_count

    tokens_per_marker = int(AUDIO_TOKENS_PER_SECOND * marker_seconds)
    duration = audio_token_count / AUDIO_TOKENS_PER_SECOND
    output: list[int] = []
    consumed = 0
    for second in range(marker_seconds, int(duration) + 1, marker_seconds):
        position = (second // marker_seconds) * tokens_per_marker
        output.extend([audio_token_id] * (position - consumed))
        consumed = position
        output.extend(digit_token_ids[digit] for digit in str(second))
    output.extend([audio_token_id] * (audio_token_count - consumed))
    return output


def expand_audio_placeholder(
    prompt: str,
    *,
    audio_token: str,
    audio_token_count: int,
    audio_token_id: int,
    digit_token_ids: dict[str, int],
    encode: Callable[[str], Sequence[int]],
    max_sequence_tokens: int = MAX_SEQUENCE_TOKENS,
) -> list[int]:
    """Expand the single placeholder without retokenizing inserted markers."""
    if prompt.count(audio_token) != 1:
        raise MossInputError(f"prompt must contain exactly one {audio_token!r}")
    before, after = prompt.split(audio_token, maxsplit=1)
    ids = [
        *map(int, encode(before)),
        *build_audio_span_ids(audio_token_count, audio_token_id, digit_token_ids),
        *map(int, encode(after)),
    ]
    if len(ids) > max_sequence_tokens:
        raise MossInputError(
            f"prompt/audio sequence exceeds {max_sequence_tokens} tokens: {len(ids)}"
        )
    return ids


def audio_embedding_positions(input_ids: Sequence[int], audio_token_id: int) -> list[int]:
    """Return scatter positions in order, including spans split by markers."""
    return [index for index, token_id in enumerate(input_ids) if int(token_id) == audio_token_id]


def validate_embedding_count(
    input_ids: Sequence[int], audio_token_id: int, embedding_count: int
) -> list[int]:
    positions = audio_embedding_positions(input_ids, audio_token_id)
    if len(positions) != embedding_count:
        raise MossInputError(
            "audio placeholder/embedding count mismatch: "
            f"placeholders={len(positions)} embeddings={embedding_count}"
        )
    return positions


def logits_symmetric_range(max_abs: float) -> float:
    """Return the release quantization range required by the calibration policy."""
    if not math.isfinite(max_abs) or max_abs < 0:
        raise ValueError("calibration max_abs must be finite and non-negative")
    required = max(64.0, max_abs * 1.1)
    return float(2 ** math.ceil(math.log2(required)))


def parse_transcript(raw_text: str) -> tuple[list[TranscriptSegment], str | None]:
    """Parse complete MOSS segments while preserving raw output on failure."""
    segments: list[TranscriptSegment] = []
    for match in _SEGMENT.finditer(raw_text):
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end < start:
            continue
        text = match.group("text").strip()
        if text:
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    speaker=match.group("speaker"),
                    text=text,
                )
            )
    error = None if segments else "transcript_parse_failed"
    return segments, error


def read_wav_info(path: Path) -> tuple[int, int, int]:
    """Return (sample_rate, channels, frame_count) for an uncompressed WAV."""
    try:
        with wave.open(str(path), "rb") as source:
            if source.getcomptype() != "NONE":
                raise MossInputError(f"compressed WAV is unsupported: {source.getcomptype()}")
            return source.getframerate(), source.getnchannels(), source.getnframes()
    except (OSError, wave.Error) as exc:
        raise MossInputError(f"cannot read WAV {path}: {exc}") from exc


def estimated_resampled_samples(sample_rate: int, frame_count: int) -> int:
    if sample_rate <= 0 or frame_count < 0:
        raise MossInputError("invalid WAV sample rate or frame count")
    return math.ceil(frame_count * SAMPLE_RATE / sample_rate)


def validate_wav(path: Path) -> list[AudioChunk]:
    sample_rate, channels, frames = read_wav_info(path)
    if channels <= 0:
        raise MossInputError("WAV has no channels")
    return plan_audio_chunks(estimated_resampled_samples(sample_rate, frames))


def result_document(
    *,
    raw_text: str,
    timings_ms: dict[str, float],
    duration_seconds: float,
    peak_pss_bytes: int | None,
    generated_tokens: int,
) -> dict[str, object]:
    segments, parse_error = parse_transcript(raw_text)
    errors = []
    if parse_error:
        errors.append(parse_error)
    if generated_tokens >= MAX_NEW_TOKENS:
        errors.append("token_budget_exhausted")
    total_ms = float(timings_ms.get("total", 0.0))
    return {
        "format": "meetnote.moss_qnn_result.v1",
        "status": "ok" if not errors else "error",
        "errors": errors,
        "raw_text": raw_text,
        "segments": [dataclasses.asdict(segment) for segment in segments],
        "generated_tokens": generated_tokens,
        "timings_ms": timings_ms,
        "rtf": total_ms / 1000.0 / duration_seconds if duration_seconds > 0 else None,
        "peak_pss_bytes": peak_pss_bytes,
    }
