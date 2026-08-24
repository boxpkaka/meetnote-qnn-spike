#!/usr/bin/env python3
"""Compare the pinned HF processor and MNN runtime audio-token contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from moss_runtime import AUDIO_SAMPLE_STRIDE, build_audio_span_ids


HF_FULL_PROMPT_REFERENCES = {
    480_000: (472, "8575d09f106e71428d5e350cfb54ce0b12655f025b178f573de536d934887a16"),
    960_000: (859, "1be9e0baa4d14e3092bcf38b46df79634f7aae10e5ae80a06d36ee253617520a"),
    1_440_000: (1246, "fd121c7fa0518d2e0b42679598f21510eab2e5431d206bc3a86f4d0ace80c2b8"),
    1_920_000: (1638, "ab36457539d11cec128ed5b96bd28d91fa3e7c6b414949ad0512d17b501870d5"),
    4_800_000: (3996, "5404384459d2711c83d9198a97fb7607efa44adb6fd3f4a5b60d41f5e40bdadc"),
}


def token_id(tokenizer: dict, token: str) -> int:
    for record in tokenizer.get("added_tokens", []):
        if record.get("content") == token:
            return int(record["id"])
    value = tokenizer.get("model", {}).get("vocab", {}).get(token)
    if value is None:
        raise ValueError(f"tokenizer is missing {token!r}")
    return int(value)


def sequence_sha256(ids: list[int]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(struct.pack("<q", value))
    return digest.hexdigest()


def source_contract(model_dir: Path) -> dict[str, object]:
    processor = json.loads((model_dir / "processor_config.json").read_text(encoding="utf-8"))
    model = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    return {
        "audio_pad": int(model["audio_token_id"]),
        "audio_start": token_id(tokenizer, "<|audio_start|>"),
        "audio_end": token_id(tokenizer, "<|audio_end|>"),
        "audio_merge_size": int(processor["audio_merge_size"]),
        "audio_tokens_per_second": float(processor["audio_tokens_per_second"]),
        "time_marker_every_seconds": int(processor["time_marker_every_seconds"]),
        "enable_time_marker": bool(processor["enable_time_marker"]),
        "digit_token_ids": {digit: token_id(tokenizer, digit) for digit in "0123456789"},
    }


def runtime_contract(llm_config: Path) -> dict[str, object]:
    config = json.loads(llm_config.read_text(encoding="utf-8"))
    keys = (
        "audio_pad",
        "audio_start",
        "audio_end",
        "audio_merge_size",
        "audio_tokens_per_second",
        "time_marker_every_seconds",
    )
    return {key: config.get(key) for key in keys}


def audio_span(contract: dict[str, object], samples: int) -> list[int]:
    audio_tokens = (samples - 1) // AUDIO_SAMPLE_STRIDE + 1
    marker_seconds = int(contract["time_marker_every_seconds"])
    if contract.get("enable_time_marker", True) is False:
        marker_seconds = 0
    return build_audio_span_ids(
        audio_tokens,
        int(contract["audio_pad"]),
        {str(key): int(value) for key, value in contract["digit_token_ids"].items()},
        marker_seconds=marker_seconds,
    )


def time_markers(contract: dict[str, object], samples: int) -> list[dict[str, object]]:
    marker_seconds = int(contract["time_marker_every_seconds"])
    if contract.get("enable_time_marker", True) is False or marker_seconds <= 0:
        return []
    audio_tokens = (samples - 1) // AUDIO_SAMPLE_STRIDE + 1
    tokens_per_marker = int(float(contract["audio_tokens_per_second"]) * marker_seconds)
    duration = audio_tokens / float(contract["audio_tokens_per_second"])
    digit_ids = {str(key): int(value) for key, value in contract["digit_token_ids"].items()}
    records = []
    inserted = 0
    for second in range(marker_seconds, int(duration) + 1, marker_seconds):
        ids = [digit_ids[digit] for digit in str(second)]
        audio_position = (second // marker_seconds) * tokens_per_marker
        records.append(
            {
                "second": second,
                "audio_embedding_position": audio_position,
                "expanded_span_index": audio_position + inserted,
                "token_ids": ids,
            }
        )
        inserted += len(ids)
    return records


def verify(
    model_dir: Path,
    llm_config: Path,
    durations: list[int],
    full_prompt_reports: list[Path] | None = None,
) -> dict[str, object]:
    source = source_contract(model_dir)
    runtime = runtime_contract(llm_config)
    comparable = {key: value for key, value in source.items() if key in runtime}
    mismatches = {
        key: {"hf": expected, "runtime": runtime.get(key)}
        for key, expected in comparable.items()
        if runtime.get(key) != expected
    }
    cases = []
    for seconds in durations:
        samples = seconds * 16_000
        span = audio_span(source, samples)
        multimodal_ids = [int(source["audio_start"]), *span, int(source["audio_end"])]
        cases.append(
            {
                "duration_seconds": seconds,
                "sample_count": samples,
                "audio_embedding_count": span.count(int(source["audio_pad"])),
                "audio_span_token_count": len(span),
                "multimodal_token_count": len(multimodal_ids),
                "multimodal_ids_sha256": sequence_sha256(multimodal_ids),
                "first_marker_ids": [
                    value for value in span if value != int(source["audio_pad"])
                ][:12],
                "last_marker_ids": [
                    value for value in span if value != int(source["audio_pad"])
                ][-12:],
                "time_markers": time_markers(source, samples),
            }
        )
    full_prompt = []
    report_errors = []
    for path in full_prompt_reports or []:
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = {
            key: report.get(key)
            for key in (
                "format",
                "status",
                "sample_count",
                "audio_embedding_count",
                "official_token_count",
                "runtime_token_count",
                "first_mismatch",
            )
        }
        summary["path"] = str(path.resolve())
        summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        official_ids = report.get("official_input_ids", [])
        summary["official_input_ids_sha256"] = sequence_sha256(official_ids)
        full_prompt.append(summary)
        if report.get("format") != "meetnote.moss_full_prompt_contract.v1":
            report_errors.append(f"unsupported full prompt report: {path}")
        elif report.get("status") != "valid" or report.get("official_input_ids") != report.get("runtime_input_ids"):
            report_errors.append(f"full prompt token mismatch: {path}")
        else:
            sample_count = int(report.get("sample_count", -1))
            reference = HF_FULL_PROMPT_REFERENCES.get(sample_count)
            if reference is None:
                report_errors.append(f"no pinned HF full prompt reference for {sample_count} samples: {path}")
            elif (len(official_ids), summary["official_input_ids_sha256"]) != reference:
                report_errors.append(
                    f"full prompt differs from pinned HF reference: {path}; "
                    f"expected={reference} actual={(len(official_ids), summary['official_input_ids_sha256'])}"
                )
    expected_samples = {seconds * 16_000 for seconds in durations}
    actual_samples = {int(row["sample_count"]) for row in full_prompt}
    if full_prompt and actual_samples != expected_samples:
        report_errors.append(
            f"full prompt duration coverage differs: expected={sorted(expected_samples)} actual={sorted(actual_samples)}"
        )

    result = {
        "format": "meetnote.moss_audio_token_contract.v1",
        "scope": "audio boundary, placeholder, time-marker, and pinned HF full-prompt token IDs",
        "status": "valid" if not mismatches and not report_errors else "invalid",
        "hf_processor": source,
        "mnn_runtime": runtime,
        "mismatches": mismatches,
        "cases": cases,
        "full_prompt": full_prompt,
        "full_prompt_errors": report_errors,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--llm-config", type=Path, required=True)
    parser.add_argument("--durations", type=int, nargs="+", default=[30, 60, 90, 120])
    parser.add_argument("--full-prompt-reports", type=Path, nargs="*")
    args = parser.parse_args()
    result = verify(args.model_dir, args.llm_config, args.durations, args.full_prompt_reports)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
