#!/usr/bin/env python3
"""Analyze a MOSS decode profile against a real-time latency budget."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROFILE_COMPONENTS = (
    "cpu_attention_ms",
    "qnn_input_copy_ms",
    "qnn_execute_sync_ms",
    "qnn_output_copy_ms",
    "qnn_state_update_ms",
)


def _finite_nonnegative(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def analyze(
    result: dict[str, Any],
    *,
    audio_duration_ms: float,
    budget_result: dict[str, Any] | None = None,
    attention_speedups: tuple[float, ...] = (3.0, 5.0, 10.0),
    cross_runtime_overheads_ms: tuple[float, ...] = (0.0, 10_000.0, 30_000.0, 50_000.0),
) -> dict[str, Any]:
    if result.get("format") != "meetnote.moss_qnn_result.v1":
        raise ValueError("unsupported MOSS result format")
    timings = result.get("timings_ms")
    profile = result.get("decode_profile")
    if not isinstance(timings, dict) or not isinstance(profile, dict):
        raise ValueError("result is missing timings_ms or decode_profile")
    buckets = profile.get("history_buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("decode_profile.history_buckets must be a non-empty list")

    duration_ms = _finite_nonnegative(audio_duration_ms, "audio_duration_ms")
    measured_decode_ms = _finite_nonnegative(timings.get("decode"), "timings_ms.decode")
    if budget_result is None:
        budget_timings = timings
    else:
        if budget_result.get("format") != "meetnote.moss_qnn_result.v1":
            raise ValueError("unsupported budget result format")
        budget_timings = budget_result.get("timings_ms")
        if not isinstance(budget_timings, dict):
            raise ValueError("budget result is missing timings_ms")
    decode_ms = _finite_nonnegative(budget_timings.get("decode"), "budget decode")
    total_ms = _finite_nonnegative(budget_timings.get("total"), "budget total")
    if measured_decode_ms == 0 or decode_ms == 0 or duration_ms == 0 or total_ms < decode_ms:
        raise ValueError("timings must contain positive decode/audio duration and total >= decode")

    measured_component_totals = {
        name: sum(_finite_nonnegative(bucket.get(name, 0), name) for bucket in buckets)
        for name in PROFILE_COMPONENTS
    }
    timing_scale = decode_ms / measured_decode_ms
    component_totals = {
        name: value * timing_scale for name, value in measured_component_totals.items()
    }
    cpu_attention_ms = component_totals["cpu_attention_ms"]
    non_attention_decode_ms = decode_ms - cpu_attention_ms
    if non_attention_decode_ms < 0:
        raise ValueError("profiled CPU attention exceeds decode time")

    fixed_ms = total_ms - decode_ms
    decode_budget_ms = duration_ms - fixed_ms
    if decode_budget_ms <= 0:
        required_overall_speedup = math.inf
    else:
        required_overall_speedup = decode_ms / decode_budget_ms

    projections = []
    for speedup in attention_speedups:
        speedup = _finite_nonnegative(speedup, "attention_speedup")
        if speedup <= 0:
            raise ValueError("attention_speedup must be positive")
        projected_decode_ms = non_attention_decode_ms + cpu_attention_ms / speedup
        projected_total_ms = fixed_ms + projected_decode_ms
        projections.append(
            {
                "attention_speedup": speedup,
                "projected_decode_ms": projected_decode_ms,
                "projected_total_ms": projected_total_ms,
                "projected_rtf": projected_total_ms / duration_ms,
                "meets_rtf_1": projected_total_ms <= duration_ms,
            }
        )

    overhead_scenarios = []
    for overhead_ms in cross_runtime_overheads_ms:
        overhead_ms = _finite_nonnegative(overhead_ms, "cross_runtime_overhead_ms")
        remaining_attention_budget_ms = (
            decode_budget_ms - non_attention_decode_ms - overhead_ms
        )
        possible = remaining_attention_budget_ms > 0
        overhead_scenarios.append(
            {
                "cross_runtime_overhead_ms": overhead_ms,
                "attention_only_can_meet_rtf_1": possible,
                "required_attention_speedup": (
                    cpu_attention_ms / remaining_attention_budget_ms if possible else None
                ),
            }
        )

    accounted_ms = sum(component_totals.values())
    return {
        "format": "meetnote.moss_decode_profile_analysis.v1",
        "audio_duration_ms": duration_ms,
        "profile_source": {
            "measured_decode_ms": measured_decode_ms,
            "budget_decode_ms": decode_ms,
            "component_timing_scale": timing_scale,
            "used_separate_budget_result": budget_result is not None,
        },
        "timing": {
            "total_ms": total_ms,
            "fixed_non_decode_ms": fixed_ms,
            "decode_ms": decode_ms,
            "decode_budget_for_rtf_1_ms": decode_budget_ms,
            "required_overall_decode_speedup": required_overall_speedup,
        },
        "decode_breakdown": {
            **component_totals,
            "cpu_attention_compute_ms": timing_scale * sum(
                _finite_nonnegative(bucket.get("cpu_attention_compute_ms", 0), "cpu_attention_compute_ms")
                for bucket in buckets
            ),
            "cpu_attention_kv_update_ms": timing_scale * sum(
                _finite_nonnegative(bucket.get("cpu_attention_kv_update_ms", 0), "cpu_attention_kv_update_ms")
                for bucket in buckets
            ),
            "profile_accounted_ms": accounted_ms,
            "profile_unaccounted_ms": decode_ms - accounted_ms,
            "cpu_attention_share": cpu_attention_ms / decode_ms,
            "qnn_execute_sync_share": component_totals["qnn_execute_sync_ms"] / decode_ms,
            "non_attention_decode_ms": non_attention_decode_ms,
        },
        "theoretical": {
            "minimum_attention_share_for_zero_cost_migration": max(
                0.0, 1.0 - decode_budget_ms / decode_ms
            ),
            "total_ms_if_attention_were_free": fixed_ms + non_attention_decode_ms,
            "rtf_if_attention_were_free": (fixed_ms + non_attention_decode_ms) / duration_ms,
        },
        "attention_speedup_projections": projections,
        "cross_runtime_overhead_scenarios": overhead_scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--audio-duration-sec", type=float, required=True)
    parser.add_argument(
        "--budget-result",
        type=Path,
        help="Use another same-input result's total/decode timings and scale measured shares to it.",
    )
    parser.add_argument(
        "--attention-speedup",
        type=float,
        action="append",
        dest="attention_speedups",
        help="Candidate CPU Attention speedup; may be repeated.",
    )
    parser.add_argument(
        "--cross-runtime-overhead-ms",
        type=float,
        action="append",
        dest="cross_runtime_overheads_ms",
        help="Candidate added runtime overhead; may be repeated.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        json.loads(args.result.read_text(encoding="utf-8")),
        audio_duration_ms=args.audio_duration_sec * 1000.0,
        budget_result=(
            json.loads(args.budget_result.read_text(encoding="utf-8"))
            if args.budget_result
            else None
        ),
        attention_speedups=tuple(args.attention_speedups or (3.0, 5.0, 10.0)),
        cross_runtime_overheads_ms=tuple(
            args.cross_runtime_overheads_ms or (0.0, 10_000.0, 30_000.0, 50_000.0)
        ),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
