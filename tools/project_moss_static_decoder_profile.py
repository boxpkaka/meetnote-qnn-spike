#!/usr/bin/env python3
"""Project end-to-end decode time from measured static decoder layer latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def project(profile: dict, plan: dict, reference: dict | None = None) -> dict:
    buckets = profile["decode_profile"]["history_buckets"]
    cpu_attention_calls = sum(item["cpu_attention_calls"] for item in buckets)
    planned_calls = sum(item["layer_calls"] for item in plan["segments"])
    if planned_calls != cpu_attention_calls:
        raise ValueError(
            f"plan covers {planned_calls} layer calls, profile has {cpu_attention_calls}"
        )

    layer_ms = sum(
        item["layer_calls"] * item["latency_ms_per_layer"]
        for item in plan["segments"]
    )
    remaining_qnn_execute_ms = 0.0
    remaining_qnn_calls = 0
    retained_qnn_io_state_ms = 0.0
    for item in buckets:
        qnn_calls = item["qnn_graph_calls"]
        layer_calls = item["cpu_attention_calls"]
        if qnn_calls < layer_calls:
            raise ValueError("QNN graph calls cannot be fewer than decoder layer calls")
        remaining_calls = qnn_calls - layer_calls
        remaining_qnn_calls += remaining_calls
        if qnn_calls:
            remaining_qnn_execute_ms += (
                item["qnn_execute_sync_ms"] * remaining_calls / qnn_calls
            )
        retained_qnn_io_state_ms += sum(
            item[name]
            for name in (
                "qnn_input_copy_ms",
                "qnn_output_copy_ms",
                "qnn_state_update_ms",
            )
        )

    unaccounted_ms = profile["decode_profile"]["unaccounted_ms"]
    projected_decode_ms = (
        layer_ms
        + remaining_qnn_execute_ms
        + retained_qnn_io_state_ms
        + unaccounted_ms
    )
    measured_total_ms = profile["timings_ms"]["total"]
    measured_decode_ms = profile["timings_ms"]["decode"]
    fixed_non_decode_ms = measured_total_ms - measured_decode_ms
    audio_duration_ms = plan["audio_duration_ms"]
    projected_total_ms = fixed_non_decode_ms + projected_decode_ms

    result = {
        "plan": plan["name"],
        "layer_count": plan["layer_count"],
        "cpu_attention_calls_replaced": cpu_attention_calls,
        "remaining_qnn_graph_calls": remaining_qnn_calls,
        "components_ms": {
            "static_decoder_layers": layer_ms,
            "remaining_qnn_execute_sync": remaining_qnn_execute_ms,
            "retained_qnn_io_state": retained_qnn_io_state_ms,
            "unaccounted": unaccounted_ms,
        },
        "measured_baseline": {
            "fixed_non_decode_ms": fixed_non_decode_ms,
            "projected_decode_ms": projected_decode_ms,
            "projected_total_ms": projected_total_ms,
            "projected_rtf": projected_total_ms / audio_duration_ms,
        },
    }
    if reference is not None:
        scale = reference["decode_ms"] / measured_decode_ms
        conservative_decode_ms = projected_decode_ms * scale
        conservative_fixed_ms = reference["total_ms"] - reference["decode_ms"]
        conservative_total_ms = conservative_fixed_ms + conservative_decode_ms
        result["conservative_baseline"] = {
            "decode_scale": scale,
            "fixed_non_decode_ms": conservative_fixed_ms,
            "projected_decode_ms": conservative_decode_ms,
            "projected_total_ms": conservative_total_ms,
            "projected_rtf": conservative_total_ms / audio_duration_ms,
        }
    result["assumptions"] = plan.get("assumptions", [])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-total-ms", type=float)
    parser.add_argument("--reference-decode-ms", type=float)
    args = parser.parse_args()

    if (args.reference_total_ms is None) != (args.reference_decode_ms is None):
        parser.error("reference total and decode times must be supplied together")
    reference = None
    if args.reference_total_ms is not None:
        reference = {
            "total_ms": args.reference_total_ms,
            "decode_ms": args.reference_decode_ms,
        }
    result = project(
        json.loads(args.profile.read_text()),
        json.loads(args.plan.read_text()),
        reference,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
