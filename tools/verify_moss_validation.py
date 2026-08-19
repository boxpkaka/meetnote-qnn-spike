#!/usr/bin/env python3
"""Validate the complete MOSS QNN PoC promotion evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative_regression(candidate: float, reference: float) -> float:
    if reference <= 0:
        return 0.0 if candidate <= reference else float("inf")
    return (candidate - reference) / reference


def validate(document: dict) -> dict[str, object]:
    require(
        document.get("format") == "meetnote.moss_qnn_validation.v1",
        "unsupported MOSS validation format",
    )
    require(
        document.get("release_id") == "moss-transcribe-diarize-0.9b-sm8850-v81-poc-v1",
        "wrong release_id",
    )

    component = document["component_alignment"]
    log_mel = component["log_mel"]
    require(log_mel["shape_exact"], "log-mel shape differs from HF")
    require(log_mel["all_finite"], "log-mel contains non-finite values")
    require(math.isfinite(log_mel["cosine"]), "log-mel cosine is non-finite")
    require(log_mel["cosine"] >= 0.9999, "log-mel cosine below 0.9999")
    require(math.isfinite(log_mel["mean_abs_error"]), "log-mel MAE is non-finite")
    require(log_mel["mean_abs_error"] <= 1e-3, "log-mel MAE exceeds 1e-3")
    require(math.isfinite(log_mel["max_abs_error"]), "log-mel max error is non-finite")
    require(component["processor_tokens"]["single_chunk_exact"], "single-chunk token IDs differ")
    require(component["processor_tokens"]["multi_chunk_exact"], "multi-chunk token IDs differ")
    require(component["processor_tokens"]["prompt_exact"], "full prompt token IDs differ")
    audio = component["audio"]
    require(
        audio["cpu_frontend_hf_bf16_cosine"] >= 0.995,
        "CPU frontend audio embedding cosine below 0.995",
    )
    require(audio["hf_bf16_qnn_cosine"] >= 0.995, "QNN audio cosine below 0.995")
    decoder = component["decoder"]
    require(decoder["qnn_mnn_top1_agreement"] >= 0.99, "decoder top-1 agreement below 99%")
    require(decoder["qnn_mnn_logits_cosine"] >= 0.98, "decoder logits cosine below 0.98")
    require(not decoder["logits_saturation"], "decoder logits are saturated")
    require(
        decoder["comparison_layers"] == ["hf_bf16", "mnn_cpu", "qnn_htp"],
        "decoder comparison must retain all three layers",
    )

    quality = document["quality"]
    require(quality["case_count"] == 10, "quality set must contain 10 cases")
    require(quality["all_duration_seconds"] == 300, "quality cases must be fixed 5-minute clips")
    require(quality["all_outputs_parseable"], "one or more quality outputs are unparseable")
    require(
        relative_regression(quality["qnn"]["cer"], quality["hf_bf16"]["cer"]) <= 0.05,
        "CER regression exceeds 5%",
    )
    require(
        relative_regression(quality["qnn"]["cpcer"], quality["hf_bf16"]["cpcer"]) <= 0.05,
        "cpCER regression exceeds 5%",
    )
    require(
        quality["qnn"]["timestamp_boundary_mae_seconds"]
        - quality["hf_bf16"]["timestamp_boundary_mae_seconds"]
        <= 0.1,
        "timestamp boundary MAE regression exceeds 100 ms",
    )
    require(len(quality["cases"]) == 10, "quality per-case evidence is incomplete")
    for case in quality["cases"]:
        require("speaker_count" in case, "speaker count evidence missing")
        require("missed_speakers" in case, "missed speaker evidence missing")
        require("merged_speakers" in case, "merged speaker evidence missing")

    device = document["device"]
    require(device["soc_id"] == 87 and device["dsp_arch"] == "v81", "wrong device target")
    require(device["qairt_version"] == "2.48.40.260702", "wrong QAIRT version")
    require(device["p95_rtf"] <= 1.0, "device p95 RTF exceeds 1.0")
    require(device["peak_pss_bytes"] <= 4 * 1024**3, "device peak PSS exceeds 4 GiB")
    failures = (
        "htp_ssr",
        "oom",
        "crash",
        "non_finite_output",
        "shared_weight_mapping_failure",
    )
    for failure in failures:
        require(not device[failure], f"device failure recorded: {failure}")
    timings = (
        "audio_encoder_front",
        "audio_encoder_back",
        "prefill",
        "decode",
        "cpu_preprocess",
    )
    for timing in timings:
        require(timing in device["timings_ms"], f"missing device timing: {timing}")
    return {"status": "valid", "release_id": document["release_id"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.validation.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
