#!/usr/bin/env python3
"""Set QNN logits tensors to a symmetric INT16 range (default ``[-64, 64]``)."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

TARGET_TENSORS = {
    "/lm/lm_head/Linear",
    "/lm/lm_head/Linear/post_convert",
    "logits",
}
DEFAULT_RANGE = 64.0
TARGET_ZERO = -32768.0


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def dump_mnn(mnn_convert: Path, model: Path, output: Path) -> None:
    run(
        [
            str(mnn_convert),
            "-f",
            "MNN",
            "--modelFile",
            str(model),
            "--JsonFile",
            str(output),
        ]
    )


def build_mnn(mnn_convert: Path, source: Path, output: Path) -> None:
    run(
        [
            str(mnn_convert),
            "-f",
            "JSON",
            "--modelFile",
            str(source),
            "--MNNModel",
            str(output),
        ]
    )


def patch_document(document: dict, target_range: float = DEFAULT_RANGE) -> None:
    target_scale = (2.0 * target_range) / 65535.0
    names = document["tensorName"]
    matched: set[str] = set()
    for description in document["extraTensorDescribe"]:
        index = int(description["index"])
        name = names[index]
        if name not in TARGET_TENSORS:
            continue
        quant = description.get("quantInfo")
        if not isinstance(quant, dict) or quant.get("type") != "DT_INT16":
            raise ValueError(f"{name} is not an INT16 quantized tensor")
        quant["scale"] = target_scale
        quant["zero"] = TARGET_ZERO
        matched.add(name)
    if matched != TARGET_TENSORS:
        raise ValueError(f"logits tensor set differs: found={sorted(matched)}")


def validate_document(document: dict, target_range: float = DEFAULT_RANGE) -> None:
    target_scale = (2.0 * target_range) / 65535.0
    names = document["tensorName"]
    matched: set[str] = set()
    for description in document["extraTensorDescribe"]:
        name = names[int(description["index"])]
        if name not in TARGET_TENSORS:
            continue
        quant = description["quantInfo"]
        if not math.isclose(float(quant["scale"]), target_scale, rel_tol=5e-4):
            raise ValueError(f"{name} scale differs after MNN round-trip")
        if float(quant["zero"]) != TARGET_ZERO:
            raise ValueError(f"{name} zero point differs after MNN round-trip")
        matched.add(name)
    if matched != TARGET_TENSORS:
        raise ValueError("widened logits tensors are missing after MNN round-trip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnn-convert", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--range", type=float, default=DEFAULT_RANGE, dest="target_range")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_range < DEFAULT_RANGE or not math.isfinite(args.target_range):
        raise ValueError(f"logits range must be finite and >= {DEFAULT_RANGE}")
    if not math.log2(args.target_range).is_integer():
        raise ValueError("logits range must be a power of two")
    target_scale = (2.0 * args.target_range) / 65535.0
    model = args.model.resolve()
    with tempfile.TemporaryDirectory(
        prefix=".meetnote-wide-logits-", dir=model.parent
    ) as directory:
        temporary = Path(directory)
        source_json = temporary / "source.json"
        output_model = temporary / "llm.mnn"
        output_json = temporary / "output.json"
        dump_mnn(args.mnn_convert, model, source_json)
        document = json.loads(source_json.read_text(encoding="utf-8"))
        patch_document(document, args.target_range)
        source_json.write_text(
            json.dumps(document, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        build_mnn(args.mnn_convert, source_json, output_model)
        dump_mnn(args.mnn_convert, output_model, output_json)
        validate_document(
            json.loads(output_json.read_text(encoding="utf-8")), args.target_range
        )
        os.replace(output_model, model)
    print(
        json.dumps(
            {
                "model": str(model),
                "tensors": sorted(TARGET_TENSORS),
                "range": [-args.target_range, args.target_range],
                "scale": target_scale,
                "zero": TARGET_ZERO,
                "status": "valid",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
