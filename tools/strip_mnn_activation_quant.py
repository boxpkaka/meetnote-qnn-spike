#!/usr/bin/env python3
"""Remove tensor activation quantization metadata while preserving quantized weights."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def dump_mnn(converter: Path, model: Path, output: Path) -> None:
    run([str(converter), "-f", "MNN", "--modelFile", str(model), "--JsonFile", str(output)])


def build_mnn(converter: Path, source: Path, output: Path) -> None:
    run([str(converter), "-f", "JSON", "--modelFile", str(source), "--MNNModel", str(output)])


def strip_activation_quant(document: dict[str, object]) -> int:
    descriptions = document.get("extraTensorDescribe", [])
    if not isinstance(descriptions, list):
        raise ValueError("extraTensorDescribe must be a list")
    document["extraTensorDescribe"] = []
    return len(descriptions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnn-convert", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    with tempfile.TemporaryDirectory(prefix=".meetnote-strip-aquant-", dir=model.parent) as directory:
        temporary = Path(directory)
        source_json = temporary / "source.json"
        output_model = temporary / "llm.mnn"
        output_json = temporary / "output.json"
        dump_mnn(args.mnn_convert, model, source_json)
        document = json.loads(source_json.read_text(encoding="utf-8"))
        removed = strip_activation_quant(document)
        source_json.write_text(
            json.dumps(document, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        build_mnn(args.mnn_convert, source_json, output_model)
        dump_mnn(args.mnn_convert, output_model, output_json)
        round_trip = json.loads(output_json.read_text(encoding="utf-8"))
        if round_trip.get("extraTensorDescribe"):
            raise ValueError("activation quantization metadata remains after round trip")
        os.replace(output_model, model)

    print(json.dumps({"model": str(model), "removed": removed, "status": "valid"}, indent=2))


if __name__ == "__main__":
    main()
