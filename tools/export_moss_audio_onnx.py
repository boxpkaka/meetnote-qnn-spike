#!/usr/bin/env python3
"""Export only the fixed-shape MOSS audio front/back ONNX graphs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnn-export-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    export_root = args.mnn_export_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "onnx").mkdir(exist_ok=True)

    os.chdir(export_root)
    sys.path.insert(0, str(export_root))
    from llmexport import LlmExporter, build_args  # noqa: PLC0415

    exporter_parser = argparse.ArgumentParser()
    build_args(exporter_parser)
    exporter_args = exporter_parser.parse_args(
        [
            "--path",
            str(args.model_dir.resolve()),
            "--tokenizer_path",
            str(args.model_dir.resolve()),
            "--dst_path",
            str(output_dir),
            "--quant_bit",
            "16",
            "--lm_quant_bit",
            "16",
        ]
    )
    exporter = LlmExporter(exporter_args)
    if exporter.audio is None:
        raise RuntimeError("the loaded model has no audio encoder")
    paths = exporter.audio.export(str(output_dir / "onnx"))
    for path in paths if isinstance(paths, (list, tuple)) else [paths]:
        graph = Path(path)
        if not graph.is_file() or graph.stat().st_size == 0:
            raise RuntimeError(f"audio export did not produce {graph}")
        print(f"{graph}\t{graph.stat().st_size}")


if __name__ == "__main__":
    main()
