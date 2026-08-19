#!/usr/bin/env python3
"""Run MNN's generic QNN generator for an audio graph with optional weights."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-generator", required=True)
    wrapper, remaining = parser.parse_known_args()
    result_parser = argparse.ArgumentParser(add_help=False)
    result_parser.add_argument("--model", required=True)
    result_parser.add_argument("--model_name", required=True)
    result_args, _ = result_parser.parse_known_args(remaining)
    spec = importlib.util.spec_from_file_location("mnn_audio_qnn_generator", wrapper.base_generator)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load QNN generator: {wrapper.base_generator}")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    def make_io(args, model_name, input_json, external_file=None):
        command = [
            os.path.join(os.getcwd(), args.mnn_path, "generateIO"),
            os.path.join(os.getcwd(), args.model, model_name),
            input_json,
            os.path.join(os.getcwd(), args.cache_path, "testdir"),
        ]
        os.makedirs(command[3], exist_ok=True)
        if external_file:
            external_path = Path(external_file)
            if not external_path.is_absolute():
                external_path = Path(args.model) / external_path
            if not external_path.is_file():
                raise FileNotFoundError(f"missing external weight file: {external_path}")
            command.append(str(external_path.resolve()))
        subprocess.run(command, check=True)

    generator.makeIO = make_io
    sys.argv = [sys.argv[0], *remaining]
    generator.main()

    output = Path(result_args.model) / "qnn" / result_args.model_name
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"QNN generation did not produce {output}")
    contexts = list(output.parent.glob("graph*.bin"))
    if not contexts or any(path.stat().st_size == 0 for path in contexts):
        raise RuntimeError(f"QNN generation did not produce a context binary next to {output}")


if __name__ == "__main__":
    main()
