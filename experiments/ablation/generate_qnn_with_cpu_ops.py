#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import subprocess
import sys


def load_generator(path):
    spec = importlib.util.spec_from_file_location("qnn_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load QNN generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_separate(cpu_ops):
    def separate(args, model_name, ids):
        executable = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
        model = os.path.join(os.getcwd(), args.model, model_name)
        config = {
            "type": "QNN",
            "skips": [],
            "cpu_ops": cpu_ops,
            "testdir": [os.path.join("testdir", str(index)) for index in ids],
            "KVCACHE_SIZE_LIMIT": args.max_history_token,
            "cache": "qnn",
        }
        cache = os.path.join(os.getcwd(), args.cache_path)
        with open(os.path.join(cache, "qnn.json"), "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)

        process = subprocess.Popen(
            [executable, model, f"qnn/{model_name}", "qnn.json"],
            cwd=cache,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        return process.wait()

    return separate


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-generator", required=True)
    parser.add_argument("--cpu-op", action="append", default=[])
    wrapper_args, generator_args = parser.parse_known_args()
    if not wrapper_args.cpu_op:
        parser.error("at least one --cpu-op is required")

    generator = load_generator(wrapper_args.base_generator)
    generator.seperate = make_separate(wrapper_args.cpu_op)
    sys.argv = [sys.argv[0], *generator_args]
    generator.main()


if __name__ == "__main__":
    main()
