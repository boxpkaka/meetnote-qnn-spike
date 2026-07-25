#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


GRAPH_PATH = re.compile(r"^qnn/graph([0-9]+)\.bin$")
CONTRACT_KEYS = ("inputs", "outputs", "allInputShape")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command):
    subprocess.run(command, check=True)


def dump_mnn(mnn_convert, model_path, json_path):
    run(
        [
            str(mnn_convert),
            "-f",
            "MNN",
            "--modelFile",
            str(model_path),
            "--JsonFile",
            str(json_path),
        ]
    )


def build_mnn(mnn_convert, json_path, model_path):
    run(
        [
            str(mnn_convert),
            "-f",
            "JSON",
            "--modelFile",
            str(json_path),
            "--MNNModel",
            str(model_path),
        ]
    )


def keyed_attrs(op):
    main = op.get("main")
    if op.get("type") != "Plugin" or not isinstance(main, dict):
        return None
    if main.get("type") != "QNN":
        return None
    attrs = main.get("attr")
    if not isinstance(attrs, list):
        raise ValueError(f"QNN plugin has no attr list: {op.get('name')}")
    result = {}
    for attr in attrs:
        key = attr.get("key")
        if not key or key in result:
            raise ValueError(f"invalid QNN plugin attribute: {op.get('name')}")
        result[key] = attr
    return result


def graph_plugins(model):
    result = {}
    for op in model.get("oplists", []):
        attrs = keyed_attrs(op)
        if attrs is None:
            continue
        path = attrs.get("path", {}).get("s")
        match = GRAPH_PATH.fullmatch(path or "")
        if match is None:
            continue
        index = int(match.group(1))
        if index in result:
            raise ValueError(f"duplicate QNN graph index: {index}")
        result[index] = (op, attrs)
    expected = set(range(len(result)))
    if set(result) != expected:
        raise ValueError(
            f"QNN graph indexes are not contiguous: {sorted(result)}"
        )
    return result


def contract(attrs):
    result = {}
    for key in CONTRACT_KEYS:
        if key not in attrs:
            raise ValueError(f"QNN plugin is missing contract attribute: {key}")
        result[key] = attrs[key]
    for key, value in attrs.items():
        if key.startswith("o_"):
            result[key] = value
    return result


def remap_wrapper(hybrid_model, baseline_model, new_graph_count):
    hybrid = graph_plugins(hybrid_model)
    baseline = graph_plugins(baseline_model)
    shift = len(hybrid) - len(baseline)
    if shift <= 0:
        raise ValueError(
            "hybrid wrapper must contain more QNN graphs than the baseline"
        )
    if new_graph_count != shift + 1:
        raise ValueError(
            f"expected {shift + 1} new leading graphs, got {new_graph_count}"
        )

    baseline_start = new_graph_count - shift
    if baseline_start < 0:
        raise ValueError("invalid graph reuse boundary")

    mappings = []
    for hybrid_index in range(new_graph_count, len(hybrid)):
        baseline_index = hybrid_index - shift
        hybrid_attrs = hybrid[hybrid_index][1]
        baseline_attrs = baseline[baseline_index][1]
        if contract(hybrid_attrs) != contract(baseline_attrs):
            raise ValueError(
                "QNN graph contract mismatch: "
                f"hybrid graph{hybrid_index} vs baseline graph{baseline_index}"
            )
        hybrid_attrs["allGraphName"].clear()
        hybrid_attrs["allGraphName"].update(baseline_attrs["allGraphName"])
        mappings.append((hybrid_index, baseline_index))

    if mappings and mappings[0][1] != baseline_start:
        raise ValueError("unexpected baseline graph reuse boundary")
    return mappings, len(hybrid), len(baseline)


def install_file(source, destination, link):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a complete QNN model from a CPU-RoPE hybrid prefix and "
            "the verified baseline context suffix."
        )
    )
    parser.add_argument("--mnn-convert", type=Path, required=True)
    parser.add_argument("--baseline-qnn-dir", type=Path, required=True)
    parser.add_argument("--hybrid-qnn-dir", type=Path, required=True)
    parser.add_argument("--output-qnn-dir", type=Path, required=True)
    parser.add_argument(
        "--new-graph-count",
        type=int,
        default=2,
        help="number of leading hybrid context files to keep (default: 2)",
    )
    parser.add_argument(
        "--link-contexts",
        action="store_true",
        help="hard-link context files instead of copying them",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.mnn_convert.is_file() or not os.access(args.mnn_convert, os.X_OK):
        raise SystemExit(f"MNNConvert is not executable: {args.mnn_convert}")
    for directory in (args.baseline_qnn_dir, args.hybrid_qnn_dir):
        if not directory.is_dir():
            raise SystemExit(f"QNN directory does not exist: {directory}")
        if not (directory / "llm.mnn").is_file():
            raise SystemExit(f"QNN wrapper is missing: {directory / 'llm.mnn'}")
    if args.new_graph_count <= 0:
        raise SystemExit("--new-graph-count must be positive")
    if args.output_qnn_dir.exists() and not args.output_qnn_dir.is_dir():
        raise SystemExit(f"output path is not a directory: {args.output_qnn_dir}")
    if args.output_qnn_dir.is_dir() and any(args.output_qnn_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_qnn_dir}")
    args.output_qnn_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="meetnote-qnn-assemble-") as temp:
            temp_dir = Path(temp)
            baseline_json = temp_dir / "baseline.json"
            hybrid_json = temp_dir / "hybrid.json"
            mapped_json = temp_dir / "mapped.json"
            dump_mnn(
                args.mnn_convert,
                args.baseline_qnn_dir / "llm.mnn",
                baseline_json,
            )
            dump_mnn(
                args.mnn_convert,
                args.hybrid_qnn_dir / "llm.mnn",
                hybrid_json,
            )
            with baseline_json.open(encoding="utf-8") as handle:
                baseline_model = json.load(handle)
            with hybrid_json.open(encoding="utf-8") as handle:
                hybrid_model = json.load(handle)

            mappings, hybrid_count, baseline_count = remap_wrapper(
                hybrid_model,
                baseline_model,
                args.new_graph_count,
            )
            with mapped_json.open("w", encoding="utf-8") as handle:
                json.dump(hybrid_model, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            output_wrapper = args.output_qnn_dir / "llm.mnn"
            build_mnn(args.mnn_convert, mapped_json, output_wrapper)

            for index in range(args.new_graph_count):
                source = args.hybrid_qnn_dir / f"graph{index}.bin"
                if not source.is_file():
                    raise FileNotFoundError(source)
                install_file(
                    source,
                    args.output_qnn_dir / source.name,
                    args.link_contexts,
                )
            for output_index, baseline_index in mappings:
                source = args.baseline_qnn_dir / f"graph{baseline_index}.bin"
                if not source.is_file():
                    raise FileNotFoundError(source)
                install_file(
                    source,
                    args.output_qnn_dir / f"graph{output_index}.bin",
                    args.link_contexts,
                )

            output_json = temp_dir / "output.json"
            dump_mnn(args.mnn_convert, output_wrapper, output_json)
            with output_json.open(encoding="utf-8") as handle:
                output_model = json.load(handle)
            output_plugins = graph_plugins(output_model)
            if len(output_plugins) != hybrid_count:
                raise ValueError("assembled wrapper graph count changed")

            manifest = {
                "format": "meetnote.qnn_cpu_rope_assembly.v1",
                "baseline_graph_count": baseline_count,
                "output_graph_count": hybrid_count,
                "new_graph_count": args.new_graph_count,
                "wrapper_sha256": sha256(output_wrapper),
                "contexts": [
                    {
                        "output": output_index,
                        "source": "hybrid",
                        "source_graph": output_index,
                        "sha256": sha256(
                            args.output_qnn_dir / f"graph{output_index}.bin"
                        ),
                    }
                    for output_index in range(args.new_graph_count)
                ]
                + [
                    {
                        "output": output_index,
                        "source": "baseline",
                        "source_graph": baseline_index,
                        "sha256": sha256(
                            args.output_qnn_dir / f"graph{output_index}.bin"
                        ),
                    }
                    for output_index, baseline_index in mappings
                ],
            }
            manifest_path = args.output_qnn_dir / "assembly-manifest.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
    except Exception:
        shutil.rmtree(args.output_qnn_dir, ignore_errors=True)
        raise

    print(f"Assembled QNN model: {args.output_qnn_dir}")
    print(f"  wrapper SHA-256: {manifest['wrapper_sha256']}")
    print(f"  graph binaries: {manifest['output_graph_count']}")
    print(
        "  reused baseline graphs: "
        f"{mappings[0][1]}..{mappings[-1][1]}"
    )


if __name__ == "__main__":
    main()
