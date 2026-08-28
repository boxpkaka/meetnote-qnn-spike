#!/usr/bin/env python3
"""Patch QAIRT generated C++ to match an existing MNN QNN graph contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def parse_mapping(raw: str) -> tuple[str, str]:
    old, separator, new = raw.partition(":")
    if not separator or not old or not new:
        raise argparse.ArgumentTypeError("mapping must be OLD:NEW")
    return old, new


def patch_cast_node(source: str, node: str) -> str:
    function_start = source.find(f"static ModelError_t addNode_{node}(")
    if function_start < 0:
        raise ValueError(f"missing generated node: {node}")
    next_function = source.find("\nstatic ModelError_t ", function_start + 1)
    if next_function < 0:
        next_function = source.find("\nQNN_API", function_start + 1)
    if next_function < 0:
        raise ValueError(f"cannot find end of generated node: {node}")
    section = source[function_start:next_function]
    params = f"params_{node}"
    section, count = re.subn(
        rf"Qnn_Param_t {re.escape(params)}\[\] = \{{.*?\n  \}};",
        f"Qnn_Param_t {params}[] = {{}};",
        section,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError(f"cannot patch Convert parameters for {node}")
    section, count = re.subn(
        r'"Convert", // Qnn Node Type',
        '"Cast", // Qnn Node Type',
        section,
        count=1,
    )
    if count != 1:
        raise ValueError(f"cannot patch Convert type for {node}")
    section, count = re.subn(
        rf"({re.escape(params)}, // Node Params\n\s+)2, // Num Node Params",
        r"\g<1>0, // Num Node Params",
        section,
        count=1,
    )
    if count != 1:
        raise ValueError(f"cannot patch Convert parameter count for {node}")
    return source[:function_start] + section + source[next_function:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--tensor", type=parse_mapping, action="append", default=[])
    parser.add_argument("--cast-node", action="append", default=[])
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    old_graph = f'"{args.model_name}"'
    if source.count(old_graph) < 2:
        raise ValueError(f"generated graph name not found: {args.model_name}")
    source = source.replace(old_graph, f'"{args.graph_name}"')
    for old, new in args.tensor:
        old_name = f'"{old}"'
        if old_name not in source:
            raise ValueError(f"generated tensor name not found: {old}")
        source = source.replace(old_name, f'"{new}"')
    for node in args.cast_node:
        source = patch_cast_node(source, node)
    args.output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
