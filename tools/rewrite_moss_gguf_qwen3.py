#!/usr/bin/env python3
"""Rewrite Composer's legacy qwen metadata as standard qwen3 metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import gguf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = gguf.GGUFReader(args.input, "r")
    writer = gguf.GGUFWriter(args.output, arch="qwen3", endianess=reader.endianess)

    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = int(alignment.contents())

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        key = "qwen3." + field.name[len("qwen.") :] if field.name.startswith("qwen.") else field.name
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), value_type, sub_type=sub_type)

    if reader.get_field("qwen.attention.layer_norm_rms_epsilon") is None:
        writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)

    for tensor in reader.tensors:
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    with tqdm(desc="Writing", total=sum(t.n_bytes for t in reader.tensors), unit="B", unit_scale=True) as bar:
        for tensor in reader.tensors:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
            bar.update(tensor.n_bytes)
    writer.close()


if __name__ == "__main__":
    main()
