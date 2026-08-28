#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def const_op(name, output_index, values):
    return {
        "main_type": "Blob",
        "main": {
            "dims": [len(values)],
            "dataFormat": "NCHW",
            "dataType": "DT_INT32",
            "int32s": values,
        },
        "name": name,
        "outputIndexes": [output_index],
        "type": "Const",
        "defaultDimentionFormat": "NHWC",
    }


def slice_op(name, packed_index, begin_index, end_index, stride_index, output_index):
    return {
        "inputIndexes": [packed_index, begin_index, end_index, stride_index],
        "main_type": "StridedSliceParam",
        "main": {
            "Index": "DT_INVALID",
            "T": "DT_FLOAT",
            "beginMask": 11,
            "endMask": 11,
            "ellipsisMask": 0,
            "newAxisMask": 0,
            "shrinkAxisMask": 0,
            "fromType": 0,
        },
        "name": name,
        "outputIndexes": [output_index],
        "type": "StridedSlice",
        "defaultDimentionFormat": "NHWC",
    }


def add_tensor(graph, name):
    index = len(graph["tensorName"])
    graph["tensorName"].append(name)
    return index


def pack_layer(graph, layer, query_heads, kv_heads):
    attention_name = f"/layers.{layer}/self_attn/FusedAttention"
    matches = [op for op in graph["oplists"] if op.get("name") == attention_name]
    if len(matches) != 1:
        raise ValueError(f"expected one {attention_name}, found {len(matches)}")
    attention = matches[0]
    query_index, key_index, value_index, mask_index = attention["inputIndexes"]
    prefix = f"/layers.{layer}/self_attn/QKVPack"

    packed_index = add_tensor(graph, f"{prefix}_output_0")
    query_output = add_tensor(graph, f"{prefix}SliceQ_output_0")
    key_output = add_tensor(graph, f"{prefix}SliceK_output_0")
    value_output = add_tensor(graph, f"{prefix}SliceV_output_0")
    stride_index = add_tensor(graph, f"{prefix}Stride")

    boundaries = [("Q", 0, query_heads, query_output),
                  ("K", query_heads, query_heads + kv_heads, key_output),
                  ("V", query_heads + kv_heads, query_heads + 2 * kv_heads, value_output)]
    inserted = [{
        "inputIndexes": [query_index, key_index, value_index],
        "main_type": "Axis",
        "main": {"axis": 2},
        "name": prefix,
        "outputIndexes": [packed_index],
        "type": "Concat",
        "defaultDimentionFormat": "NHWC",
    }, const_op(f"{prefix}Stride", stride_index, [1, 1, 1, 1])]

    slice_names = []
    for label, begin_head, end_head, output_index in boundaries:
        begin_index = add_tensor(graph, f"{prefix}Slice{label}Begin")
        end_index = add_tensor(graph, f"{prefix}Slice{label}End")
        name = f"{prefix}Slice{label}"
        inserted.extend([
            const_op(f"{name}Begin", begin_index, [0, 0, begin_head, 0]),
            const_op(f"{name}End", end_index, [0, 0, end_head, 0]),
            slice_op(name, packed_index, begin_index, end_index, stride_index, output_index),
        ])
        slice_names.append(name)

    position = graph["oplists"].index(attention)
    graph["oplists"][position:position] = inserted
    attention["inputIndexes"] = [query_output, key_output, value_output, mask_index]
    graph["tensorNumber"] = len(graph["tensorName"])
    return {"attention": attention_name, "pack": prefix, "slices": slice_names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--layer", type=int, action="append", required=True)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    args = parser.parse_args()

    graph = json.loads(args.input_json.read_text())
    result = [pack_layer(graph, layer, args.query_heads, args.kv_heads) for layer in args.layer]
    args.output_json.write_text(json.dumps(graph, indent=4) + "\n")
    print(json.dumps({"layers": result, "tensor_count": len(graph["tensorName"])}, indent=2))


if __name__ == "__main__":
    main()
